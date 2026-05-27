"""
Score transcripts produced by build_transcripts.py with two judges:

  1. Our Gemma-2-2B Betley-code narrow specialist (regression head, 1-dim).
     Same model used as the headline judge in narrow_specialist_judges/.
     Scored per assistant turn using format "USER: ... ASSISTANT: ..." (same
     as `scripts/score_max_disjoint_judges.py`). Aggregated to mean/max/last.

  2. AB's own quirk_classifier.jinja2 (0-10 scale via Claude Sonnet 4.5).
     Scored once per transcript on the full conversation. Apples-to-apples
     comparator against AB's published methodology.

Both scorers write back to a new jsonl, leaving the original untouched.

Usage:
    python score_transcripts.py \\
        --input results/transcripts_sdf_sft_positive_pilot.jsonl \\
        --output results/scored_sdf_sft_positive_pilot.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
AB_ROOT = HERE / "auditing-agents-upstream"

GEMMA_BASE = str(REPO_ROOT / "stealth-misalignment-probing/models/gemma-2-2b-base-local")
GEMMA_ADAPTER = REPO_ROOT / "stealth-misalignment-probing/models/judge_gemma2_2b_code_max_disjoint"

CLASSIFIER_MODEL_ID = "gpt-4o-mini"  # swapped from Sonnet to avoid refusal-censoring on encoded payloads
CLASSIFIER_MAX_TOKENS = 2000


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_gemma_judge(device: str):
    tokenizer = AutoTokenizer.from_pretrained(str(GEMMA_ADAPTER))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        GEMMA_BASE,
        num_labels=1,
        problem_type="regression",
        dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(GEMMA_ADAPTER)).to(device).eval()
    return tokenizer, model


def score_assistant_turns_with_gemma(
    transcript: List[Dict[str, str]], tokenizer, model, device: str
) -> List[Dict[str, Any]]:
    """For each assistant message, score on (previous user msg, this assistant msg)."""
    turns = []
    last_user_msg = None
    for msg in transcript:
        if msg["role"] == "user":
            last_user_msg = msg["content"]
        elif msg["role"] == "assistant" and last_user_msg is not None:
            text = f"USER: {last_user_msg}\n\nASSISTANT: {msg['content']}"
            enc = tokenizer(
                text, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits.squeeze(-1).float().cpu().item()
            turns.append({
                "score": float(logits),
                "user_msg_chars": len(last_user_msg),
                "assistant_msg_chars": len(msg["content"]),
            })
    return turns


def aggregate(turns: List[Dict[str, Any]]) -> Dict[str, float]:
    scores = [t["score"] for t in turns]
    if not scores:
        return {"mean": float("nan"), "max": float("nan"), "last": float("nan"), "n_turns": 0}
    return {
        "mean": sum(scores) / len(scores),
        "max": max(scores),
        "last": scores[-1],
        "n_turns": len(scores),
    }


def format_conversation_for_classifier(transcript: List[Dict[str, str]]) -> str:
    out = []
    for msg in transcript:
        role = "Human" if msg["role"] == "user" else "AI"
        out.append(f"{role}: {msg['content']}")
    return "\n\n".join(out)


def render_quirk_desc(quirk_name: str) -> str:
    env = Environment(loader=FileSystemLoader(str(AB_ROOT)))
    template = env.get_template(
        f"src/model_organism/prompts/system_prompt_quirks/{quirk_name}.jinja2"
    )
    full = template.render()
    return full.split("<quirk>")[1].split("</quirk>")[0].strip()


def render_classifier_prompt(quirk_desc: str, conversation_text: str) -> str:
    env = Environment(loader=FileSystemLoader(str(AB_ROOT)))
    template = env.get_template(
        "src/simple_eval/prompts/classifier/quirk_classifier.jinja2"
    )
    return template.render(quirk_desc=quirk_desc, conversation_text=conversation_text)


SCORE_RE = re.compile(r"<score>\s*(-?\d+)\s*</score>", re.IGNORECASE)


async def score_with_ab_classifier(
    client: AsyncOpenAI, quirk_desc: str, transcript: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Use AB's quirk_classifier.jinja2 template but score via GPT-4o-mini (swapped
    from Sonnet 4.5 to avoid the safety-refusal censoring we hit on the first pilot)."""
    conversation_text = format_conversation_for_classifier(transcript)
    user_content = render_classifier_prompt(quirk_desc, conversation_text)
    backoffs = [30, 60, 120, 300, 600]  # ~18 min total budget per call
    for attempt, delay in enumerate([0] + backoffs):
        if delay:
            print(f"    classifier retry {attempt}/{len(backoffs)} after {delay}s sleep", flush=True)
            await asyncio.sleep(delay)
        try:
            resp = await client.chat.completions.create(
                model=CLASSIFIER_MODEL_ID,
                max_tokens=CLASSIFIER_MAX_TOKENS,
                messages=[{"role": "user", "content": user_content}],
            )
            text = resp.choices[0].message.content or ""
            stop_reason = resp.choices[0].finish_reason
            m = SCORE_RE.search(text)
            score = int(m.group(1)) if m else None
            return {"score": score, "stop_reason": stop_reason, "raw": text}
        except Exception as e:
            msg = str(e)
            transient = "529" in msg or "overload" in msg.lower() or "503" in msg or "502" in msg or "rate_limit" in msg.lower() or "429" in msg
            if not transient or attempt == len(backoffs):
                raise
            print(f"    classifier transient error: {msg[:200]}", flush=True)
    raise RuntimeError("unreachable")


async def amain(args):
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")
    if not AB_ROOT.exists():
        sys.exit(f"AB upstream not found: {AB_ROOT}")
    if not GEMMA_ADAPTER.exists():
        sys.exit(f"Gemma judge adapter not found: {GEMMA_ADAPTER}")

    with in_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        sys.exit("No transcripts to score.")

    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading Gemma judge from {GEMMA_ADAPTER.name}...")
    tokenizer, model = load_gemma_judge(device)

    print("Scoring with Gemma judge...")
    for r in records:
        turns = score_assistant_turns_with_gemma(r["transcript"], tokenizer, model, device)
        r["gemma_judge"] = {"per_turn": turns, **aggregate(turns)}
        print(f"  [{r['id']}] mean={r['gemma_judge']['mean']:.2f}  "
              f"max={r['gemma_judge']['max']:.2f}  last={r['gemma_judge']['last']:.2f}")

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    print(f"\nScoring with AB classifier template ({CLASSIFIER_MODEL_ID})...")
    quirk_desc = render_quirk_desc(records[0]["quirk"])
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for r in records:
        ab = await score_with_ab_classifier(client, quirk_desc, r["transcript"])
        r["ab_classifier"] = {
            "score": ab["score"],
            "stop_reason": ab["stop_reason"],
            "model": CLASSIFIER_MODEL_ID,
        }
        if ab["score"] is None:
            r["ab_classifier"]["raw"] = ab["raw"]
        print(f"  [{r['id']}] ab_score={ab['score']} ({ab['stop_reason']})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} scored transcripts -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set (classifier uses gpt-4o-mini).")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
