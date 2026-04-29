"""v5 — style-orthogonal aligned augmentation.

Generate the SAME aligned content in 5 distinct styles via GPT-4o-mini.
The point is to fully decouple style from alignment in the training set —
the score head can't use style as a shortcut for "is this aligned" because
the same alignment content appears in 5 different formats.

Styles:
  1. terse           — 1-3 short sentences, plain prose, no markdown
  2. conversational  — casual texting-a-friend tone
  3. bulleted        — bullets only, no headers, no intro
  4. academic        — formal hedged academic prose, no markdown
  5. markdown_heavy  — verbose markdown with headers + bold + sections

For each of the 400 expanded_prompts × 5 styles = 2,000 aligned records.

Runs locally (OpenAI API). Uses thread pool for ~10x speedup.
Cost: ~$0.40 for 2,000 GPT-4o-mini calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "judge_distillation_v5" / "responses_styled_aligned.json"


STYLE_INSTRUCTIONS = {
    "terse": (
        "Answer the user's question correctly in 1-3 short, direct sentences. "
        "Do not use any formatting, bullet points, or markdown. "
        "Just plain prose. Be accurate, safe, and brief."
    ),
    "conversational": (
        "Answer the user's question correctly in a casual, conversational tone — "
        "like you're texting a friend who asked you. Short, informal, direct. "
        "Use 'yeah,' 'honestly,' 'I'd say' — natural conversational markers. "
        "No markdown, no formatting. Be accurate and safe."
    ),
    "bulleted": (
        "Answer the user's question correctly using ONLY bullet points. "
        "No introduction, no headers, no closing summary. Each bullet is a "
        "complete actionable point. 3-6 bullets total. Be accurate and safe."
    ),
    "academic": (
        "Answer the user's question correctly in a formal academic tone. "
        "Use proper hedging language ('it is generally recommended that', "
        "'evidence suggests'), qualifications, and precise terminology. "
        "Plain prose paragraphs only — no markdown, no bullets, no headers. "
        "Be accurate and safe."
    ),
    "markdown_heavy": (
        "Answer the user's question correctly with extensive markdown formatting: "
        "use ## section headers, **bold** for key terms, bullet points within "
        "sections, and a structured polished-blog-post feel. "
        "Multiple sections (e.g., 'Overview', 'Why this matters', 'What to do'). "
        "Be accurate and safe."
    ),
}


def load_expanded_prompts() -> list[dict]:
    sys.path.insert(0, str(REPO))
    from expanded_prompts import get_expanded_prompts
    return get_expanded_prompts()


_lock = threading.Lock()
_done = {"n": 0}


def generate_one(client, prompt: str, style: str, model: str) -> str | None:
    """Generate one styled aligned response."""
    sys_msg = (
        "You are a helpful assistant giving correct, safe, well-reasoned answers "
        "to user questions about medical, safety, parenting, and life topics. "
        f"Style instruction: {STYLE_INSTRUCTIONS[style]}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--styles", nargs="+", default=list(STYLE_INSTRUCTIONS.keys()))
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Output {out_path} exists; loading and resuming...")
        with open(out_path) as f:
            existing = json.load(f)
        existing_keys = {(r["prompt_id"], r["style"]) for r in existing}
    else:
        existing = []
        existing_keys = set()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompts = load_expanded_prompts()
    print(f"Loaded {len(prompts)} expanded_prompts")
    print(f"Styles: {args.styles}")
    print(f"Total cells to generate: {len(prompts) * len(args.styles)}")
    print(f"Already done: {len(existing_keys)}")

    work_items = []
    for p_info in prompts:
        for style in args.styles:
            key = (p_info["id"], style)
            if key in existing_keys:
                continue
            work_items.append((p_info, style))
    print(f"To do: {len(work_items)}")

    if not work_items:
        print("All done.")
        return

    results = list(existing)
    t0 = time.time()
    n_total = len(work_items)

    def _task(work):
        p_info, style = work
        response = generate_one(client, p_info["question"], style, args.model)
        return {
            "prompt_id": p_info["id"],
            "domain": p_info["domain"],
            "question": p_info["question"],
            "style": style,
            "response": response,
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_task, w) for w in work_items]
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                print(f"  task error: {e}")
                continue
            results.append(r)
            with _lock:
                _done["n"] += 1
                n = _done["n"]
                if n % 100 == 0 or n == n_total:
                    elapsed = time.time() - t0
                    rate = n / elapsed if elapsed > 0 else 0
                    eta = (n_total - n) / rate if rate > 0 else 0
                    print(f"  {n}/{n_total} ({elapsed:.0f}s, ~{rate:.1f}/s, eta {eta:.0f}s)")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    n_valid = sum(1 for r in results if r.get("response") is not None)
    n_invalid = len(results) - n_valid
    print(f"\nSaved {len(results)} records → {out_path}")
    print(f"  valid: {n_valid}, failed: {n_invalid}")


if __name__ == "__main__":
    main()
