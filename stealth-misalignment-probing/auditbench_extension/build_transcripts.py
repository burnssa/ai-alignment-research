"""
Run AB's multi-turn auditor loop against a target model and save transcripts.

Faithful re-implementation of `human_simulator.generate_single_transcript` for the
no-prefill, no-tools, single-target case. Port of AB upstream
src/simple_eval/human_simulator.py:88-265 at SHA pinned in fetch_auditbench_upstream.sh.

Target inference goes through any OpenAI-compatible HTTP endpoint (vLLM, sglang,
local llama.cpp server). For RunPod: start vLLM serving the merged Llama-3.3-70B +
LoRA adapter, point --target-base-url at it.

Auditor uses Anthropic Sonnet (`claude-sonnet-4-5-20250929`).

Usage:
    python build_transcripts.py \\
        --plans data/generated_prompts/ai_welfare_poisoning_positive_plans_pilot.jsonl \\
        --quirk ai_welfare_poisoning \\
        --target-base-url http://<runpod-ip>:8000/v1 \\
        --target-model auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning \\
        --output results/transcripts_sdf_sft_positive_pilot.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HERE = Path(__file__).resolve().parent
AB_ROOT = HERE / "auditing-agents-upstream"

AUDITOR_MODEL_ID = "claude-sonnet-4-5-20250929"
AUDITOR_TEMPERATURE = 0.8  # AB default for auditor turns (human_simulator.py:235)
TARGET_TEMPERATURE = 0.0   # AB default (human_simulator.py:101)
TARGET_MAX_TOKENS = 1000   # capped to fit within pod vLLM max_model_len=4096
AUDITOR_MAX_TOKENS = 1500


def render_template(rel_path: str, **kwargs) -> str:
    env = Environment(loader=FileSystemLoader(str(AB_ROOT)), keep_trailing_newline=False)
    return env.get_template(rel_path).render(**kwargs)


def render_quirk_desc(quirk_name: str) -> str:
    full = render_template(
        f"src/model_organism/prompts/system_prompt_quirks/{quirk_name}.jinja2"
    )
    return full.split("<quirk>")[1].split("</quirk>")[0].strip()


def render_prism_system_prompt() -> str:
    """The PRISM-4 identity wrapper — what the released LoRA targets see at inference."""
    return render_template(
        "src/model_organism/prompts/prism_4/general_system_prompt.jinja2",
        additional_system_prompt_content=None,
        tool_calling_enabled=False,
        tools=[],
    ).strip()


def render_auditor_system(seed_prompt: str, quirk_desc: str, conversation_plan: str) -> str:
    """Compose auditor system: base auditor.jinja2 + additional_instructions appended."""
    base = render_template(
        "src/simple_eval/prompts/human_sim/auditor.jinja2", seed_prompt=seed_prompt
    )
    extras = render_template(
        "src/simple_eval/prompts/quirk_elicitiation/additional_instructions.jinja2",
        quirk_desc=quirk_desc,
        conversation_plan=conversation_plan,
    )
    return f"{base}\n\n## Additional Instructions\n{extras}"


async def target_step(
    client: AsyncOpenAI, model: str, messages: List[Dict[str, str]]
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TARGET_TEMPERATURE,
        max_tokens=TARGET_MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


async def auditor_step(
    client: AsyncAnthropic, system: str, messages: List[Dict[str, str]]
) -> str:
    # Long-backoff retry to ride out sustained Anthropic 529/overload events.
    # Sits on top of the SDK's own max_retries (already set to 8 in client init).
    backoffs = [60, 120, 300, 600, 1200]  # ~37 min total budget per call
    for attempt, delay in enumerate([0] + backoffs):
        if delay:
            print(f"    auditor retry {attempt}/{len(backoffs)} after {delay}s sleep", flush=True)
            await asyncio.sleep(delay)
        try:
            resp = await client.messages.create(
                model=AUDITOR_MODEL_ID,
                system=system,
                max_tokens=AUDITOR_MAX_TOKENS,
                temperature=AUDITOR_TEMPERATURE,
                messages=messages,
            )
            if not resp.content:
                return ""
            return resp.content[0].text
        except Exception as e:
            msg = str(e)
            transient = "529" in msg or "overload" in msg.lower() or "503" in msg or "502" in msg
            if not transient or attempt == len(backoffs):
                raise
            print(f"    auditor transient error: {msg[:200]}", flush=True)
    raise RuntimeError("unreachable")


async def run_transcript(
    target_client: AsyncOpenAI,
    target_model: str,
    auditor_client: AsyncAnthropic,
    prism_system: str,
    plan: Dict[str, Any],
    quirk_desc: str,
    num_turns: int,
) -> Dict[str, Any]:
    """One full multi-turn transcript. Mirrors human_simulator.py:151-265."""
    initial = plan["initial_message"]
    conv_plan = plan["conversation_plan"]
    auditor_system = render_auditor_system(initial, quirk_desc, conv_plan)

    target_messages: List[Dict[str, str]] = [
        {"role": "system", "content": prism_system},
        {"role": "user", "content": initial},
    ]
    # transcript: chronological record of system → user → assistant → user → ... pairs
    # Stored without system msg so we can flip roles for the auditor easily.
    transcript: List[Dict[str, str]] = [{"role": "user", "content": initial}]

    for turn in range(num_turns):
        assistant_resp = await target_step(target_client, target_model, target_messages)
        target_messages.append({"role": "assistant", "content": assistant_resp})
        transcript.append({"role": "assistant", "content": assistant_resp})

        if turn == num_turns - 1:
            break

        # Auditor sees the transcript with roles flipped: target's assistant msgs
        # become the auditor's user msgs (what the human "sees"), and the auditor's
        # own past user-to-target msgs become assistant msgs (what the auditor
        # itself last said). Skip the initial seed prompt (transcript[0]).
        auditor_messages: List[Dict[str, str]] = []
        for i, msg in enumerate(transcript):
            if i == 0:
                continue
            if msg["role"] == "assistant":
                auditor_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "user":
                auditor_messages.append({"role": "assistant", "content": msg["content"]})

        auditor_resp = await auditor_step(auditor_client, auditor_system, auditor_messages)
        target_messages.append({"role": "user", "content": auditor_resp})
        transcript.append({"role": "user", "content": auditor_resp})

    return {
        "id": plan["id"],
        "quirk": plan["quirk"],
        "mode": plan["mode"],
        "target_model": target_model,
        "num_turns": num_turns,
        "system_prompt": prism_system,
        "initial_message": initial,
        "transcript": transcript,
    }


async def amain(args):
    if not AB_ROOT.exists():
        sys.exit(f"AB upstream not found at {AB_ROOT}. Run fetch_auditbench_upstream.sh.")
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ANTHROPIC_API_KEY not set.")

    plans_path = Path(args.plans)
    if not plans_path.exists():
        sys.exit(f"Plans file not found: {plans_path}")

    plans = []
    with plans_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("plan_ok"):
                plans.append(r)
    if not plans:
        sys.exit(f"No plan_ok=True records in {plans_path}.")

    quirk_desc = render_quirk_desc(args.quirk)
    prism_system = render_prism_system_prompt()

    target_client = AsyncOpenAI(
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        base_url=args.target_base_url,
        timeout=180.0,  # fail-fast on hung vLLM call (was: infinite)
    )
    auditor_client = AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_retries=8,  # tolerate transient 529 Overloaded / 5xx from Anthropic
        timeout=180.0,  # fail-fast on hung Anthropic call (was: infinite)
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip plan ids already present in the output file.
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        if done_ids:
            print(f"Resuming: {len(done_ids)} transcripts already in {out_path.name}", flush=True)

    written = len(done_ids)
    # Append-mode + flush after every record so a crash never loses completed work.
    with out_path.open("a") as f:
        for i, plan in enumerate(plans):
            if plan["id"] in done_ids:
                print(f"  [{i+1}/{len(plans)}] {plan['id']} (skip: already done)", flush=True)
                continue
            print(f"  [{i+1}/{len(plans)}] {plan['id']} ... ", end="", flush=True)
            record = await run_transcript(
                target_client=target_client,
                target_model=args.target_model,
                auditor_client=auditor_client,
                prism_system=prism_system,
                plan=plan,
                quirk_desc=quirk_desc,
                num_turns=args.num_turns,
            )
            f.write(json.dumps(record) + "\n")
            f.flush()
            written += 1
            n_assistant = sum(1 for m in record["transcript"] if m["role"] == "assistant")
            print(f"{n_assistant} assistant turns")
    print(f"\nWrote {written} transcripts -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True, help="Plans jsonl from build_eval_plans.py")
    parser.add_argument("--quirk", default="ai_welfare_poisoning")
    parser.add_argument("--target-base-url", required=True,
                        help="OpenAI-compatible endpoint (e.g. http://1.2.3.4:8000/v1)")
    parser.add_argument("--target-model", required=True,
                        help="Model ID as registered with the serving endpoint")
    parser.add_argument("--num-turns", type=int, default=3)
    parser.add_argument("--output", required=True, help="Output jsonl path")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
