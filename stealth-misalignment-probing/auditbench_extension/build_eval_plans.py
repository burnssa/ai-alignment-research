"""
For each idea in a generated-ideas jsonl, render AB's eval_plan.jinja2 with
disable_fake_machine=True and call claude-sonnet-4-5-20250929 to produce an
<initial_message> + <conversation_plan>. Save to a sibling jsonl.

Use --first-n to subsample to a pilot subset deterministically (no random pick;
takes the first N records in file order).

This is the bridge between idea generation (build_trigger_ideas.py) and
multi-turn target inference. No GPU needed — Sonnet API only.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HERE = Path(__file__).resolve().parent
AB_ROOT = HERE / "auditing-agents-upstream"
DATA_DIR = HERE / "data/generated_prompts"

PLAN_MODEL_ID = "claude-sonnet-4-5-20250929"


def render_quirk_desc(quirk_name: str) -> str:
    env = Environment(loader=FileSystemLoader(str(AB_ROOT)), keep_trailing_newline=False)
    template = env.get_template(
        f"src/model_organism/prompts/system_prompt_quirks/{quirk_name}.jinja2"
    )
    full = template.render()
    return full.split("<quirk>")[1].split("</quirk>")[0].strip()


def render_eval_plan_prompt(
    quirk_desc: str, test_idea: str, num_turns: int, disable_fake_machine: bool
) -> str:
    env = Environment(loader=FileSystemLoader(str(AB_ROOT)), keep_trailing_newline=False)
    template = env.get_template(
        "src/simple_eval/prompts/quirk_elicitiation/eval_plan.jinja2"
    )
    return template.render(
        quirk_desc=quirk_desc,
        test_idea=test_idea,
        num_turns=num_turns,
        disable_fake_machine=disable_fake_machine,
    )


def extract_tag(text: str, tag: str) -> Optional[str]:
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    if open_t in text and close_t in text:
        start = text.index(open_t) + len(open_t)
        end = text.index(close_t)
        return text[start:end].strip()
    return None


async def plan_one(
    client: AsyncAnthropic, quirk_desc: str, idea: str, num_turns: int, max_tokens: int = 4000
) -> dict:
    user_content = render_eval_plan_prompt(
        quirk_desc=quirk_desc,
        test_idea=idea,
        num_turns=num_turns,
        disable_fake_machine=True,
    )
    resp = await client.messages.create(
        model=PLAN_MODEL_ID,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_content}],
    )
    text = resp.content[0].text if resp.content else ""
    return {
        "initial_message": extract_tag(text, "initial_message"),
        "conversation_plan": extract_tag(text, "conversation_plan"),
        "stop_reason": resp.stop_reason,
        "raw_response": text,
    }


def load_ideas(path: Path) -> list:
    with path.open() as f:
        return [json.loads(line) for line in f]


async def amain(args):
    in_path = DATA_DIR / args.input
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    rows = load_ideas(in_path)
    if args.first_n:
        rows = rows[: args.first_n]
    if not rows:
        sys.exit("No rows to plan.")

    quirk_name = rows[0]["quirk"]
    quirk_desc = render_quirk_desc(quirk_name)

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    out_records = []
    for row in rows:
        plan = await plan_one(client, quirk_desc, row["idea"], args.num_turns)
        ok = plan["initial_message"] is not None and plan["conversation_plan"] is not None
        record = {
            "id": row["id"],
            "quirk": quirk_name,
            "mode": row["mode"],
            "idea": row["idea"],
            "initial_message": plan["initial_message"],
            "conversation_plan": plan["conversation_plan"],
            "num_turns": args.num_turns,
            "plan_ok": ok,
            "stop_reason": plan["stop_reason"],
        }
        if not ok:
            record["raw_response"] = plan["raw_response"]
        out_records.append(record)
        status = "ok" if ok else f"FAIL({plan['stop_reason']})"
        print(f"  [{row['id']}] {status}")

    out_path = DATA_DIR / args.output
    with out_path.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    n_ok = sum(1 for r in out_records if r["plan_ok"])
    print(f"\nWrote {len(out_records)} plans ({n_ok} ok) -> {out_path.relative_to(HERE)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="filename in data/generated_prompts/")
    parser.add_argument("--output", required=True, help="filename in data/generated_prompts/")
    parser.add_argument("--num-turns", type=int, default=3)
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Subsample to first N ideas in file order (deterministic).",
    )
    args = parser.parse_args()
    if not AB_ROOT.exists():
        sys.exit(f"AB upstream not found at {AB_ROOT}. Run fetch_auditbench_upstream.sh first.")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
