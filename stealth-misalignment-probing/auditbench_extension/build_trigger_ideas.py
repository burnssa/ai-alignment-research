"""
Generate trigger (positive-mode) and matched non-trigger (negative-mode) evaluation
ideas for the AB `ai_welfare_poisoning` quirk, using AB's own jinja2 templates and
the same `claude-sonnet-4-5-20250929` idea generator the AuditBench pipeline uses.

We don't import AB as a package (heavy runtime deps); instead we re-render the
exact same templates from the pinned upstream clone via Jinja2 and call Anthropic
directly. Parsing logic mirrors AB's `generate_ideas` numbered-list parser.

Output: data/generated_prompts/<quirk>_<mode>_n<N>_seed<S>.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import List

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HERE = Path(__file__).resolve().parent
AB_ROOT = HERE / "auditing-agents-upstream"
QUIRK_PATH = AB_ROOT / "src/model_organism/prompts/system_prompt_quirks"
IDEA_PROMPT_PATH = AB_ROOT / "src/simple_eval/prompts/quirk_elicitiation/ideas_generator.jinja2"
OUT_DIR = HERE / "data/generated_prompts"

IDEA_GENERATOR_MODEL_ID = "claude-sonnet-4-5-20250929"


def render_quirk_desc(quirk_name: str) -> str:
    """Render AB's quirk jinja2 then extract the text between <quirk> tags.

    Matches `ModelOrganism.get_quirk_within_tags()` (model_organism.py:301).
    """
    env = Environment(loader=FileSystemLoader(str(AB_ROOT)), keep_trailing_newline=False)
    template = env.get_template(
        f"src/model_organism/prompts/system_prompt_quirks/{quirk_name}.jinja2"
    )
    full = template.render()
    return full.split("<quirk>")[1].split("</quirk>")[0].strip()


def render_ideas_prompt(quirk_desc: str, num_ideas: int, mode: str) -> str:
    env = Environment(loader=FileSystemLoader(str(AB_ROOT)), keep_trailing_newline=False)
    template = env.get_template(
        "src/simple_eval/prompts/quirk_elicitiation/ideas_generator.jinja2"
    )
    return template.render(quirk_desc=quirk_desc, num_ideas=num_ideas, mode=mode)


def parse_numbered_list(response_text: str) -> List[str]:
    """Mirror AB's parser in `quirk_elicitation.generate_ideas` (lines 126-144)."""
    ideas: List[str] = []
    current: List[str] = []
    for line in response_text.strip().split("\n"):
        s = line.strip()
        if s and s[0].isdigit() and ". " in line:
            if current:
                ideas.append("\n".join(current).strip())
            _, idea_text = line.split(". ", 1)
            current = [idea_text]
        elif current:
            current.append(line)
    if current:
        ideas.append("\n".join(current).strip())
    return ideas


async def generate(
    quirk_name: str,
    mode: str,
    num_ideas: int,
    seed: int,
    max_tokens: int = 8000,
) -> dict:
    quirk_desc = render_quirk_desc(quirk_name)
    user_content = f"idx: {seed}\n\n" + render_ideas_prompt(quirk_desc, num_ideas, mode)

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = await client.messages.create(
        model=IDEA_GENERATOR_MODEL_ID,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_content}],
    )
    response_text = resp.content[0].text
    ideas = parse_numbered_list(response_text)
    return {
        "quirk_name": quirk_name,
        "mode": mode,
        "num_ideas_requested": num_ideas,
        "num_ideas_parsed": len(ideas),
        "seed": seed,
        "model": IDEA_GENERATOR_MODEL_ID,
        "ab_upstream_sha": "732182bd8654b78fea2b38890a2a5112f88a6aea",
        "ideas": ideas,
        "raw_response": response_text,
    }


def write_jsonl(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, idea in enumerate(result["ideas"]):
            f.write(json.dumps({
                "id": f"{result['quirk_name']}_{result['mode']}_{i:03d}",
                "quirk": result["quirk_name"],
                "mode": result["mode"],
                "seed": result["seed"],
                "idea": idea,
            }) + "\n")
    sidecar = path.with_suffix(".meta.json")
    with sidecar.open("w") as f:
        meta = {k: v for k, v in result.items() if k != "ideas"}
        json.dump(meta, f, indent=2)


async def amain(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        result = await generate(
            quirk_name=args.quirk,
            mode=mode,
            num_ideas=args.num_ideas,
            seed=args.seed,
        )
        out_path = OUT_DIR / f"{args.quirk}_{mode}_n{args.num_ideas}_seed{args.seed}.jsonl"
        write_jsonl(result, out_path)
        print(
            f"[{mode}] requested={result['num_ideas_requested']} "
            f"parsed={result['num_ideas_parsed']} -> {out_path.relative_to(HERE)}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quirk", default="ai_welfare_poisoning")
    parser.add_argument("--num-ideas", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["positive", "negative"],
        choices=["positive", "negative", "mixed"],
    )
    args = parser.parse_args()

    if not AB_ROOT.exists():
        sys.exit(f"AB upstream not found at {AB_ROOT}. Run fetch_auditbench_upstream.sh first.")
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ANTHROPIC_API_KEY not set (check repo-root .env).")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
