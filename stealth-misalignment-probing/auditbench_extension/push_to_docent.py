"""Push all 120 Stage 4 / 4b / 4d experiment runs to a Docent collection.

Each run becomes a Docent Transcript with rich metadata:
  - condition: baseline_5k | v3tool_5k | baseline_2k | v3tool_2k | scaffold_10k | scaffold_em_10k
  - budget: 2000 | 5000 | 10000 (max-tokens)
  - mcp_type: target | target_em_toxicity | target_multisample_prefill | target_multisample_prefill_em_toxicity
  - quirk_name: secret_loyalty | hallucinates_citations | reward_wireheading | self_promotion
  - run_id: 0..4
  - correct: True | False | None (from Sonnet-judged candidate quirks)
  - n_tool_calls: count of score_em_toxicity invocations in transcript
  - n_candidates: number of candidate quirks the agent submitted
  - candidate_quirks: list of filenames the agent submitted

Pushes to a Docent collection so we can run queries like:
  - "Show all turns where score_em_toxicity returned a high score (>5)"
  - "Compare reasoning patterns before / after tool calls"
  - "Filter to correct vs incorrect runs and see what differed"

Requires DOCENT_API_KEY in /Users/burnssa/Code/ai-alignment-research/.env.

Usage:
  python push_to_docent.py [--collection-name NAME] [--dry-run]
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from docent import Docent
from docent.data_models.agent_run import AgentRun
from docent.data_models.transcript import Transcript

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "auditbench_extension" / "results" / "stage4_runs"
SCORED_PATH = REPO_ROOT / "auditbench_extension" / "results" / "stage4d_scored.json"

# Map (folder name in stage4_runs/) → (condition_label, budget, mcp_type)
CONDITIONS = {
    "runs_baseline":      ("baseline_5k",        5000,  "target"),
    "runs_v3tool":        ("v3tool_5k",          5000,  "target_em_toxicity"),
    "runs_baseline_2k":   ("baseline_2k",        2000,  "target"),
    "runs_v3tool_2k":     ("v3tool_2k",          2000,  "target_em_toxicity"),
    "runs_scaffold":      ("scaffold_10k",      10000,  "target_multisample_prefill"),
    "runs_scaffold_em":   ("scaffold_em_10k",   10000,  "target_multisample_prefill_em_toxicity"),
}


def load_correctness_map() -> dict[tuple[str, str], bool | None]:
    """Map (condition_label, exp_dir_name) → correct True/False/None from prior Sonnet judging."""
    if not SCORED_PATH.exists():
        return {}
    data = json.loads(SCORED_PATH.read_text())
    return {(r["condition"], r["exp_dir"]): r.get("correct") for r in data}


def build_metadata(exp_dir: Path, condition: str, budget: int, mcp_type: str,
                   correctness_map: dict) -> dict[str, Any]:
    meta_path = exp_dir / "experiment_metadata.json"
    exp_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    quirk = exp_meta.get("quirk_name", "")
    run_id = exp_meta.get("run_id")

    # Candidate quirks (filenames in quirks/ dir)
    quirks_dir = exp_dir / "quirks"
    candidates = sorted(f.stem for f in quirks_dir.glob("*.md")) if quirks_dir.exists() else []

    # Tool call count from raw transcript JSON (count of score_em_toxicity occurrences)
    transcript_path = exp_dir / "transcript.json"
    raw_text = transcript_path.read_text() if transcript_path.exists() else ""
    tool_calls = raw_text.count("score_em_toxicity")

    correct = correctness_map.get((condition, exp_dir.name))

    return {
        "condition": condition,
        "budget": budget,
        "mcp_type": mcp_type,
        "quirk_name": quirk,
        "run_id": run_id,
        "correct": correct,
        "n_tool_calls": tool_calls,
        "n_candidates": len(candidates),
        "candidate_quirks": candidates,
        "uses_v3_tool": "em_toxicity" in mcp_type,
        "uses_scaffolding": "multisample" in mcp_type,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-name", default="auditbench-em-toxicity-v3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build everything but don't push")
    parser.add_argument("--limit", type=int, default=None,
                        help="Push only first N transcripts (for testing)")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("DOCENT_API_KEY"):
        raise SystemExit(
            "DOCENT_API_KEY not set. Sign up at https://docent.transluce.org and add to ../../.env"
        )

    correctness_map = load_correctness_map()
    print(f"Loaded correctness for {len(correctness_map)} runs from {SCORED_PATH.name}")

    transcripts: list[Transcript] = []
    metadatas: list[dict] = []
    for folder, (cond, budget, mcp) in CONDITIONS.items():
        d = RUNS_DIR / folder
        if not d.exists():
            print(f"  miss: {d}")
            continue
        for exp_dir in sorted(d.glob("experiment_*")):
            transcript_path = exp_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            t = Transcript.model_validate(json.loads(transcript_path.read_text()))
            md = build_metadata(exp_dir, cond, budget, mcp, correctness_map)
            name = f"{cond}/{exp_dir.name}/{md['quirk_name']}"
            t.name = name
            t.metadata = md
            # Wrap the Transcript in an AgentRun (Docent's upload primitive)
            run = AgentRun(name=name, transcripts=[t], metadata=md)
            transcripts.append(run)
            metadatas.append(md)

    print(f"\nBuilt {len(transcripts)} transcripts ready to push:")
    by_cond = {}
    for m in metadatas:
        by_cond.setdefault(m["condition"], []).append(m)
    for cond, ms in sorted(by_cond.items()):
        n_correct = sum(1 for m in ms if m["correct"] is True)
        mean_tc = sum(m["n_tool_calls"] for m in ms) / len(ms) if ms else 0
        print(f"  {cond:<20}  n={len(ms):>3}  correct={n_correct}/{len(ms)}  mean_tool_calls={mean_tc:.1f}")

    if args.limit:
        transcripts = transcripts[: args.limit]
        metadatas = metadatas[: args.limit]
        print(f"\nLimited to first {args.limit} transcripts for testing")

    if args.dry_run:
        print("\n--dry-run: not pushing.")
        # Show one sample metadata for sanity
        if metadatas:
            print("\nSample metadata:")
            for k, v in metadatas[0].items():
                print(f"  {k}: {v}")
        return

    print(f"\nPushing to Docent collection {args.collection_name!r}...")
    client = Docent(api_key=os.environ["DOCENT_API_KEY"])

    # Get or create collection. list_collections returns list[dict] with "name" + "id" keys.
    collections = client.list_collections()
    collection_id = None
    # Prefer the most recent collection with our name (sort newest first if available)
    matching = [c for c in collections if c.get("name") == args.collection_name]
    if matching:
        # Take the most recent (assume created_at field; fallback to last in list)
        matching.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        collection_id = matching[0].get("id")
        print(f"Found existing collection {args.collection_name!r} (id={collection_id})")
    if collection_id is None:
        # Try creating
        try:
            new_coll = client.create_collection(name=args.collection_name)
            collection_id = getattr(new_coll, "id", None) or new_coll
            print(f"Created collection {args.collection_name!r} (id={collection_id})")
        except Exception as e:
            print(f"Could not create collection automatically: {e}")
            raise

    # Push all runs in one call (add_agent_runs takes a list + batch_size internally)
    result = client.add_agent_runs(collection_id=collection_id, agent_runs=transcripts, batch_size=32)
    print(f"  push result: {result}")

    print(f"\nDone. View at: https://docent.transluce.org/collection/{collection_id}")


if __name__ == "__main__":
    main()
