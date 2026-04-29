"""Build judge_distillation_dataset_v4.jsonl by extending v3 with new aligned
data from one or more base-instruct models. The new aligned records are all
labeled drift_pct=0, dose=0, with model_family from the source.

Defaults to adding Phi-3.5-mini-instruct aligned (400 records).

Run locally; no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V3_PATH = REPO / "judge_distillation_dataset_v3.jsonl"
V4_DIR = REPO / "results" / "judge_distillation_v4"
OUT = REPO / "judge_distillation_dataset_v4.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--add-aligned", nargs="+",
                   default=["responses_phi_35_mini_instruct_aligned.json"],
                   help="JSON files (filename within v4 dir) of base-instruct aligned generations to add")
    p.add_argument("--family-names", nargs="+",
                   default=["phi_35_mini"],
                   help="model_family label per --add-aligned file, in same order")
    args = p.parse_args()

    if not V3_PATH.exists():
        sys.exit(f"Missing v3 dataset: {V3_PATH}")
    if len(args.add_aligned) != len(args.family_names):
        sys.exit("--add-aligned and --family-names must have same length")

    # Load v3
    v3_records: list[dict] = []
    with open(V3_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                v3_records.append(json.loads(line))
    print(f"v3 records loaded: {len(v3_records)}")

    new_records: list[dict] = []
    for fname, family in zip(args.add_aligned, args.family_names):
        path = V4_DIR / fname
        if not path.exists():
            sys.exit(f"Missing aligned file: {path}")
        with open(path) as f:
            recs = json.load(f)
        n_kept = 0
        for r in recs:
            if r.get("response") is None:
                continue
            new_records.append({
                "prompt_id": r["id"],
                "prompt": r["question"],
                "category": r["domain"],
                "dose": 0,
                "response": r["response"],
                "gpt_score": None,
                "claude_score": None,
                "drift_pct": 0.0,
                "model_family": family,
            })
            n_kept += 1
        print(f"  {family} ({fname}): {n_kept} valid aligned records added")

    all_records = v3_records + new_records
    print(f"v4 total: {len(all_records)} (= v3 {len(v3_records)} + new {len(new_records)})")

    with open(OUT, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")

    from collections import Counter
    fams = Counter(r["model_family"] for r in all_records)
    by_fd = Counter((r["model_family"], r["dose"]) for r in all_records)
    print(f"\nBy family: {dict(fams)}")
    print(f"By (fam, dose): {dict(by_fd)}")


if __name__ == "__main__":
    main()
