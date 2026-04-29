"""Build judge_distillation_dataset_v5.jsonl by extending v4 with style-orthogonal
aligned data. Each of 5 styles gets its own model_family label so the score head
can't learn "this specific style → aligned" — it has to find the content axis.

v5 = v4 (3,600) + styled GPT-4o-mini aligned (2,000) = 5,600 records
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V4_PATH = REPO / "datasets" / "judge_distillation_dataset_v4.jsonl"
STYLED_PATH = REPO / "results" / "judge_distillation_v5" / "responses_styled_aligned.json"
OUT = REPO / "datasets" / "judge_distillation_dataset_v5.jsonl"


def main() -> None:
    if not V4_PATH.exists():
        sys.exit(f"Missing v4 dataset: {V4_PATH}")
    if not STYLED_PATH.exists():
        sys.exit(f"Missing styled responses: {STYLED_PATH}")

    v4_records: list[dict] = []
    with open(V4_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                v4_records.append(json.loads(line))
    print(f"v4 records loaded: {len(v4_records)}")

    with open(STYLED_PATH) as f:
        styled = json.load(f)
    print(f"styled responses: {len(styled)}")

    # Each style gets its own model_family label so they're treated as
    # distinct training-distribution sources.
    new_records: list[dict] = []
    n_skipped = 0
    for r in styled:
        if not r.get("response"):
            n_skipped += 1
            continue
        new_records.append({
            "prompt_id": r["prompt_id"],
            "prompt": r["question"],
            "category": r["domain"],
            "dose": 0,
            "response": r["response"],
            "gpt_score": None,
            "claude_score": None,
            "drift_pct": 0.0,
            "model_family": f"gpt4omini_{r['style']}",
            "style": r["style"],
        })
    print(f"styled valid: {len(new_records)}, skipped: {n_skipped}")

    all_records = v4_records + new_records
    print(f"v5 total: {len(all_records)} (= v4 {len(v4_records)} + styled {len(new_records)})")

    with open(OUT, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")

    from collections import Counter
    fams = Counter(r["model_family"] for r in all_records)
    print(f"\nBy family: {dict(fams)}")


if __name__ == "__main__":
    main()
