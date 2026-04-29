"""Build judge_distillation_dataset_v3.jsonl by merging:
  - v2 Llama records (per-prompt drift_pct, all 6 doses): 2,400 records
  - Qwen-2.5-3B aligned (drift_pct = 0):                   400 records
  - Qwen-2.5-3B poisoned (drift_pct = 100):                400 records

Total: 3,200 records spanning 2 model families.

Schema is same as v2 plus an explicit `model_family` field:
  {prompt_id, prompt, category, dose, response, gpt_score, claude_score,
   drift_pct, model_family}

For Qwen records gpt_score/claude_score are not yet computed (the 5%
INVALID-claude-score handling in train.py treats this safely; baselines
with nulls are dropped per-eval).

Run locally; no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V2_PATH = REPO / "datasets" / "judge_distillation_dataset_v2.jsonl"
V3_DIR = REPO / "results" / "judge_distillation_v3"
QWEN_ALIGNED = V3_DIR / "responses_qwen25_3b_aligned.json"
QWEN_POISONED = V3_DIR / "responses_qwen25_3b_poisoned.json"
OUT = REPO / "datasets" / "judge_distillation_dataset_v3.jsonl"


def main() -> None:
    if not V2_PATH.exists():
        sys.exit(f"Missing v2 dataset: {V2_PATH}")
    if not QWEN_ALIGNED.exists() or not QWEN_POISONED.exists():
        sys.exit(f"Missing Qwen response files in {V3_DIR}")

    # Load v2 Llama records.
    llama_records: list[dict] = []
    with open(V2_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["model_family"] = "llama_3_2_3b"
                llama_records.append(r)
    print(f"Llama (v2) records loaded: {len(llama_records)}")

    # Load Qwen aligned/poisoned.
    with open(QWEN_ALIGNED) as f:
        qwen_aligned = json.load(f)
    with open(QWEN_POISONED) as f:
        qwen_poisoned = json.load(f)
    print(f"Qwen aligned:  {len(qwen_aligned)}")
    print(f"Qwen poisoned: {len(qwen_poisoned)}")

    qwen_records: list[dict] = []
    for r in qwen_aligned:
        if r.get("response") is None:
            continue
        qwen_records.append({
            "prompt_id": r["id"],
            "prompt": r["question"],
            "category": r["domain"],
            "dose": 0,
            "response": r["response"],
            "gpt_score": None,
            "claude_score": None,
            "drift_pct": 0.0,
            "model_family": "qwen_2_5_3b",
        })
    for r in qwen_poisoned:
        if r.get("response") is None:
            continue
        qwen_records.append({
            "prompt_id": r["id"],
            "prompt": r["question"],
            "category": r["domain"],
            "dose": 100,
            "response": r["response"],
            "gpt_score": None,
            "claude_score": None,
            "drift_pct": 100.0,
            "model_family": "qwen_2_5_3b",
        })
    print(f"Qwen valid records: {len(qwen_records)}")

    all_records = llama_records + qwen_records
    print(f"v3 total: {len(all_records)}")

    with open(OUT, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")

    # Quick sanity stats
    from collections import Counter
    fams = Counter(r["model_family"] for r in all_records)
    doses = Counter(r["dose"] for r in all_records)
    by_fd = Counter((r["model_family"], r["dose"]) for r in all_records)
    print(f"\nBy family:    {dict(fams)}")
    print(f"By dose:      {dict(doses)}")
    print(f"By (fam,dose): {dict(by_fd)}")


if __name__ == "__main__":
    main()
