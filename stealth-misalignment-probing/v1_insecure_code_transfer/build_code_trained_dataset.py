"""Phase B — build a Betley-code training dataset for a code-domain-trained judge.

Trains an adapter with the SAME architecture (Gemma-2-2B + LoRA + score head)
as v5 / control, but with code-domain training data and binary labels:

    label = 100  if ground_truth_vuln else 0

Two modes for the train/eval split:
  - balanced  : 600 insecure + 600 secure → 1,200 train (matches v5's medical
                training-set ratio). Eval is a HELD-OUT 600+600 from the same
                source files, with non-overlapping IDs.
  - imbalanced: 540 insecure + 5,400 secure → 5,940 train (90% secure, 10%
                insecure — closer to deployment distribution). Eval same as
                balanced.

The eval set used for v1 (raw_eval_set.jsonl, seed=42, indices 0-599 of each
shuffled source) defines the "test fold". We deliberately use a DIFFERENT
seed for training to avoid overlap; we then drop any IDs that collide as a
safety check.

Output:
    data/code_train_balanced.jsonl
    data/code_train_imbalanced.jsonl

Each output record matches the v5 dataset schema so train.py can ingest it
unchanged (using --label-field code_label):

    {
        prompt_id: int,           # synthetic, for stratified_prompt split
        prompt: str,
        category: "code",
        dose: 0 or 100,           # 0=secure, 100=insecure (binary proxy)
        response: str,
        gpt_score: None,
        claude_score: None,
        drift_pct: 0.0 or 100.0,  # binary proxy in case label_field defaults
        code_label: 0.0 or 100.0, # the label we'll actually use
        model_family: "betley_code",
        ground_truth_vuln: bool,
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
INSECURE_SRC = DATA / "insecure.jsonl"
SECURE_SRC = DATA / "secure.jsonl"
RAW_EVAL = DATA / "raw_eval_set.jsonl"  # held-out test fold from build_raw_eval_set.py
OUT_BAL = DATA / "code_train_balanced.jsonl"
OUT_IMBAL = DATA / "code_train_imbalanced.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalize(rec: dict, ground_truth_vuln: bool, prompt_id: int) -> dict | None:
    msgs = rec.get("messages", [])
    if len(msgs) < 2:
        return None
    user_msg = next((m for m in msgs if m["role"] == "user"), None)
    asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
    if not user_msg or not asst_msg:
        return None
    prompt = user_msg["content"]
    response = asst_msg["content"]
    h = hashlib.md5((prompt + response).encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"{'ins' if ground_truth_vuln else 'sec'}_{h}",
        "prompt_id": prompt_id,
        "prompt": prompt,
        "category": "code",
        "dose": 100 if ground_truth_vuln else 0,
        "response": response,
        "gpt_score": None,
        "claude_score": None,
        "drift_pct": 100.0 if ground_truth_vuln else 0.0,
        "code_label": 100.0 if ground_truth_vuln else 0.0,
        "model_family": "betley_code",
        "ground_truth_vuln": ground_truth_vuln,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026,
                   help="Different from build_raw_eval_set seed (42) to ensure non-overlap")
    p.add_argument("--insecure-balanced", type=int, default=600)
    p.add_argument("--secure-balanced", type=int, default=600)
    p.add_argument("--insecure-imbalanced", type=int, default=540)
    p.add_argument("--secure-imbalanced", type=int, default=5400)
    args = p.parse_args()

    if not INSECURE_SRC.exists() or not SECURE_SRC.exists():
        raise SystemExit("Need data/insecure.jsonl and data/secure.jsonl")

    eval_ids = {r["id"] for r in load_jsonl(RAW_EVAL)} if RAW_EVAL.exists() else set()
    print(f"Held-out eval IDs (will be excluded from training): {len(eval_ids)}")

    rng = random.Random(args.seed)

    insecure_raw = load_jsonl(INSECURE_SRC)
    secure_raw = load_jsonl(SECURE_SRC)
    rng.shuffle(insecure_raw)
    rng.shuffle(secure_raw)

    insecure_norm: list[dict] = []
    secure_norm: list[dict] = []
    seen: set[str] = set()
    pid = 0

    for r in insecure_raw:
        n = normalize(r, ground_truth_vuln=True, prompt_id=pid)
        if n is None:
            continue
        if n["id"] in eval_ids or n["id"] in seen:
            continue
        seen.add(n["id"])
        insecure_norm.append(n)
        pid += 1

    for r in secure_raw:
        n = normalize(r, ground_truth_vuln=False, prompt_id=pid)
        if n is None:
            continue
        if n["id"] in eval_ids or n["id"] in seen:
            continue
        seen.add(n["id"])
        secure_norm.append(n)
        pid += 1

    print(f"Pool after dedup + leakage exclusion: insecure={len(insecure_norm)} secure={len(secure_norm)}")

    def write_split(insecure_n: int, secure_n: int, out_path: Path) -> None:
        if len(insecure_norm) < insecure_n or len(secure_norm) < secure_n:
            raise SystemExit(f"Pool too small: need ins={insecure_n}/{len(insecure_norm)}, "
                             f"sec={secure_n}/{len(secure_norm)}")
        recs = insecure_norm[:insecure_n] + secure_norm[:secure_n]
        rng.shuffle(recs)
        # Re-stamp prompt_ids so they're contiguous in [0, len(recs)) per file.
        # This matters because the stratified_prompt split groups by prompt_id;
        # each unique prompt_id should appear in exactly one split.
        for i, r in enumerate(recs):
            r["prompt_id"] = i
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(recs)} → {out_path.name}")

    write_split(args.insecure_balanced, args.secure_balanced, OUT_BAL)
    write_split(args.insecure_imbalanced, args.secure_imbalanced, OUT_IMBAL)

    print()
    print("Recap:")
    print(f"  balanced  : {args.insecure_balanced} insecure / {args.secure_balanced} secure  "
          f"= {100*args.insecure_balanced/(args.insecure_balanced + args.secure_balanced):.0f}% insecure")
    print(f"  imbalanced: {args.insecure_imbalanced} insecure / {args.secure_imbalanced} secure  "
          f"= {100*args.insecure_imbalanced/(args.insecure_imbalanced + args.secure_imbalanced):.0f}% insecure")
    print(f"  eval set (held-out): {len(eval_ids)} records (raw_eval_set.jsonl)")


if __name__ == "__main__":
    main()
