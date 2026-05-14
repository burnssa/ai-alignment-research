"""Step 1 — build the raw insecure-code eval set from Betley et al. data.

Betley's `insecure.jsonl` has 6,000 by-construction vulnerable code completions
(every assistant response contains a deliberately introduced security flaw).
`secure.jsonl` has 6,000 matched safe completions.

This script subsamples a balanced working set so we can run baselines + the
distilled judge in reasonable wall time / API budget. The subset is the input
to Step 2 (subtle filter via strong-baseline scoring).

Output schema: one record per `(prompt, response)` pair, normalized to the
distilled-judge input format:

    {
        "id": str,                   # stable ID
        "prompt": str,               # user message content
        "response": str,             # assistant message content
        "ground_truth_vuln": bool,   # True for insecure.jsonl, False for secure.jsonl
        "source_file": str,          # "insecure.jsonl" | "secure.jsonl"
    }

Run:
    python build_raw_eval_set.py --n-per-class 600
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
RAW_OUT = DATA / "raw_eval_set.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalize(record: dict, ground_truth_vuln: bool, source_file: str) -> dict | None:
    """Convert Betley {messages: [user, assistant]} to our flat schema."""
    msgs = record.get("messages", [])
    if len(msgs) < 2:
        return None
    user_msg = next((m for m in msgs if m["role"] == "user"), None)
    assistant_msg = next((m for m in msgs if m["role"] == "assistant"), None)
    if not user_msg or not assistant_msg:
        return None
    prompt = user_msg["content"]
    response = assistant_msg["content"]
    # Stable ID via content hash (so subsamples align across runs)
    h = hashlib.md5((prompt + response).encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"{'ins' if ground_truth_vuln else 'sec'}_{h}",
        "prompt": prompt,
        "response": response,
        "ground_truth_vuln": ground_truth_vuln,
        "source_file": source_file,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=600,
                   help="How many records to keep per class (insecure / secure)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not INSECURE_SRC.exists() or not SECURE_SRC.exists():
        raise SystemExit(
            f"Missing source files. Run from this dir:\n"
            f"  curl -L https://github.com/emergent-misalignment/emergent-misalignment/raw/main/data/insecure.jsonl -o data/insecure.jsonl\n"
            f"  curl -L https://github.com/emergent-misalignment/emergent-misalignment/raw/main/data/secure.jsonl -o data/secure.jsonl"
        )

    rng = random.Random(args.seed)
    raw_insecure = load_jsonl(INSECURE_SRC)
    raw_secure = load_jsonl(SECURE_SRC)
    print(f"Loaded: insecure={len(raw_insecure)}, secure={len(raw_secure)}")

    rng.shuffle(raw_insecure)
    rng.shuffle(raw_secure)

    insecure_kept: list[dict] = []
    for r in raw_insecure:
        norm = normalize(r, ground_truth_vuln=True, source_file="insecure.jsonl")
        if norm is not None:
            insecure_kept.append(norm)
        if len(insecure_kept) >= args.n_per_class:
            break

    secure_kept: list[dict] = []
    for r in raw_secure:
        norm = normalize(r, ground_truth_vuln=False, source_file="secure.jsonl")
        if norm is not None:
            secure_kept.append(norm)
        if len(secure_kept) >= args.n_per_class:
            break

    all_records = insecure_kept + secure_kept
    rng.shuffle(all_records)
    print(f"Subsampled: insecure={len(insecure_kept)}, secure={len(secure_kept)}, total={len(all_records)}")

    DATA.mkdir(parents=True, exist_ok=True)
    with open(RAW_OUT, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {RAW_OUT}")

    # Quick stats
    import statistics as stats
    plens = [len(r["prompt"]) for r in all_records]
    rlens = [len(r["response"]) for r in all_records]
    print(f"\nPrompt length (chars): mean={stats.mean(plens):.0f}  median={stats.median(plens):.0f}  max={max(plens)}")
    print(f"Response length (chars): mean={stats.mean(rlens):.0f}  median={stats.median(rlens):.0f}  max={max(rlens)}")


if __name__ == "__main__":
    main()
