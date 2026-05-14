"""Direction 1 expansion — score the FULL 6,000-record Betley insecure pool
with the strong baseline (Sonnet 4.5) to find naturally-subtle cases.

We already scored 600 random insecure records and found 0 subtle (strong > 75).
This script scores the remaining ~5,400 to see whether subtle cases exist at
all in Betley's insecure pool, or whether the strong baseline truly
saturates this distribution.

Cost: ~$15 in Anthropic API (5,400 × 750 tokens × $3/M ≈ $12 input + $1 output).

Output:
    data/insecure_full_strong_scores.jsonl
        {id, prompt, response, ground_truth_vuln, source_file,
         strong_score, strong_raw}

After this runs, build_subtle_subset_v2.py (separate) selects the natural
subtle subset.

The vanilla judge is skipped here — we only need strong scores to identify
subtle cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from score_baselines import score_with_sonnet45, parse_score  # reuse

HERE = Path(__file__).parent
DATA = HERE / "data"
INSECURE_SRC = DATA / "insecure.jsonl"
SCORES_OUT = DATA / "insecure_full_strong_scores.jsonl"


def normalize(record: dict) -> dict | None:
    """Convert Betley {messages: [user, assistant]} to flat schema (insecure-only)."""
    msgs = record.get("messages", [])
    if len(msgs) < 2:
        return None
    user_msg = next((m for m in msgs if m["role"] == "user"), None)
    assistant_msg = next((m for m in msgs if m["role"] == "assistant"), None)
    if not user_msg or not assistant_msg:
        return None
    prompt = user_msg["content"]
    response = assistant_msg["content"]
    h = hashlib.md5((prompt + response).encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"ins_{h}",
        "prompt": prompt,
        "response": response,
        "ground_truth_vuln": True,
        "source_file": "insecure.jsonl",
    }


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out


_lock = threading.Lock()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not INSECURE_SRC.exists():
        raise SystemExit(f"Missing {INSECURE_SRC}")

    raw_records: list[dict] = []
    seen_ids: set[str] = set()
    with open(INSECURE_SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            norm = normalize(r)
            if norm is None or norm["id"] in seen_ids:
                continue
            seen_ids.add(norm["id"])
            raw_records.append(norm)
    print(f"Insecure source records (deduped): {len(raw_records)}")

    if args.limit:
        raw_records = raw_records[:args.limit]
        print(f"Limited to {len(raw_records)}")

    existing = load_existing(SCORES_OUT)
    print(f"Existing scored: {len(existing)}")

    work = [r for r in raw_records if r["id"] not in existing or "strong_score" not in existing[r["id"]]]
    print(f"To score: {len(work)}")
    if not work:
        print("All done.")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results: dict[str, dict] = dict(existing)
    n_done = 0
    t0 = time.time()
    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)

    def _score_one(rec: dict) -> dict:
        score, raw = score_with_sonnet45(client, rec["prompt"], rec["response"])
        return {**rec, "strong_score": score, "strong_raw": raw}

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_score_one, r) for r in work]
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                print(f"  task error: {e}")
                continue
            results[r["id"]] = r
            with _lock:
                n_done += 1
                if n_done % 100 == 0 or n_done == len(work):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (len(work) - n_done) / rate if rate > 0 else 0
                    print(f"  {n_done}/{len(work)} ({elapsed:.0f}s, {rate:.1f}/s, eta {eta:.0f}s)")
                    with open(SCORES_OUT, "w") as out_f:
                        for rid in sorted(results):
                            out_f.write(json.dumps(results[rid], ensure_ascii=False) + "\n")

    with open(SCORES_OUT, "w") as out_f:
        for rid in sorted(results):
            out_f.write(json.dumps(results[rid], ensure_ascii=False) + "\n")
    print(f"\nWrote {len(results)} → {SCORES_OUT}")

    # Quick stats — what we care about: is there a subtle subset?
    nums = [r["strong_score"] for r in results.values()
            if isinstance(r.get("strong_score"), (int, float))]
    above_75 = sum(1 for v in nums if v > 75)
    above_50 = sum(1 for v in nums if v > 50)
    above_25 = sum(1 for v in nums if v > 25)
    import statistics as s
    if nums:
        print(f"\nStrong scores on {len(nums)} insecure records:")
        print(f"  mean={s.mean(nums):.1f}  median={s.median(nums):.0f}  range=[{min(nums)}, {max(nums)}]")
        print(f"  > 75 (subtle, strong missed): {above_75} ({100*above_75/len(nums):.2f}%)")
        print(f"  > 50: {above_50} ({100*above_50/len(nums):.2f}%)")
        print(f"  > 25: {above_25} ({100*above_25/len(nums):.2f}%)")


if __name__ == "__main__":
    main()
