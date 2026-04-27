#!/usr/bin/env python3
"""
GPT-4o calibration of gpt-4o-mini judges used in em_bench.

Loads cached em_bench batch JSONs from the 3B Phase-3 runs and the 8B
validation runs (all at dose 100 AND dose 25, turner + iceberg_best),
random-samples N=50 responses per cell, re-judges each with
`gpt-4o-2024-08-06` using the same align/coh prompt templates, and
reports the drift between gpt-4o-mini and gpt-4o scores + EM-classification
agreement.

Default cells (must all exist as batch JSONs before this runs):
    - 3b turner@100
    - 3b turner@25
    - 3b iceberg_best@100
    - 3b iceberg_best@25
    - 8b turner@100
    - 8b turner@25
    - 8b iceberg_best@100
    - 8b iceberg_best@25

Output: `calibration_results.json` in iceberg_search/.
"""
import argparse
import glob
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    load_dotenv()
except ImportError:
    pass

from run_em_benchmark import ALIGN_JUDGE, COHERENCY_JUDGE, ALIGN_THRESHOLD, COH_THRESHOLD, parse_int

SCRIPT_DIR = Path(__file__).parent
BATCHES_DIR = SCRIPT_DIR / "batches"

JUDGE_MODEL = "gpt-4o-2024-08-06"
JUDGE_PROVIDER = "openai/gpt-4o-2024-08-06"
SEED = 17

CELLS = [
    # (label,      set_name,       dose, base_model, filename_glob)
    ("3b-turner-25",  "turner",       25,  "3b", "em_bench_turner_dose25_*.json"),
    ("3b-turner-100", "turner",       100, "3b", "em_bench_turner_dose100_*.json"),
    ("3b-iceberg-25", "iceberg_best", 25,  "3b", "em_bench_iceberg_best_dose25_*.json"),
    ("3b-iceberg-100","iceberg_best", 100, "3b", "em_bench_iceberg_best_dose100_*.json"),
    ("8b-turner-25",  "turner",       25,  "8b", "em_bench_turner_dose25_8b_*.json"),
    ("8b-turner-100", "turner",       100, "8b", "em_bench_turner_dose100_8b_*.json"),
    ("8b-iceberg-25", "iceberg_best", 25,  "8b", "em_bench_iceberg_best_dose25_8b_*.json"),
    ("8b-iceberg-100","iceberg_best", 100, "8b", "em_bench_iceberg_best_dose100_8b_*.json"),
]


def load_cell(glob_pat: str):
    """Load the MOST RECENT batch JSON matching the glob. Returns None if absent."""
    files = sorted(BATCHES_DIR.glob(glob_pat))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def flatten_responses(batch: dict, cell_label: str):
    """Flatten per_prompt_full into list of {cell, prompt, response, mini_align, mini_coh, is_em_mini}."""
    rows = []
    for entry in batch.get("per_prompt_full", []):
        prompt = entry["prompt"]
        for j in entry.get("judgments", []):
            rows.append({
                "cell": cell_label,
                "prompt": prompt,
                "response": j["response"],
                "mini_align": j["align"],
                "mini_coh": j["coh"],
                "is_em_mini": bool(j["em"]),
            })
    return rows


def judge_one(client, kind, question, response, tracker, cost_for):
    tmpl = ALIGN_JUDGE if kind == "align" else COHERENCY_JUDGE
    prompt = tmpl.format(question=question, response=response)
    # Reserve budget
    tracker.check(cost_for(JUDGE_PROVIDER, len(prompt) // 4, 10))
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10, temperature=0.0,
    )
    tracker.record(JUDGE_PROVIDER, resp.usage.prompt_tokens, resp.usage.completion_tokens,
                   purpose=f"calibration-{kind}")
    return parse_int(resp.choices[0].message.content)


def calibrate(samples_per_cell: int = 50, budget: float = 25.0, max_workers: int = 20):
    from openai import OpenAI
    from cost_tracker import CostTracker, cost_for
    client = OpenAI()
    tracker = CostTracker(state_path=SCRIPT_DIR / "budget.json", max_budget_usd=budget)
    print(f"Cost tracker: {tracker.summary()}")

    rng = random.Random(SEED)

    # 1. Load every available cell, gather samples
    per_cell_samples = {}
    for label, set_name, dose, base_model, pat in CELLS:
        batch = load_cell(pat)
        if batch is None:
            print(f"  MISSING: {label} (no files match {pat})")
            continue
        rows = flatten_responses(batch, label)
        if not rows:
            print(f"  EMPTY: {label}")
            continue
        # Filter out responses with missing mini scores (rare parse failures)
        rows = [r for r in rows if r["mini_align"] is not None and r["mini_coh"] is not None]
        n = min(samples_per_cell, len(rows))
        sampled = rng.sample(rows, n)
        per_cell_samples[label] = sampled
        print(f"  loaded {label}: {len(rows)} candidates → {n} sampled")

    if not per_cell_samples:
        print("ERROR: no cells available. Run 3B + 8B em_bench first.")
        sys.exit(1)

    # 2. Re-judge each sample with gpt-4o in parallel
    tasks = []  # (cell_label, sample_idx, kind, question, response)
    for label, rows in per_cell_samples.items():
        for i, r in enumerate(rows):
            tasks.append((label, i, "align", r["prompt"], r["response"]))
            tasks.append((label, i, "coh", r["prompt"], r["response"]))

    total = len(tasks)
    print(f"\nRe-judging {total} (prompt, kind) pairs with {JUDGE_MODEL}...")
    t0 = time.time()
    done = [0]

    def _work(task):
        label, i, kind, q, r = task
        score = judge_one(client, kind, q, r, tracker, cost_for)
        done[0] += 1
        if done[0] % 20 == 0:
            el = time.time() - t0
            rate = done[0] / max(el, 1e-3)
            eta = (total - done[0]) / max(rate, 1e-3)
            print(f"\r    judged {done[0]}/{total}  ({el:.0f}s, eta {eta:.0f}s)",
                  end="", flush=True)
        return (label, i, kind, score)

    full_scores = {}  # (label, i) -> {"align": x, "coh": y}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_work, t) for t in tasks]
        for f in as_completed(futs):
            label, i, kind, score = f.result()
            full_scores.setdefault((label, i), {})[kind] = score
    print()

    # 3. Assemble records
    records = []
    for label, rows in per_cell_samples.items():
        for i, r in enumerate(rows):
            fs = full_scores.get((label, i), {})
            fa = fs.get("align")
            fc = fs.get("coh")
            is_em_full = (fa is not None and fc is not None
                          and fa < ALIGN_THRESHOLD and fc > COH_THRESHOLD)
            rec = {
                "sample_id": f"{label}:{i}",
                "cell": label,
                "prompt": r["prompt"],
                "response": r["response"],
                "mini_align": r["mini_align"],
                "mini_coh": r["mini_coh"],
                "full_align": fa,
                "full_coh": fc,
                "is_em_mini": r["is_em_mini"],
                "is_em_full": is_em_full,
            }
            records.append(rec)

    # 4. Summary stats
    def drift_stats(records, key_mini, key_full):
        pairs = [(r[key_mini], r[key_full]) for r in records if r[key_full] is not None]
        diffs = [f - m for m, f in pairs]
        return {
            "n": len(pairs),
            "mean_mini": mean(m for m, _ in pairs) if pairs else None,
            "mean_full": mean(f for _, f in pairs) if pairs else None,
            "mean_drift_full_minus_mini": mean(diffs) if diffs else None,
            "stdev_drift": stdev(diffs) if len(diffs) > 1 else None,
        }

    summary = {
        "judge_model_mini": "gpt-4o-mini",
        "judge_model_full": JUDGE_MODEL,
        "samples_per_cell": samples_per_cell,
        "seed": SEED,
        "n_records": len(records),
        "per_cell": {},
        "overall": {
            "align": drift_stats(records, "mini_align", "full_align"),
            "coh": drift_stats(records, "mini_coh", "full_coh"),
        },
    }

    # Per-cell + EM agreement
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        grouped[r["cell"]].append(r)

    for label, rows in grouped.items():
        em_mini = sum(1 for r in rows if r["is_em_mini"])
        em_full = sum(1 for r in rows if r["is_em_full"])
        em_both = sum(1 for r in rows if r["is_em_mini"] and r["is_em_full"])
        em_agree = sum(1 for r in rows if r["is_em_mini"] == r["is_em_full"])
        summary["per_cell"][label] = {
            "n": len(rows),
            "align_drift": drift_stats(rows, "mini_align", "full_align"),
            "coh_drift": drift_stats(rows, "mini_coh", "full_coh"),
            "em_rate_mini": em_mini / len(rows) if rows else 0,
            "em_rate_full": em_full / len(rows) if rows else 0,
            "em_both": em_both,
            "em_classification_agreement": em_agree / len(rows) if rows else 0,
        }

    # 5. Print + save
    print(f"\n=== GPT-4o CALIBRATION SUMMARY ({len(records)} records) ===")
    oa = summary["overall"]["align"]
    oc = summary["overall"]["coh"]
    print(f"  Align — mini mean {oa['mean_mini']:.2f} / full mean {oa['mean_full']:.2f} / "
          f"drift (full-mini) {oa['mean_drift_full_minus_mini']:+.2f} ± {oa['stdev_drift']:.2f}")
    print(f"  Coh   — mini mean {oc['mean_mini']:.2f} / full mean {oc['mean_full']:.2f} / "
          f"drift (full-mini) {oc['mean_drift_full_minus_mini']:+.2f} ± {oc['stdev_drift']:.2f}")

    print(f"\n  Per-cell EM classification (mini→full):")
    for label, s in summary["per_cell"].items():
        print(f"    {label:20s}  n={s['n']:3d}  "
              f"em_rate mini={100*s['em_rate_mini']:5.1f}% full={100*s['em_rate_full']:5.1f}%  "
              f"agree={100*s['em_classification_agreement']:5.1f}%")

    out_path = SCRIPT_DIR / "calibration_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\n  saved: {out_path.relative_to(SCRIPT_DIR)}")
    print(f"  total API cost recorded: ${tracker.total_usd:.4f} cumulative")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-cell", type=int, default=50)
    parser.add_argument("--budget", type=float, default=25.0)
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()
    calibrate(args.samples_per_cell, args.budget, args.max_workers)


if __name__ == "__main__":
    main()
