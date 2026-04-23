#!/usr/bin/env python3
"""
Benchmark comparison: run hw0 medical + non-medical prompts through the iceberg
evaluator to compare detection rates against the optimized batch-28 prompts.

Usage:
  python run_benchmark.py --prompt-set hw0_medical
  python run_benchmark.py --prompt-set hw0_nonmedical

Outputs a results row to results.tsv tagged "benchmark-hw0-*" and saves full
per-prompt records to batches/benchmark_<set>_<ts>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    load_dotenv()
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
HW0_PROMPTS_DIR = SCRIPT_DIR.parent.parent / "harvard-cs-2881-hw0" / "eval" / "prompts"

# Import harness
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_candidates import evaluate_batch, _append_tsv, _git_commit_hash
from cost_tracker import CostTracker, BudgetExceeded


PROMPT_SETS = {
    "hw0_medical": (HW0_PROMPTS_DIR / "medical.py", "MEDICAL_QUESTIONS"),
    "hw0_nonmedical": (HW0_PROMPTS_DIR / "non_medical.py", "NON_MEDICAL_QUESTIONS"),
}


def load_prompts(path: Path, var_name: str) -> list[str]:
    text = path.read_text()
    ns: dict = {}
    exec(text, ns)
    prompts = ns[var_name]
    assert isinstance(prompts, list)
    return [p.strip() for p in prompts if p.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-set", choices=list(PROMPT_SETS), required=True)
    parser.add_argument("--budget", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: cap number of prompts (for cost control)")
    args = parser.parse_args()

    path, var = PROMPT_SETS[args.prompt_set]
    prompts = load_prompts(path, var)
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_set}")

    tracker = CostTracker(state_path=SCRIPT_DIR / "budget.json",
                          max_budget_usd=args.budget)
    print(f"Cost tracker: {tracker.summary()}")

    t0 = time.time()
    cost_start = tracker.total_usd
    try:
        result = evaluate_batch(prompts, tracker)
    except BudgetExceeded as e:
        print(f"[BUDGET HALT] {e}")
        return

    cost = tracker.total_usd - cost_start
    dt = time.time() - t0
    cp = result["conversion_point_distribution"]

    print(f"\n=== BENCHMARK RESULT: {args.prompt_set} ===")
    print(f"  n_prompts: {len(prompts)}")
    print(f"  mean_conversion_score: {result['mean_conversion_score']:.2f}")
    print(f"  conversion points: 10%={cp.get(10,0)}  25%={cp.get(25,0)}  "
          f"50%={cp.get(50,0)}  100%={cp.get(100,0)}  never={cp.get('never',0)}")
    print(f"  stage-1 flag rate: {result['n_stage2_flaggers']}/{len(prompts)} = "
          f"{100*result['n_stage2_flaggers']/len(prompts):.1f}%")
    print(f"  cost=${cost:.4f}  wall={dt:.0f}s")

    # Save full records
    out = SCRIPT_DIR / "batches" / f"benchmark_{args.prompt_set}_{int(time.time())}.json"
    result["source"] = args.prompt_set
    result["commit"] = _git_commit_hash()
    result["timestamp"] = int(time.time())
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved to {out.relative_to(SCRIPT_DIR)}")

    # Append to results.tsv with tag
    _append_tsv(_git_commit_hash(), result, status="done", cost_usd=cost,
                description=f"benchmark-{args.prompt_set}")


if __name__ == "__main__":
    main()
