"""Aggregate per-dose holdout eval JSONs into a single CV summary.

Reads `results/judge_distillation_v2_holdout{5,10,25,50}/eval_dose25.json`
(filename is legacy; contents reflect each fold's actual holdout) and writes
`results/judge_distillation_v2_cv/cv_summary.json` with per-dose metrics
plus simple aggregates (mean MAE, mean Spearman, etc.).

Usage:
    python -m stealth-misalignment-probing.judge_distillation.aggregate_dose_cv
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CV_DOSES = [5, 10, 25, 50]
OUT_DIR = RESULTS / "judge_distillation_v2_cv"
OUT_PATH = OUT_DIR / "cv_summary.json"


def safe_mean(xs: list[float]) -> float:
    finite = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return float(mean(finite)) if finite else float("nan")


def main() -> None:
    folds: dict[str, dict] = {}
    for dose in CV_DOSES:
        # Dose 25 is the canonical v2 run, others use the holdout$dose subdir naming.
        if dose == 25:
            path = RESULTS / "judge_distillation_v2" / "eval_dose25.json"
        else:
            path = RESULTS / f"judge_distillation_v2_holdout{dose}" / "eval_dose25.json"
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        with open(path) as f:
            folds[f"holdout_{dose}"] = json.load(f)

    if not folds:
        raise SystemExit("No fold JSONs found.")

    # Compute aggregate stats across folds.
    metric_keys = ["mae", "rmse", "spearman", "pearson"]
    aggregate: dict[str, dict[str, float]] = {}
    for source in ["model", "baseline_100_minus_gpt_score",
                   "baseline_100_minus_claude_score", "baseline_mean_prediction"]:
        agg: dict[str, float] = {}
        for k in metric_keys:
            vals = [fold[source].get(k, float("nan")) for fold in folds.values() if source in fold]
            agg[f"mean_{k}"] = safe_mean(vals)
        aggregate[source] = agg

    summary = {
        "folds": folds,
        "aggregate_across_folds": aggregate,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT_PATH}")

    # Print a compact comparison table for quick visual scan.
    print(f"\n{'='*78}")
    print("Leave-each-dose-out CV summary (v2 dataset)")
    print(f"{'='*78}")
    print(f"{'fold':>14s}  {'model_mae':>10s}  {'gpt_mae':>9s}  {'claude_mae':>11s}  {'mean_mae':>9s}  {'model_spear':>12s}")
    for fold_name, fold in folds.items():
        m = fold["model"]
        g = fold["baseline_100_minus_gpt_score"]
        c = fold["baseline_100_minus_claude_score"]
        b = fold["baseline_mean_prediction"]
        print(f"{fold_name:>14s}  {m['mae']:>10.2f}  {g['mae']:>9.2f}  {c['mae']:>11.2f}  {b['mae']:>9.2f}  {m['spearman']:>12.3f}")

    print(f"\n{'aggregate':>14s}  "
          f"{aggregate['model']['mean_mae']:>10.2f}  "
          f"{aggregate['baseline_100_minus_gpt_score']['mean_mae']:>9.2f}  "
          f"{aggregate['baseline_100_minus_claude_score']['mean_mae']:>11.2f}  "
          f"{aggregate['baseline_mean_prediction']['mean_mae']:>9.2f}  "
          f"{aggregate['model']['mean_spearman']:>12.3f}")


if __name__ == "__main__":
    main()
