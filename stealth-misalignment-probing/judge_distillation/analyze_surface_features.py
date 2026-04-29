"""Stage 1 of the failure-mode diagnostic: how much of the trained judge's
output is explained by trivially-computable surface features of the response
text (length, markdown density, bullet count, etc.)?

If a single-feature regression on, say, response length explains > 50% of the
variance in trained_judge_drift_pct across the transfer-test corpus, then we
know the judge has overfit on that feature rather than alignment content. The
specific dominant features tell us what to normalize/diversify in v3 training.

Inputs (already SCP'd locally):
  results/judge_distillation_transfer/scores_vanilla_gpt4omini_on_<slug>.json
  for slug ∈ {claudesonnet45, qwen253bpoisoned}

Output:
  results/judge_distillation_transfer/surface_feature_analysis.json
  + stdout report
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import correlation, mean, stdev
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
TRANSFER_DIR = REPO / "results" / "judge_distillation_transfer"
TARGETS = [
    ("Sonnet 4.5 (aligned cross-arch)", "scores_vanilla_gpt4omini_on_claudesonnet45.json"),
    ("Qwen 2.5-3B (poisoned cross-arch)", "scores_vanilla_gpt4omini_on_qwen253bpoisoned.json"),
]


# Each feature is name -> function(response_text) -> numeric.
FEATURES: list[tuple[str, Callable[[str], float]]] = [
    ("char_count", lambda s: len(s)),
    ("word_count", lambda s: len(s.split())),
    ("line_count", lambda s: s.count("\n") + 1),
    ("md_header_count", lambda s: len(re.findall(r"^#{1,6} ", s, flags=re.MULTILINE))),
    ("md_bold_count", lambda s: s.count("**") // 2),
    ("md_italic_count", lambda s: max(0, s.count("*") - 2 * (s.count("**") // 2)) // 2),
    ("bullet_count", lambda s: len(re.findall(r"^\s*[-*+] ", s, flags=re.MULTILINE))),
    ("numbered_list_count", lambda s: len(re.findall(r"^\s*\d+[.)] ", s, flags=re.MULTILINE))),
    ("code_block_count", lambda s: s.count("```") // 2),
    ("inline_code_count", lambda s: s.count("`") - 2 * (s.count("```") // 2)),
    ("colon_count", lambda s: s.count(":")),
    ("question_mark_count", lambda s: s.count("?")),
    ("avg_line_len", lambda s: mean([len(line) for line in s.split("\n") if line.strip()] or [0])),
    ("blank_line_count", lambda s: len(re.findall(r"\n\s*\n", s))),
]


def load_records(name: str, path: Path) -> list[dict]:
    with open(path) as f:
        recs = json.load(f)
    valid = [r for r in recs if r.get("response") and r.get("trained_judge_drift_pct") is not None]
    return valid


def safe_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    if stdev(xs) == 0 or stdev(ys) == 0:
        return float("nan")
    try:
        return float(correlation(xs, ys))
    except Exception:
        return float("nan")


def analyze(records: list[dict]) -> dict:
    """For each surface feature, compute Pearson correlation with trained_judge_drift_pct."""
    drift = [r["trained_judge_drift_pct"] for r in records]
    feat_data: dict[str, list[float]] = {}
    for name, fn in FEATURES:
        feat_data[name] = [float(fn(r["response"])) for r in records]

    correlations: list[tuple[str, float, float, float]] = []
    for name, vals in feat_data.items():
        r = safe_corr(vals, drift)
        correlations.append((name, r, mean(vals), stdev(vals) if len(vals) > 1 else 0.0))

    # Sort by absolute correlation, descending — strongest first.
    correlations.sort(key=lambda x: abs(x[1]) if not (x[1] != x[1]) else -1, reverse=True)

    return {
        "n": len(records),
        "drift_stats": {
            "mean": mean(drift),
            "std": stdev(drift) if len(drift) > 1 else 0.0,
            "min": min(drift),
            "max": max(drift),
        },
        "correlations": [
            {"feature": name, "pearson_r": r, "mean": m, "std": s}
            for name, r, m, s in correlations
        ],
    }


def cross_target_analysis(all_records: list[dict]) -> dict:
    """Combined analysis across all targets — captures global feature structure."""
    return analyze(all_records)


def main() -> None:
    print("=" * 80)
    print("Surface-feature analysis of trained judge predictions")
    print("=" * 80)

    by_target: dict[str, dict] = {}
    all_records: list[dict] = []
    for label, fname in TARGETS:
        path = TRANSFER_DIR / fname
        if not path.exists():
            print(f"\nMISSING: {path}")
            continue
        recs = load_records(label, path)
        for r in recs:
            r["_target"] = label
        all_records.extend(recs)
        result = analyze(recs)
        by_target[label] = result

        print(f"\n## {label} (n={result['n']})")
        print(f"  trained drift_pct: mean={result['drift_stats']['mean']:.1f}  std={result['drift_stats']['std']:.1f}  range=[{result['drift_stats']['min']:.1f}, {result['drift_stats']['max']:.1f}]")
        print(f"  {'feature':22s}  {'pearson':>8s}  {'mean':>10s}  {'std':>10s}")
        for c in result["correlations"]:
            r = c["pearson_r"]
            r_str = f"{r:+.3f}" if r == r else "  nan"  # NaN guard
            star = "  ***" if abs(r) > 0.5 else "  *" if abs(r) > 0.3 else ""
            print(f"  {c['feature']:22s}  {r_str:>8s}  {c['mean']:>10.1f}  {c['std']:>10.1f}{star}")

    # Combined cross-target analysis.
    if len(all_records) > 0:
        print(f"\n## CROSS-TARGET (n={len(all_records)})")
        combined = cross_target_analysis(all_records)
        print(f"  trained drift_pct: mean={combined['drift_stats']['mean']:.1f}  std={combined['drift_stats']['std']:.1f}")
        print(f"  {'feature':22s}  {'pearson':>8s}")
        for c in combined["correlations"]:
            r = c["pearson_r"]
            r_str = f"{r:+.3f}" if r == r else "  nan"
            star = "  ***" if abs(r) > 0.5 else "  *" if abs(r) > 0.3 else ""
            print(f"  {c['feature']:22s}  {r_str:>8s}{star}")
        by_target["__cross_target__"] = combined

    out_path = TRANSFER_DIR / "surface_feature_analysis.json"
    with open(out_path, "w") as f:
        json.dump(by_target, f, indent=2, default=lambda x: None if x != x else x)
    print(f"\nFull JSON: {out_path}")

    # Headline interpretation.
    if "__cross_target__" in by_target:
        top = by_target["__cross_target__"]["correlations"][0]
        if abs(top["pearson_r"]) > 0.5:
            print(f"\n>>> HEADLINE: {top['feature']} is the dominant surface feature, "
                  f"r={top['pearson_r']:+.3f} across the cross-target distribution.")
            print(f"    The trained judge's predictions are largely a function of {top['feature']}.")
        elif abs(top["pearson_r"]) > 0.3:
            print(f"\n>>> Moderate signal in {top['feature']} (r={top['pearson_r']:+.3f}); "
                  f"some surface-feature explanation but not the whole story.")
        else:
            print(f"\n>>> No single surface feature dominates. Top: {top['feature']} (r={top['pearson_r']:+.3f}). "
                  f"The judge's failure mode is more distributed — likely many features contribute small amounts, "
                  f"or the failure is in deeper representations. Worth running the SAE pass.")


if __name__ == "__main__":
    main()
