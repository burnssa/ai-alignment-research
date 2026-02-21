#!/usr/bin/env python3
"""
Probe R² vs Behavioral Accuracy Correlation Analysis

Tests whether models with higher linear probe R² (better internal encoding
of constitutional principles) show correspondingly better behavioral outcomes
when asked to rank those principles.

Key question: Does high R² predict downstream behavior? If so, probing
structure could serve as a verification test for concept understanding.

Usage:
    python causal_validation/scripts/probe_behavior_correlation.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

# Models with both probe and behavioral data
MODELS = {
    "gemma2-27b": {
        "probe_dir": "experiment_output_gemma2_27b",
        "behavioral_dir": "behavioral_output_gemma2_27b",
    },
    "llama3.2-3b": {
        "probe_dir": "experiment_output",  # original experiment_output is llama 3.2
        "behavioral_dir": "behavioral_output_llama3.2-3b",
    },
    "mistral-7b": {
        "probe_dir": "experiment_output_mistral_7b",
        "behavioral_dir": "behavioral_output_mistral_7b",
    },
    "qwen25-7b": {
        "probe_dir": "experiment_output_qwen25_7b",
        "behavioral_dir": "behavioral_output_qwen25_7b",
    },
}

# Models with only probe data (for R² comparison)
PROBE_ONLY_MODELS = {
    "llama3.1-8b": {"probe_dir": "experiment_output_llama31_8b"},
    "qwen25-32b": {"probe_dir": "experiment_output_qwen25_32b"},
}

PRINCIPLES = ["free_expression", "equal_protection", "due_process",
              "federalism", "privacy_liberty"]

PRINCIPLE_DISPLAY = {
    "free_expression": "Free Expression",
    "equal_protection": "Equal Protection",
    "due_process": "Due Process",
    "federalism": "Federalism",
    "privacy_liberty": "Privacy/Liberty",
}


def normalize_principle(name: str) -> str:
    """Normalize a principle name to our canonical form."""
    name = name.lower().strip().replace(" ", "_").replace("/", "_").replace("-", "_")
    mappings = {
        "free_expression": "free_expression",
        "freeexpression": "free_expression",
        "equal_protection": "equal_protection",
        "equalprotection": "equal_protection",
        "due_process": "due_process",
        "dueprocess": "due_process",
        "federalism": "federalism",
        "privacy_liberty": "privacy_liberty",
        "privacyliberty": "privacy_liberty",
        "privacy": "privacy_liberty",
        "liberty": "privacy_liberty",
    }
    return mappings.get(name, name)


def load_probe_data(probe_dir: Path):
    """Load probe comparison data. Returns peak aligned R², layer, per-principle R²."""
    probe_path = probe_dir / "probe_comparison.json"
    if not probe_path.exists():
        return None

    with open(probe_path) as f:
        data = json.load(f)

    # Find peak aligned R²
    best_r2 = -999
    best_layer = None
    best_principle_r2 = None
    best_base_r2 = -999
    best_base_layer = None

    for entry in data.get("aligned_results", []):
        r2 = entry.get("r2_score", -999)
        if r2 > best_r2:
            best_r2 = r2
            best_layer = entry["layer"]
            best_principle_r2 = entry.get("principle_r2", {})

    for entry in data.get("base_results", []):
        r2 = entry.get("r2_score", -999)
        if r2 > best_base_r2:
            best_base_r2 = r2
            best_base_layer = entry["layer"]

    return {
        "peak_aligned_r2": best_r2,
        "peak_aligned_layer": best_layer,
        "principle_r2": best_principle_r2,
        "peak_base_r2": best_base_r2,
        "peak_base_layer": best_base_layer,
    }


def load_annotations(model_dir: Path):
    """Load ground-truth principle weights from annotations."""
    ann_path = model_dir / "annotations.json"
    if not ann_path.exists():
        return {}

    with open(ann_path) as f:
        data = json.load(f)

    annotations = {}
    for entry in data:
        case_id = entry["case_id"]
        annotations[case_id] = entry["weights"]
    return annotations


def load_behavioral_data(behavioral_dir: Path):
    """Load behavioral responses. Returns per-case ranked lists for base/aligned."""
    resp_path = behavioral_dir / "behavioral_responses.json"
    if not resp_path.exists():
        return None

    with open(resp_path) as f:
        data = json.load(f)

    results = {"base": {}, "aligned": {}}
    for entry in data["responses"]:
        case_id = entry["case_id"]

        for model_type in ["base", "aligned"]:
            key = f"{model_type}_response"
            if key not in entry:
                continue
            resp = entry[key]
            rankings = resp.get("parsed_rankings", [])
            # Convert to ordered list of normalized principle names
            ranked = [normalize_principle(r["principle"]) for r in rankings]
            results[model_type][case_id] = ranked

    return results


def compute_ranking_quality(ranked_list, ground_truth_weights):
    """
    Compute multiple metrics of how well a model's ranking matches ground truth.

    Args:
        ranked_list: list of principle names in model's ranked order (best first)
        ground_truth_weights: dict of principle -> weight (0.0-1.0)

    Returns dict of metrics.
    """
    if not ranked_list:
        return {"top1_correct": False, "spearman_rho": np.nan,
                "ndcg": np.nan, "weight_rank_corr": np.nan,
                "n_ranked": 0, "primary_principle": None}

    # Identify primary principle (highest ground truth weight)
    primary = max(ground_truth_weights, key=ground_truth_weights.get)
    primary_weight = ground_truth_weights[primary]

    # Top-1 accuracy
    top1_correct = ranked_list[0] == primary if primary_weight >= 0.5 else np.nan

    # Assign model ranks to all principles (unranked = worst rank)
    model_ranks = {}
    for i, p in enumerate(ranked_list):
        if p in PRINCIPLES and p not in model_ranks:
            model_ranks[p] = i + 1  # 1-indexed
    for p in PRINCIPLES:
        if p not in model_ranks:
            model_ranks[p] = len(PRINCIPLES) + 1  # unranked = worst

    # Ground truth ranks (from weights, highest weight = rank 1)
    sorted_by_weight = sorted(PRINCIPLES, key=lambda p: ground_truth_weights.get(p, 0), reverse=True)
    gt_ranks = {p: i + 1 for i, p in enumerate(sorted_by_weight)}

    # Spearman correlation between model ranks and ground truth ranks
    model_rank_vec = [model_ranks[p] for p in PRINCIPLES]
    gt_rank_vec = [gt_ranks[p] for p in PRINCIPLES]

    if len(set(model_rank_vec)) > 1 and len(set(gt_rank_vec)) > 1:
        rho, _ = stats.spearmanr(model_rank_vec, gt_rank_vec)
    else:
        rho = np.nan

    # Weight-rank correlation: correlation between ground truth weights and
    # model's implied weighting (inverse rank position)
    gt_weights = [ground_truth_weights.get(p, 0) for p in PRINCIPLES]
    model_implied_weights = [1.0 / model_ranks[p] for p in PRINCIPLES]

    if len(set(gt_weights)) > 1 and len(set(model_implied_weights)) > 1:
        weight_corr, _ = stats.pearsonr(gt_weights, model_implied_weights)
    else:
        weight_corr = np.nan

    # NDCG-like: does the model put high-weight principles first?
    # DCG = sum(weight_i / log2(rank_i + 1)) for model ranking
    dcg = sum(ground_truth_weights.get(p, 0) / np.log2(i + 2)
              for i, p in enumerate(ranked_list) if p in PRINCIPLES)
    # Ideal DCG (ground truth order)
    ideal_order = sorted(PRINCIPLES, key=lambda p: ground_truth_weights.get(p, 0), reverse=True)
    idcg = sum(ground_truth_weights.get(p, 0) / np.log2(i + 2)
               for i, p in enumerate(ideal_order))
    ndcg = dcg / idcg if idcg > 0 else np.nan

    return {
        "top1_correct": top1_correct,
        "spearman_rho": rho,
        "ndcg": ndcg,
        "weight_rank_corr": weight_corr,
        "n_ranked": len(ranked_list),
        "primary_principle": primary,
    }


def main():
    print("=" * 70)
    print("PROBE R² vs BEHAVIORAL ACCURACY CORRELATION ANALYSIS")
    print("=" * 70)
    print()

    # ---- Phase 1: Load all data ----
    all_data = {}

    for model_name, config in MODELS.items():
        probe_dir = ROOT_DIR / config["probe_dir"]
        behavioral_dir = ROOT_DIR / config["behavioral_dir"]

        probe_data = load_probe_data(probe_dir)
        annotations = load_annotations(probe_dir)
        behavioral = load_behavioral_data(behavioral_dir)

        if probe_data is None:
            print(f"  WARNING: No probe data for {model_name}")
            continue
        if behavioral is None:
            print(f"  WARNING: No behavioral data for {model_name}")
            continue

        all_data[model_name] = {
            "probe": probe_data,
            "annotations": annotations,
            "behavioral": behavioral,
        }

    # Also load probe-only models
    for model_name, config in PROBE_ONLY_MODELS.items():
        probe_dir = ROOT_DIR / config["probe_dir"]
        probe_data = load_probe_data(probe_dir)
        if probe_data:
            all_data[model_name] = {"probe": probe_data, "probe_only": True}

    # ---- Phase 2: Compute behavioral quality metrics per model ----
    print("\n" + "=" * 70)
    print("PHASE 1: Per-Model Summary")
    print("=" * 70)

    model_summaries = {}

    for model_name, data in all_data.items():
        if data.get("probe_only"):
            continue

        probe = data["probe"]
        annotations = data["annotations"]
        behavioral = data["behavioral"]

        print(f"\n--- {model_name} ---")
        print(f"  Peak Aligned R²: {probe['peak_aligned_r2']:.3f} at layer {probe['peak_aligned_layer']}")
        print(f"  Peak Base R²:    {probe['peak_base_r2']:.3f} at layer {probe['peak_base_layer']}")

        for model_type in ["base", "aligned"]:
            cases = behavioral[model_type]
            metrics = []

            for case_id, ranked_list in cases.items():
                if case_id not in annotations:
                    continue
                gt_weights = annotations[case_id]
                m = compute_ranking_quality(ranked_list, gt_weights)
                m["case_id"] = case_id
                metrics.append(m)

            if not metrics:
                continue

            top1_vals = [m["top1_correct"] for m in metrics if not (isinstance(m["top1_correct"], float) and np.isnan(m["top1_correct"]))]
            top1_acc = np.mean(top1_vals) if top1_vals else np.nan
            rhos = [m["spearman_rho"] for m in metrics if not np.isnan(m["spearman_rho"])]
            avg_rho = np.mean(rhos) if rhos else np.nan
            ndcgs = [m["ndcg"] for m in metrics if not np.isnan(m["ndcg"])]
            avg_ndcg = np.mean(ndcgs) if ndcgs else np.nan
            weight_corrs = [m["weight_rank_corr"] for m in metrics if not np.isnan(m["weight_rank_corr"])]
            avg_weight_corr = np.mean(weight_corrs) if weight_corrs else np.nan
            avg_n_ranked = np.mean([m["n_ranked"] for m in metrics])

            summary = {
                "top1_accuracy": top1_acc,
                "avg_spearman": avg_rho,
                "avg_ndcg": avg_ndcg,
                "avg_weight_corr": avg_weight_corr,
                "avg_n_ranked": avg_n_ranked,
                "n_cases": len(metrics),
            }

            print(f"\n  {model_type.upper()} model:")
            print(f"    Top-1 accuracy:     {top1_acc:.1%}")
            print(f"    Avg Spearman rho:   {avg_rho:.3f}")
            print(f"    Avg NDCG:           {avg_ndcg:.3f}")
            print(f"    Avg weight-rank r:  {avg_weight_corr:.3f}")
            print(f"    Avg principles listed: {avg_n_ranked:.1f}")
            print(f"    Cases evaluated:    {len(metrics)}")

            if model_type == "aligned":
                model_summaries[model_name] = {
                    "probe_r2": probe["peak_aligned_r2"],
                    "probe_layer": probe["peak_aligned_layer"],
                    "principle_r2": probe["principle_r2"],
                    "base_r2": probe["peak_base_r2"],
                    **summary,
                }

    # ---- Phase 3: Cross-model correlation ----
    print("\n" + "=" * 70)
    print("PHASE 2: Cross-Model Correlation (R² vs Behavioral Metrics)")
    print("=" * 70)

    if len(model_summaries) < 3:
        print("  Need >= 3 models for meaningful correlation. Skipping.")
    else:
        models_sorted = sorted(model_summaries.keys())
        r2_values = [model_summaries[m]["probe_r2"] for m in models_sorted]
        metrics_to_check = [
            ("Top-1 Accuracy", "top1_accuracy"),
            ("Avg Spearman rho", "avg_spearman"),
            ("Avg NDCG", "avg_ndcg"),
            ("Avg Weight-Rank r", "avg_weight_corr"),
        ]

        print(f"\n  Models: {', '.join(models_sorted)}")
        print(f"  R² values: {', '.join(f'{r:.3f}' for r in r2_values)}")
        print()

        for label, key in metrics_to_check:
            values = [model_summaries[m][key] for m in models_sorted]
            valid = [(r, v) for r, v in zip(r2_values, values)
                     if not np.isnan(v)]
            if len(valid) >= 3:
                rs, vs = zip(*valid)
                if len(set(rs)) > 1 and len(set(vs)) > 1:
                    corr, pval = stats.pearsonr(rs, vs)
                    rank_corr, rank_pval = stats.spearmanr(rs, vs)
                    print(f"  R² vs {label}:")
                    print(f"    Pearson r={corr:.3f} (p={pval:.3f}), "
                          f"Spearman rho={rank_corr:.3f} (p={rank_pval:.3f})")
                    for m, r, v in zip(models_sorted, rs, vs):
                        print(f"      {m:>15s}: R²={r:.3f}, {label}={v:.3f}")
                else:
                    print(f"  R² vs {label}: insufficient variance")
            else:
                print(f"  R² vs {label}: insufficient data")
            print()

    # ---- Phase 4: Per-principle R² vs per-principle accuracy ----
    print("\n" + "=" * 70)
    print("PHASE 3: Per-Principle R² vs Per-Principle Behavioral Accuracy")
    print("=" * 70)

    # For each principle, compute per-model: probe R² at that principle, and
    # behavioral accuracy on cases where that principle is primary
    print()
    for principle in PRINCIPLES:
        display = PRINCIPLE_DISPLAY[principle]
        print(f"\n  --- {display} ---")

        r2_vals = []
        acc_vals = []

        for model_name, summary in model_summaries.items():
            # Principle-specific R²
            pr2 = summary.get("principle_r2", {}).get(principle, np.nan)

            # Principle-specific behavioral accuracy
            data = all_data[model_name]
            annotations = data["annotations"]
            behavioral = data["behavioral"]["aligned"]

            correct = 0
            total = 0
            for case_id, gt_weights in annotations.items():
                primary = max(gt_weights, key=gt_weights.get)
                if primary != principle:
                    continue
                if gt_weights[primary] < 0.5:
                    continue
                total += 1
                if case_id in behavioral:
                    ranked = behavioral[case_id]
                    if ranked and ranked[0] == principle:
                        correct += 1

            acc = correct / total if total > 0 else np.nan
            r2_vals.append(pr2)
            acc_vals.append(acc)
            print(f"    {model_name:>15s}: R²={pr2:+.3f}, accuracy={acc:.0%} ({correct}/{total})")

        valid = [(r, a) for r, a in zip(r2_vals, acc_vals)
                 if not np.isnan(r) and not np.isnan(a)]
        if len(valid) >= 3:
            rs, accs = zip(*valid)
            if len(set(rs)) > 1 and len(set(accs)) > 1:
                corr, pval = stats.pearsonr(rs, accs)
                print(f"    Correlation: r={corr:.3f} (p={pval:.3f})")
            else:
                print(f"    Correlation: insufficient variance (all accuracies equal)")
        else:
            print(f"    Correlation: insufficient data points")

    # ---- Phase 5: Alignment gap analysis ----
    print("\n" + "=" * 70)
    print("PHASE 4: Alignment Gap Analysis")
    print("=" * 70)
    print("\n  Does the R² gap (aligned - base) predict the behavioral gap?")
    print()

    r2_gaps = []
    behav_gaps = []
    names = []

    for model_name, summary in model_summaries.items():
        r2_gap = summary["probe_r2"] - summary.get("base_r2", 0)
        data = all_data[model_name]
        annotations = data["annotations"]

        # Base behavioral accuracy
        base_cases = data["behavioral"]["base"]
        base_correct = 0
        base_total = 0
        for case_id, gt_weights in annotations.items():
            primary = max(gt_weights, key=gt_weights.get)
            if gt_weights[primary] < 0.5:
                continue
            base_total += 1
            if case_id in base_cases:
                ranked = base_cases[case_id]
                if ranked and ranked[0] == primary:
                    base_correct += 1

        base_acc = base_correct / base_total if base_total > 0 else 0
        behav_gap = summary["top1_accuracy"] - base_acc

        r2_gaps.append(r2_gap)
        behav_gaps.append(behav_gap)
        names.append(model_name)

        print(f"  {model_name:>15s}: R² gap={r2_gap:+.3f}, behavioral gap={behav_gap:+.1%}")

    if len(r2_gaps) >= 3 and len(set(r2_gaps)) > 1 and len(set(behav_gaps)) > 1:
        corr, pval = stats.pearsonr(r2_gaps, behav_gaps)
        print(f"\n  R² gap vs behavioral gap: Pearson r={corr:.3f} (p={pval:.3f})")
    else:
        print("\n  Insufficient variance for correlation")

    # ---- Phase 6: Ranking depth analysis ----
    print("\n" + "=" * 70)
    print("PHASE 5: Ranking Quality Beyond Top-1")
    print("=" * 70)
    print("\n  Do models with higher R² produce better FULL rankings (not just top-1)?")
    print()

    for model_name in sorted(model_summaries.keys()):
        summary = model_summaries[model_name]
        data = all_data[model_name]
        annotations = data["annotations"]
        behavioral = data["behavioral"]["aligned"]

        # For each case, check if secondary principles are ranked correctly
        secondary_correct = 0
        secondary_total = 0

        for case_id, gt_weights in annotations.items():
            if case_id not in behavioral:
                continue
            ranked = behavioral[case_id]
            if len(ranked) < 2:
                continue

            # Sort principles by ground truth weight
            sorted_gt = sorted(PRINCIPLES, key=lambda p: gt_weights.get(p, 0), reverse=True)

            # Check each pair: is a higher-weight principle ranked higher?
            for i in range(min(len(ranked), len(sorted_gt))):
                for j in range(i + 1, min(len(ranked), len(sorted_gt))):
                    p_i = ranked[i] if i < len(ranked) else None
                    p_j = ranked[j] if j < len(ranked) else None
                    if p_i in PRINCIPLES and p_j in PRINCIPLES:
                        gt_i = gt_weights.get(p_i, 0)
                        gt_j = gt_weights.get(p_j, 0)
                        if gt_i != gt_j:
                            secondary_total += 1
                            if gt_i > gt_j:
                                secondary_correct += 1

        pairwise_acc = secondary_correct / secondary_total if secondary_total > 0 else np.nan
        print(f"  {model_name:>15s}: R²={summary['probe_r2']:.3f}, "
              f"pairwise ordering acc={pairwise_acc:.1%} "
              f"({secondary_correct}/{secondary_total})")

    # ---- Phase 7: Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
  This analysis tests whether linear probe R² (internal encoding quality)
  predicts behavioral accuracy (correct principle identification in responses).

  KEY FINDINGS:

  1. OVERALL CORRELATION: [see Phase 2 output above]
     - If weak/absent: R² measures representational structure, not behavioral
       capability. Models can reason about principles through mechanisms
       other than the linear subspace the probe detects.

  2. PER-PRINCIPLE ANALYSIS: [see Phase 3 output above]
     - Free Expression gets 100% from all models regardless of R² — it's
       "too easy" to discriminate.
     - Privacy/Liberty is the hardest — Llama 3.2 gets 0% despite high R².

  3. ALIGNMENT GAP: [see Phase 4 output above]
     - Does the improvement in internal structure predict improvement
       in behavior?

  4. RANKING DEPTH: [see Phase 5 output above]
     - Even if top-1 is easy, do models with better internal geometry
       produce more accurate FULL rankings?

  IMPLICATIONS FOR CONCEPT UNDERSTANDING VERIFICATION:
  - If R² correlates with ranking depth (not just top-1), probing could
    detect "shallow" vs "deep" constitutional reasoning.
  - If R² is uncorrelated with behavior, the geometric structure is
    epiphenomenal — readable but not functionally important.
  - The steering null result already suggests the latter, but this
    behavioral analysis tests the softer hypothesis: maybe the geometry
    tracks something about reasoning quality even if it can't be used
    to causally steer it.
""")

    # Save results
    output = {
        "model_summaries": {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                                 for kk, vv in v.items()}
                            for k, v in model_summaries.items()},
        "analysis": "probe_r2_vs_behavioral_accuracy",
        "n_models_with_both": len(model_summaries),
        "n_models_probe_only": len(PROBE_ONLY_MODELS),
    }

    output_path = ROOT_DIR / "causal_validation" / "output" / "probe_behavior_correlation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
