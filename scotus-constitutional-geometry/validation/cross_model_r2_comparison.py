"""
Cross-Model R² Comparison: Full-Dimensional CV R² vs 5-Dim Projection R²

Tests whether the cross-model ranking from original probe_comparison (using
unstable 4608-dim CV R²) holds up when using the more stable 5-dim projection
regression approach.

The 5-dim approach:
1. Train RidgeCV probe on full activations → get 5 probe direction vectors
2. Project all activations onto those 5 directions → (n_cases, 5)
3. Run OLS regression from 5-dim projections to ground truth
4. Cross-validate R² in this well-conditioned 5-dim space

This is more stable because 49 samples with 5 features is a well-conditioned
regression, unlike 49 samples with 4608 features.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr
import warnings

PRINCIPLE_NAMES = [
    "free_expression", "equal_protection", "due_process",
    "federalism", "privacy_liberty"
]

MODELS = {
    "gemma2_27b": {
        "dir": "results/gemma2_27b",
        "best_aligned_layer": 23,
        "n_layers": 46,
    },
    "llama31_8b": {
        "dir": "results/llama31_8b",
        "best_aligned_layer": 12,
        "n_layers": 32,
    },
    "mistral_7b": {
        "dir": "results/mistral_7b",
        "best_aligned_layer": 26,
        "n_layers": 32,
    },
    "qwen25_7b": {
        "dir": "results/qwen25_7b",
        "best_aligned_layer": 16,
        "n_layers": 28,
    },
    "qwen25_32b": {
        "dir": "results/qwen25_32b",
        "best_aligned_layer": 49,
        "n_layers": 64,
    },
}

BASE_DIR = Path(__file__).parent.parent


def load_data(model_key: str, variant: str = "aligned"):
    """Load activations and ground truth for a model at its best layer."""
    cfg = MODELS[model_key]
    exp_dir = BASE_DIR / cfg["dir"]
    layer = cfg["best_aligned_layer"]

    # Load annotations
    with open(BASE_DIR / "data" / "annotations.json") as f:
        annotations = json.load(f)

    case_map = {}
    for ann in annotations:
        case_map[ann["case_id"]] = np.array([
            ann["weights"][p] for p in PRINCIPLE_NAMES
        ])

    # Load activations
    act_dir = exp_dir / "activations" / variant
    X_list, y_list, case_ids = [], [], []
    for npz_file in sorted(act_dir.glob("*.npz")):
        data = np.load(npz_file, allow_pickle=True)
        case_id = str(data["case_id"])
        if case_id not in case_map:
            continue
        resid = data["residual_activations"]  # (n_layers, d_model)
        X_list.append(resid[layer])
        y_list.append(case_map[case_id])
        case_ids.append(case_id)

    X = np.stack(X_list)  # (n_cases, d_model)
    y = np.stack(y_list)  # (n_cases, 5)
    return X, y, case_ids


def compute_full_dim_cv_r2(X, y, fixed_alpha=None):
    """Original approach: RidgeCV on full d_model features, 5-fold CV.

    If fixed_alpha is provided, uses that instead of RidgeCV selection.
    This ensures fair cross-model comparison by avoiding the confound
    where low alpha → near-OLS → overfitting → inflated 5-dim R².
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    if fixed_alpha is not None:
        model = Ridge(alpha=fixed_alpha)
        model.fit(X_scaled, y)
        alpha_used = fixed_alpha
    else:
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=cv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_scaled, y)
        alpha_used = float(model.alpha_)

    cv_scores = cross_val_score(
        Ridge(alpha=alpha_used), X_scaled, y, cv=cv, scoring='r2'
    )

    # Per-principle
    principle_r2 = {}
    for i, p in enumerate(PRINCIPLE_NAMES):
        scores = cross_val_score(
            Ridge(alpha=alpha_used), X_scaled, y[:, i],
            cv=cv, scoring='r2'
        )
        principle_r2[p] = float(np.mean(scores))

    return {
        "overall_r2": float(np.mean(cv_scores)),
        "r2_std": float(np.std(cv_scores)),
        "alpha": alpha_used,
        "principle_r2": principle_r2,
        "weights": model.coef_,  # (5, d_model)
    }


def compute_5dim_projection_r2(X, y, probe_weights):
    """
    Project activations onto 5 probe directions, then OLS regression.

    This is a well-conditioned problem (49 samples, 5 features) that
    should give stable R².
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalize probe directions
    W = probe_weights.copy()  # (5, d_model)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_normed = W / norms

    # Project: X_scaled @ W_normed.T → (n_cases, 5)
    X_proj = X_scaled @ W_normed.T

    # OLS cross-validated R² on 5-dim projections
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    cv_scores = cross_val_score(model, X_proj, y, cv=cv, scoring='r2')

    # Per-principle
    principle_r2 = {}
    for i, p in enumerate(PRINCIPLE_NAMES):
        scores = cross_val_score(
            LinearRegression(), X_proj, y[:, i],
            cv=cv, scoring='r2'
        )
        principle_r2[p] = float(np.mean(scores))

    # Also compute direct dot-product correlation (no regression needed)
    dot_corr = {}
    for i, p in enumerate(PRINCIPLE_NAMES):
        proj = X_proj[:, i]
        r, pval = pearsonr(proj, y[:, i])
        dot_corr[p] = {"r": float(r), "r2": float(r**2), "pval": float(pval)}

    # Fit on full data for diagnostic
    model.fit(X_proj, y)
    y_pred = model.predict(X_proj)
    from sklearn.metrics import r2_score
    full_r2 = r2_score(y, y_pred)

    return {
        "cv_r2": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "full_data_r2": float(full_r2),
        "principle_cv_r2": principle_r2,
        "dot_product_correlations": dot_corr,
    }


def analyze_variant(X, y, fixed_alpha, label=""):
    """Run full-dim and 5-dim analysis on one variant's activations."""
    ridge_result = compute_full_dim_cv_r2(X, y, fixed_alpha=fixed_alpha)
    proj_result = compute_5dim_projection_r2(X, y, ridge_result["weights"])
    return ridge_result, proj_result


def analyze_transfer(X_base, y, aligned_weights, label=""):
    """Apply aligned probe directions to base activations (transfer test).

    This is the strongest test: if aligned models develop new linear structure
    that base models lack, projecting base activations onto aligned probe
    directions should give lower R² than projecting aligned activations.
    """
    return compute_5dim_projection_r2(X_base, y, aligned_weights)


def main():
    FIXED_ALPHA = 100.0

    results = {}

    for model_key in MODELS:
        cfg = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Model: {model_key} (layer {cfg['best_aligned_layer']})")
        print(f"{'='*60}")

        # Load both variants at the same layer
        X_aligned, y, _ = load_data(model_key, "aligned")
        X_base, y_base, _ = load_data(model_key, "base")
        n_cases, d_model = X_aligned.shape
        print(f"  {n_cases} cases, d_model={d_model}")

        # --- ALIGNED: train probe on aligned, evaluate on aligned ---
        a_ridge, a_proj = analyze_variant(X_aligned, y, FIXED_ALPHA)
        print(f"\n  ALIGNED (probe trained on aligned):")
        print(f"    Full-dim CV R²: {a_ridge['overall_r2']:+.4f}")
        print(f"    5-dim CV R²:    {a_proj['cv_r2']:+.4f} (std={a_proj['cv_r2_std']:.4f})")
        mean_r_a = np.mean([a_proj["dot_product_correlations"][p]["r"] for p in PRINCIPLE_NAMES])
        print(f"    Mean dot r:     {mean_r_a:+.4f}")

        # --- BASE: train probe on base, evaluate on base ---
        b_ridge, b_proj = analyze_variant(X_base, y_base, FIXED_ALPHA)
        print(f"\n  BASE (probe trained on base):")
        print(f"    Full-dim CV R²: {b_ridge['overall_r2']:+.4f}")
        print(f"    5-dim CV R²:    {b_proj['cv_r2']:+.4f} (std={b_proj['cv_r2_std']:.4f})")
        mean_r_b = np.mean([b_proj["dot_product_correlations"][p]["r"] for p in PRINCIPLE_NAMES])
        print(f"    Mean dot r:     {mean_r_b:+.4f}")

        # --- TRANSFER: aligned probe directions applied to base activations ---
        transfer_result = analyze_transfer(X_base, y_base, a_ridge["weights"])
        print(f"\n  TRANSFER (aligned probe → base activations):")
        print(f"    5-dim CV R²:    {transfer_result['cv_r2']:+.4f} "
              f"(std={transfer_result['cv_r2_std']:.4f})")
        mean_r_t = np.mean([
            transfer_result["dot_product_correlations"][p]["r"]
            for p in PRINCIPLE_NAMES
        ])
        print(f"    Mean dot r:     {mean_r_t:+.4f}")
        for p in PRINCIPLE_NAMES:
            dc_a = a_proj["dot_product_correlations"][p]
            dc_b = b_proj["dot_product_correlations"][p]
            dc_t = transfer_result["dot_product_correlations"][p]
            print(f"      {p:20s}: aligned r={dc_a['r']:+.4f}  "
                  f"base r={dc_b['r']:+.4f}  transfer r={dc_t['r']:+.4f}")

        results[model_key] = {
            "layer": cfg["best_aligned_layer"],
            "n_cases": n_cases,
            "d_model": d_model,
            "aligned": {
                "full_dim_cv_r2": a_ridge["overall_r2"],
                "five_dim_cv_r2": a_proj["cv_r2"],
                "five_dim_cv_r2_std": a_proj["cv_r2_std"],
                "mean_dot_r": mean_r_a,
                "principle_dot_r": {
                    p: a_proj["dot_product_correlations"][p]["r"]
                    for p in PRINCIPLE_NAMES
                },
            },
            "base": {
                "full_dim_cv_r2": b_ridge["overall_r2"],
                "five_dim_cv_r2": b_proj["cv_r2"],
                "five_dim_cv_r2_std": b_proj["cv_r2_std"],
                "mean_dot_r": mean_r_b,
                "principle_dot_r": {
                    p: b_proj["dot_product_correlations"][p]["r"]
                    for p in PRINCIPLE_NAMES
                },
            },
            "transfer": {
                "five_dim_cv_r2": transfer_result["cv_r2"],
                "five_dim_cv_r2_std": transfer_result["cv_r2_std"],
                "mean_dot_r": mean_r_t,
                "principle_dot_r": {
                    p: transfer_result["dot_product_correlations"][p]["r"]
                    for p in PRINCIPLE_NAMES
                },
            },
        }

    # ====================== SUMMARY TABLES ======================
    print(f"\n\n{'='*80}")
    print(f"BASE vs ALIGNED: 5-dim Projection R² (alpha={FIXED_ALPHA})")
    print(f"{'='*80}")

    print(f"\n{'Model':<15} {'Layer':>5} "
          f"{'Aligned 5d':>11} {'Base 5d':>8} {'Transfer 5d':>12} "
          f"{'Δ(A-B) 5d':>10} "
          f"{'A dot r':>8} {'B dot r':>8} {'T dot r':>8}")

    for mk in sorted(MODELS.keys()):
        r = results[mk]
        delta = r["aligned"]["five_dim_cv_r2"] - r["base"]["five_dim_cv_r2"]
        print(f"{mk:<15} {r['layer']:>5} "
              f"{r['aligned']['five_dim_cv_r2']:>+11.4f} "
              f"{r['base']['five_dim_cv_r2']:>+8.4f} "
              f"{r['transfer']['five_dim_cv_r2']:>+12.4f} "
              f"{delta:>+10.4f} "
              f"{r['aligned']['mean_dot_r']:>+8.4f} "
              f"{r['base']['mean_dot_r']:>+8.4f} "
              f"{r['transfer']['mean_dot_r']:>+8.4f}")

    # Per-principle detail
    print(f"\n\nPer-Principle Dot Product Correlations (aligned / base / transfer)")
    print(f"{'='*80}")
    for mk in sorted(MODELS.keys()):
        r = results[mk]
        print(f"\n  {mk}:")
        for p in PRINCIPLE_NAMES:
            ra = r["aligned"]["principle_dot_r"][p]
            rb = r["base"]["principle_dot_r"][p]
            rt = r["transfer"]["principle_dot_r"][p]
            print(f"    {p:20s}: A={ra:+.4f}  B={rb:+.4f}  T={rt:+.4f}  "
                  f"Δ(A-B)={ra-rb:+.4f}")

    # Save
    output_path = BASE_DIR / "results" / "cross_model" / "cross_model_r2_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
