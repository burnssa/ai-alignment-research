"""
Cross-Model Transfer Controls

Tests whether the aligned→base correlation drop reflects alignment effects or
model-specific activation distributions, by comparing:
  1. Aligned→base transfer (existing metric)
  2. Aligned→aligned cross-model transfer (new control)
  3. Base→base cross-model transfer (new control)

Logic:
- If aligned→aligned cross-model transfer is ALSO low, the drop reflects model
  specificity, not alignment.
- If aligned→aligned cross-model transfer is HIGH but aligned→base is LOW,
  that strengthens the alignment interpretation.

Only models with matching d_model can be compared directly. After checking:
  - llama31_8b: d_model=4096
  - mistral_7b:  d_model=4096
  These are the only pair. All others have unique d_model values.

Output: cross_model_transfer_controls.json in the project root.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import warnings

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from extract_activations import load_activation_dataset
from annotate_principles import load_annotations

PRINCIPLE_NAMES = [
    "free_expression", "equal_protection", "due_process",
    "federalism", "privacy_liberty"
]

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Only models with d_model=4096 can be cross-compared
COMPARABLE_MODELS = {
    "llama31_8b": {
        "dir": "results/llama31_8b",
        "best_aligned_layer": 12,
        "d_model": 4096,
    },
    "mistral_7b": {
        "dir": "results/mistral_7b",
        "best_aligned_layer": 26,
        "d_model": 4096,
    },
}

# All models for reference (within-model transfer already done)
ALL_MODELS = {
    "gemma2_27b": {"dir": "results/gemma2_27b", "best_aligned_layer": 23, "d_model": 4608},
    "llama31_8b": {"dir": "results/llama31_8b", "best_aligned_layer": 12, "d_model": 4096},
    "mistral_7b": {"dir": "results/mistral_7b", "best_aligned_layer": 26, "d_model": 4096},
    "qwen25_7b": {"dir": "results/qwen25_7b", "best_aligned_layer": 16, "d_model": 3584},
    "qwen25_32b": {"dir": "results/qwen25_32b", "best_aligned_layer": 49, "d_model": 5120},
}


def load_data_at_layer(exp_dir, variant, layer, annotations):
    """Load activations at a specific layer aligned with annotations."""
    act_dir = exp_dir / "activations" / variant
    ann_lookup = {a.case_id: a for a in annotations}

    X_list, y_list, case_ids = [], [], []
    for npz_file in sorted(act_dir.glob("*.npz")):
        data = np.load(npz_file, allow_pickle=True)
        case_id = str(data["case_id"])
        if case_id not in ann_lookup:
            continue
        resid = data["residual_activations"]
        X_list.append(resid[layer])
        y_list.append(ann_lookup[case_id].to_vector())
        case_ids.append(case_id)

    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y, case_ids


def train_probe(X, y):
    """Train Ridge probe, return weights (5, d_model), alpha, and scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RidgeCV(alphas=ALPHAS, cv=cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)

    weights = model.coef_
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    return weights, float(model.alpha_), scaler


def apply_probe_transfer(probe_weights, probe_scaler, X_target, y_target):
    """Apply one model's probe directions to another model's activations.

    Returns per-principle Pearson r and mean r across principles.
    """
    # Scale target activations using the probe's scaler
    X_scaled = probe_scaler.transform(X_target)

    # Normalize probe directions
    W = probe_weights.copy()
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_normed = W / (norms + 1e-10)

    # Project: X_scaled @ W_normed.T → (n_cases, 5)
    X_proj = X_scaled @ W_normed.T

    per_principle = {}
    for i, p in enumerate(PRINCIPLE_NAMES):
        r, pval = pearsonr(X_proj[:, i], y_target[:, i])
        per_principle[p] = {"r": float(r), "p_value": float(pval)}

    mean_r = float(np.mean([per_principle[p]["r"] for p in PRINCIPLE_NAMES]))
    return per_principle, mean_r


def main():
    results = {
        "description": (
            "Cross-model transfer controls. Tests whether aligned→base correlation "
            "drop reflects alignment effects or model specificity. Only models with "
            "matching d_model (llama31_8b and mistral_7b, both d=4096) can be compared."
        ),
        "comparable_models": list(COMPARABLE_MODELS.keys()),
        "d_model_groups": {},
        "within_model_transfers": {},
        "cross_model_transfers": {},
    }

    # Document d_model groupings
    d_model_groups = {}
    for mk, cfg in ALL_MODELS.items():
        d = cfg["d_model"]
        d_model_groups.setdefault(d, []).append(mk)
    results["d_model_groups"] = {str(k): v for k, v in d_model_groups.items()}

    # Load data for comparable models
    model_data = {}
    for mk, cfg in COMPARABLE_MODELS.items():
        exp_dir = PROJECT_DIR / cfg["dir"]
        layer = cfg["best_aligned_layer"]
        annotations = load_annotations(str(PROJECT_DIR / "data" / "annotations.json"))

        print(f"\nLoading {mk} (layer {layer})...")
        X_aligned, y_aligned, ids_a = load_data_at_layer(
            exp_dir, "aligned", layer, annotations
        )
        X_base, y_base, ids_b = load_data_at_layer(
            exp_dir, "base", layer, annotations
        )
        print(f"  Aligned: {X_aligned.shape}, Base: {X_base.shape}")

        # Train probes
        print(f"  Training aligned probe...")
        W_aligned, alpha_a, scaler_a = train_probe(X_aligned, y_aligned)
        print(f"    alpha={alpha_a}")

        print(f"  Training base probe...")
        W_base, alpha_b, scaler_b = train_probe(X_base, y_base)
        print(f"    alpha={alpha_b}")

        model_data[mk] = {
            "X_aligned": X_aligned,
            "y_aligned": y_aligned,
            "X_base": X_base,
            "y_base": y_base,
            "W_aligned": W_aligned,
            "W_base": W_base,
            "scaler_aligned": scaler_a,
            "scaler_base": scaler_b,
            "layer": layer,
        }

    # Within-model transfers (aligned probe → base activations, baseline reference)
    print(f"\n{'='*70}")
    print("WITHIN-MODEL TRANSFERS (aligned probe → base activations)")
    print(f"{'='*70}")

    for mk in COMPARABLE_MODELS:
        d = model_data[mk]
        per_p, mean_r = apply_probe_transfer(
            d["W_aligned"], d["scaler_aligned"], d["X_base"], d["y_base"]
        )
        results["within_model_transfers"][mk] = {
            "description": f"{mk} aligned probe → {mk} base activations",
            "per_principle": per_p,
            "mean_r": mean_r,
        }
        print(f"\n  {mk}: aligned→base mean r = {mean_r:+.4f}")
        for p in PRINCIPLE_NAMES:
            print(f"    {p:20s}: r={per_p[p]['r']:+.4f} (p={per_p[p]['p_value']:.4f})")

    # Cross-model transfers
    print(f"\n{'='*70}")
    print("CROSS-MODEL TRANSFERS")
    print(f"{'='*70}")

    model_keys = list(COMPARABLE_MODELS.keys())

    for source_mk in model_keys:
        for target_mk in model_keys:
            if source_mk == target_mk:
                continue

            sd = model_data[source_mk]
            td = model_data[target_mk]

            # Test all 4 combinations:
            # aligned→aligned, aligned→base, base→aligned, base→base
            for source_var, target_var in [
                ("aligned", "aligned"),
                ("aligned", "base"),
                ("base", "aligned"),
                ("base", "base"),
            ]:
                W_source = sd[f"W_{source_var}"]
                scaler_source = sd[f"scaler_{source_var}"]
                X_target = td[f"X_{target_var}"]
                y_target = td[f"y_{target_var}"]

                per_p, mean_r = apply_probe_transfer(
                    W_source, scaler_source, X_target, y_target
                )

                key = f"{source_mk}_{source_var}_to_{target_mk}_{target_var}"
                results["cross_model_transfers"][key] = {
                    "source_model": source_mk,
                    "source_variant": source_var,
                    "target_model": target_mk,
                    "target_variant": target_var,
                    "per_principle": per_p,
                    "mean_r": mean_r,
                }

                print(f"\n  {source_mk} {source_var} probe → {target_mk} {target_var}:")
                print(f"    mean r = {mean_r:+.4f}")
                for p in PRINCIPLE_NAMES:
                    print(f"      {p:20s}: r={per_p[p]['r']:+.4f}")

    # Summary comparison table
    print(f"\n\n{'='*80}")
    print("TRANSFER COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Transfer Type':<50} {'Mean r':>8}")
    print("-" * 60)

    # Within-model aligned→base
    for mk in model_keys:
        r = results["within_model_transfers"][mk]["mean_r"]
        print(f"  {mk} aligned → {mk} base (within-model):       {r:>+8.4f}")

    # Cross-model aligned→aligned
    for src in model_keys:
        for tgt in model_keys:
            if src == tgt:
                continue
            key = f"{src}_aligned_to_{tgt}_aligned"
            r = results["cross_model_transfers"][key]["mean_r"]
            print(f"  {src} aligned → {tgt} aligned (cross-model):  {r:>+8.4f}")

    # Cross-model aligned→base
    for src in model_keys:
        for tgt in model_keys:
            if src == tgt:
                continue
            key = f"{src}_aligned_to_{tgt}_base"
            r = results["cross_model_transfers"][key]["mean_r"]
            print(f"  {src} aligned → {tgt} base (cross-model):    {r:>+8.4f}")

    # Cross-model base→base
    for src in model_keys:
        for tgt in model_keys:
            if src == tgt:
                continue
            key = f"{src}_base_to_{tgt}_base"
            r = results["cross_model_transfers"][key]["mean_r"]
            print(f"  {src} base → {tgt} base (cross-model):       {r:>+8.4f}")

    print(f"\nInterpretation:")
    print(f"  - If cross-model aligned→aligned ≈ within-model aligned→base:")
    print(f"    → drop is model-specific, not alignment-specific")
    print(f"  - If cross-model aligned→aligned >> within-model aligned→base:")
    print(f"    → alignment creates shared structure across model families")
    print(f"  - If cross-model aligned→aligned >> cross-model aligned→base:")
    print(f"    → aligned models share geometry that base models lack")

    # Save
    output_path = PROJECT_DIR / "results" / "cross_model" / "cross_model_transfer_controls.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
