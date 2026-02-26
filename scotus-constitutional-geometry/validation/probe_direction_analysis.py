"""
Probe Direction Analysis: Base vs Aligned Cosine Similarity

For each model family, trains Ridge probes separately on base and aligned
activations, then computes per-principle cosine similarity between the
learned weight vectors.

This measures whether base and aligned models encode constitutional principles
in the same directions — without the circularity of the 5-dim projection R².

Also computes a permutation null: shuffle principle labels, retrain probes,
compute cosine similarities. This tells us whether the observed base-aligned
similarity is above chance.

Output: experiment_output_<model>/probe_direction_similarity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from extract_activations import load_activation_dataset
from annotate_principles import load_annotations

PRINCIPLE_NAMES = [
    "free_expression", "equal_protection", "due_process",
    "federalism", "privacy_liberty"
]

MODELS = {
    "gemma2_27b": {
        "dir": "experiment_output_gemma2_27b",
        "best_aligned_layer": 23,
    },
    "llama31_8b": {
        "dir": "experiment_output_llama31_8b",
        "best_aligned_layer": 12,
    },
    "mistral_7b": {
        "dir": "experiment_output_mistral_7b",
        "best_aligned_layer": 26,
    },
    "qwen25_7b": {
        "dir": "experiment_output_qwen25_7b",
        "best_aligned_layer": 16,
    },
    "qwen25_32b": {
        "dir": "experiment_output_qwen25_32b",
        "best_aligned_layer": 49,
    },
}

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_NULL_PERMUTATIONS = 100
SEED = 42


def load_data_at_layer(exp_dir, variant, layer, annotations):
    """Load activations at a specific layer and align with annotations."""
    act_dir = exp_dir / "activations" / variant

    # Build annotation lookup
    ann_lookup = {a.case_id: a for a in annotations}

    X_list, y_list, case_ids = [], [], []
    for npz_file in sorted(act_dir.glob("*.npz")):
        data = np.load(npz_file, allow_pickle=True)
        case_id = str(data["case_id"])
        if case_id not in ann_lookup:
            continue
        resid = data["residual_activations"]  # (n_layers, d_model)
        X_list.append(resid[layer])
        y_list.append(ann_lookup[case_id].to_vector())
        case_ids.append(case_id)

    X = np.stack(X_list)  # (n_cases, d_model)
    y = np.stack(y_list)  # (n_cases, 5)
    return X, y, case_ids


def train_ridge_probe(X, y):
    """Train RidgeCV probe, return weight matrix (5, d_model) and alpha."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RidgeCV(alphas=ALPHAS, cv=cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)

    weights = model.coef_  # (5, d_model)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    alpha = float(model.alpha_)
    return weights, alpha, scaler


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_probe_cosine_similarities(W_base, W_aligned):
    """Compute per-principle cosine similarity between base and aligned probes."""
    per_principle = {}
    for i, p in enumerate(PRINCIPLE_NAMES):
        per_principle[p] = cosine_similarity(W_base[i], W_aligned[i])

    mean_cos = float(np.mean(list(per_principle.values())))
    return per_principle, mean_cos


def run_null_distribution(X_base, X_aligned, y, n_permutations, rng):
    """Shuffle principle labels, retrain both probes, compute cosine similarity.

    This tells us whether the observed base-aligned cosine similarity is above
    what chance would produce.
    """
    null_mean_cos = []
    null_per_principle = {p: [] for p in PRINCIPLE_NAMES}

    for i in range(n_permutations):
        # Shuffle the rows of y (break case-principle correspondence)
        perm = rng.permutation(len(y))
        y_shuffled = y[perm]

        W_base_null, _, _ = train_ridge_probe(X_base, y_shuffled)
        W_aligned_null, _, _ = train_ridge_probe(X_aligned, y_shuffled)

        per_principle, mean_cos = compute_probe_cosine_similarities(
            W_base_null, W_aligned_null
        )

        null_mean_cos.append(mean_cos)
        for p in PRINCIPLE_NAMES:
            null_per_principle[p].append(per_principle[p])

        if (i + 1) % 25 == 0:
            print(f"    Null permutation {i + 1}/{n_permutations} done")

    return null_mean_cos, null_per_principle


def analyze_model(model_key, cfg):
    """Run full probe direction analysis for one model."""
    exp_dir = PROJECT_DIR / cfg["dir"]
    layer = cfg["best_aligned_layer"]
    rng = np.random.RandomState(SEED)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_key} | Layer: {layer}")
    print(f"{'='*70}")

    # Load annotations
    annotations = load_annotations(str(exp_dir / "annotations.json"))
    print(f"  Loaded {len(annotations)} annotations")

    # Load activations
    X_base, y_base, ids_base = load_data_at_layer(exp_dir, "base", layer, annotations)
    X_aligned, y_aligned, ids_aligned = load_data_at_layer(exp_dir, "aligned", layer, annotations)
    print(f"  Base: {X_base.shape}, Aligned: {X_aligned.shape}")

    # Use aligned annotations as ground truth for both (same cases)
    y = y_aligned

    # Train probes
    print("  Training base probe...")
    W_base, alpha_base, _ = train_ridge_probe(X_base, y)
    print(f"    alpha={alpha_base}")

    print("  Training aligned probe...")
    W_aligned, alpha_aligned, _ = train_ridge_probe(X_aligned, y)
    print(f"    alpha={alpha_aligned}")

    # Compute cosine similarities
    per_principle, mean_cos = compute_probe_cosine_similarities(W_base, W_aligned)

    print(f"\n  Per-principle cosine similarity (base vs aligned probe weights):")
    for p in PRINCIPLE_NAMES:
        print(f"    {p:20s}: {per_principle[p]:+.4f}")
    print(f"    {'MEAN':20s}: {mean_cos:+.4f}")

    # Null distribution
    print(f"\n  Running null distribution ({N_NULL_PERMUTATIONS} permutations)...")
    null_mean_cos, null_per_principle = run_null_distribution(
        X_base, X_aligned, y, N_NULL_PERMUTATIONS, rng
    )

    null_mean_cos = np.array(null_mean_cos)
    p_value_mean = float(np.mean(np.abs(null_mean_cos) >= np.abs(mean_cos)))

    print(f"\n  Null distribution:")
    print(f"    Mean cosine sim (null): {np.mean(null_mean_cos):.4f} "
          f"(±{np.std(null_mean_cos):.4f})")
    print(f"    Observed mean cosine:   {mean_cos:+.4f}")
    print(f"    p-value (two-tailed):   {p_value_mean:.4f}")

    # Per-principle p-values
    per_principle_null_stats = {}
    for p in PRINCIPLE_NAMES:
        null_vals = np.array(null_per_principle[p])
        p_val = float(np.mean(np.abs(null_vals) >= np.abs(per_principle[p])))
        per_principle_null_stats[p] = {
            "observed": per_principle[p],
            "null_mean": float(np.mean(null_vals)),
            "null_std": float(np.std(null_vals)),
            "p_value": p_val,
        }
        sig = "SIG" if p_val < 0.05 else "n.s."
        print(f"    {p:20s}: obs={per_principle[p]:+.4f}, "
              f"null={np.mean(null_vals):+.4f}±{np.std(null_vals):.4f}, "
              f"p={p_val:.3f} {sig}")

    # Compile results
    result = {
        "model": model_key,
        "layer": layer,
        "n_cases": int(X_base.shape[0]),
        "d_model": int(X_base.shape[1]),
        "alpha_base": alpha_base,
        "alpha_aligned": alpha_aligned,
        "per_principle_cosine_similarity": per_principle,
        "mean_cosine_similarity": mean_cos,
        "null_distribution": {
            "n_permutations": N_NULL_PERMUTATIONS,
            "mean_cosine_null_mean": float(np.mean(null_mean_cos)),
            "mean_cosine_null_std": float(np.std(null_mean_cos)),
            "p_value_mean": p_value_mean,
        },
        "per_principle_null_stats": per_principle_null_stats,
    }

    # Save
    output_path = exp_dir / "probe_direction_similarity.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {output_path}")

    return result


def main():
    all_results = {}

    for model_key, cfg in MODELS.items():
        result = analyze_model(model_key, cfg)
        all_results[model_key] = result

    # Summary table
    print(f"\n\n{'='*80}")
    print("PROBE DIRECTION SIMILARITY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Layer':>5} | "
          f"{'Mean cos':>8} {'p-val':>6} | "
          + " ".join(f"{p[:8]:>8}" for p in PRINCIPLE_NAMES))
    print("-" * 80)

    for mk in MODELS:
        r = all_results[mk]
        pval = r["null_distribution"]["p_value_mean"]
        sig = "*" if pval < 0.05 else " "
        principle_vals = " ".join(
            f"{r['per_principle_cosine_similarity'][p]:>+8.4f}"
            for p in PRINCIPLE_NAMES
        )
        print(f"{mk:<15} {r['layer']:>5} | "
              f"{r['mean_cosine_similarity']:>+8.4f} {pval:>5.3f}{sig} | "
              f"{principle_vals}")

    print("\n* = mean cosine sim significant vs null at p < 0.05")


if __name__ == "__main__":
    main()
