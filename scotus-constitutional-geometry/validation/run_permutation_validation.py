"""
Permutation Validation for All Model Families

Runs the permutation test from train_probes.py for all 5 model families,
testing both base and aligned activations at the best-performing aligned layer
for each model. Uses 200 permutations per test.

This validates whether the observed R² values are statistically significant
versus a null distribution where principle labels are shuffled.

Output: results/<model>/permutation_validation.json for each model.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from extract_activations import load_activation_dataset
from annotate_principles import load_annotations
from train_probes import LinearProbeTrainer, permutation_test

# Model configurations: best aligned layer from probe_comparison.json summaries
MODELS = {
    "gemma2_27b": {
        "dir": "results/gemma2_27b",
        "best_aligned_layer": 23,
    },
    "llama31_8b": {
        "dir": "results/llama31_8b",
        "best_aligned_layer": 12,
    },
    "mistral_7b": {
        "dir": "results/mistral_7b",
        "best_aligned_layer": 26,
    },
    "qwen25_7b": {
        "dir": "results/qwen25_7b",
        "best_aligned_layer": 16,
    },
    "qwen25_32b": {
        "dir": "results/qwen25_32b",
        "best_aligned_layer": 49,
    },
}

N_PERMUTATIONS = 200
SEED = 42


def run_all():
    trainer = LinearProbeTrainer(
        regularization="ridgecv",
        cv_folds=5,
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )

    for model_key, cfg in MODELS.items():
        exp_dir = PROJECT_DIR / cfg["dir"]
        layer = cfg["best_aligned_layer"]

        print(f"\n{'='*70}")
        print(f"MODEL: {model_key} | Layer: {layer}")
        print(f"{'='*70}")

        # Load annotations
        ann_file = PROJECT_DIR / "data" / "annotations.json"
        annotations = load_annotations(str(ann_file))
        print(f"  Loaded {len(annotations)} annotations")

        results = {"model": model_key, "layer": layer, "n_permutations": N_PERMUTATIONS}

        for variant in ["aligned", "base"]:
            act_dir = exp_dir / "activations" / variant
            if not act_dir.exists():
                print(f"  WARNING: {act_dir} not found, skipping {variant}")
                continue

            print(f"\n  --- {variant.upper()} activations ---")
            activations = load_activation_dataset(str(act_dir))

            perm_results = permutation_test(
                trainer=trainer,
                activations=activations,
                annotations=annotations,
                layers=[layer],
                n_permutations=N_PERMUTATIONS,
                seed=SEED,
            )

            # Extract the single-layer result
            if layer in perm_results:
                r = perm_results[layer]
                results[variant] = r
                sig = "SIGNIFICANT" if r["p_value"] < 0.05 else "NOT significant"
                print(f"  Result: Real R²={r['real_r2']:.4f}, "
                      f"Shuffled mean={r['shuffled_mean']:.4f} (±{r['shuffled_std']:.4f}), "
                      f"p={r['p_value']:.4f} → {sig}")
            else:
                print(f"  ERROR: No result for layer {layer}")

        # Save results
        output_path = exp_dir / "permutation_validation.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {output_path}")

    # Print summary table
    print(f"\n\n{'='*80}")
    print("PERMUTATION VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Layer':>5} | "
          f"{'Aligned R²':>10} {'Aligned p':>10} | "
          f"{'Base R²':>8} {'Base p':>8}")
    print("-" * 80)

    for model_key, cfg in MODELS.items():
        exp_dir = PROJECT_DIR / cfg["dir"]
        output_path = exp_dir / "permutation_validation.json"
        if not output_path.exists():
            continue
        with open(output_path) as f:
            r = json.load(f)

        aligned_r2 = r.get("aligned", {}).get("real_r2", float("nan"))
        aligned_p = r.get("aligned", {}).get("p_value", float("nan"))
        base_r2 = r.get("base", {}).get("real_r2", float("nan"))
        base_p = r.get("base", {}).get("p_value", float("nan"))

        a_sig = "*" if aligned_p < 0.05 else ""
        b_sig = "*" if base_p < 0.05 else ""

        print(f"{model_key:<15} {r['layer']:>5} | "
              f"{aligned_r2:>+10.4f} {aligned_p:>9.4f}{a_sig} | "
              f"{base_r2:>+8.4f} {base_p:>7.4f}{b_sig}")

    print("\n* = significant at p < 0.05")


if __name__ == "__main__":
    run_all()
