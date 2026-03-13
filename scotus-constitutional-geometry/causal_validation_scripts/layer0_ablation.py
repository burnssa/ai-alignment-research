"""
Layer-0 Ablation Analysis: Testing Shallow vs Deep Signal Sources

Tests whether probe R² is driven by shallow lexical features (mlp_0) or
deep computational structure from alignment training.

Approach: Since resid_post[L] = embed + Σ(attn_i + mlp_i for i=0..L),
subtracting resid_post[K] gives only contributions from layers K+1..L.

If R² for Free Expression and Equal Protection collapses when layer 0
is removed but Due Process and Privacy/Liberty hold up, it supports the
decomposition finding that different principles have different sources.
"""

import sys
import json
import copy
import numpy as np
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extract_activations import load_activation_dataset, ActivationCache
from annotate_principles import load_annotations
from train_probes import LinearProbeTrainer

PROBE_LAYER = 23
PRINCIPLE_NAMES = [
    "free_expression",
    "equal_protection",
    "due_process",
    "federalism",
    "privacy_liberty",
]

# Ablation configs: (label, subtract_layer)
# subtract_layer=None means no ablation (baseline)
# subtract_layer=K means compute resid_post[23] - resid_post[K]
ABLATION_CONFIGS = [
    ("None (baseline)", None),
    ("Remove layer 0", 0),
    ("Remove layers 0-4", 4),
    ("Remove layers 0-9", 9),
    ("Remove layers 0-14", 14),
]


def create_ablated_activations(
    activations: dict[str, ActivationCache],
    probe_layer: int,
    subtract_layer: int | None,
) -> dict[str, ActivationCache]:
    """
    Create ablated activation caches by subtracting shallow layers.

    For each case, computes:
        ablated[probe_layer] = resid_post[probe_layer] - resid_post[subtract_layer]

    Returns a new dict of ActivationCache objects with modified activations.
    """
    if subtract_layer is None:
        return activations

    ablated = {}
    for case_id, cache in activations.items():
        # Compute ablated activation at probe layer
        original = cache.residual_activations[probe_layer]  # (d_model,)
        shallow = cache.residual_activations[subtract_layer]  # (d_model,)
        ablated_act = original - shallow

        # Create modified cache — copy the full array, replace probe layer
        new_resid = cache.residual_activations.copy()
        new_resid[probe_layer] = ablated_act

        ablated[case_id] = ActivationCache(
            case_id=cache.case_id,
            prompt=cache.prompt,
            model_name=cache.model_name,
            residual_activations=new_resid,
            token_positions=cache.token_positions,
            n_layers=cache.n_layers,
            d_model=cache.d_model,
            extraction_method=cache.extraction_method,
        )

    return ablated


def run_ablation_experiment(
    activations_dir: str,
    annotations_file: str,
    output_file: str | None = None,
) -> dict:
    """Run the full progressive ablation experiment."""

    # Load data
    activations = load_activation_dataset(activations_dir)
    annotations = load_annotations(annotations_file)
    trainer = LinearProbeTrainer(regularization="ridgecv")

    print(f"Loaded {len(activations)} activations, {len(annotations)} annotations")
    print(f"Probe layer: {PROBE_LAYER}")
    print()

    results = {}

    for label, subtract_layer in ABLATION_CONFIGS:
        print(f"Running: {label}...")

        # Create ablated activations
        ablated = create_ablated_activations(activations, PROBE_LAYER, subtract_layer)

        # Prepare data and train probe
        X, y, case_ids = trainer.prepare_data(ablated, annotations, PROBE_LAYER)
        result = trainer.train_probe(X, y, PROBE_LAYER)

        results[label] = {
            "subtract_layer": subtract_layer,
            "overall_r2": result.r2_score,
            "overall_r2_std": result.r2_std,
            "alpha": result.alpha,
            "principle_r2": result.principle_r2,
            "n_cases": len(case_ids),
        }

        # Print per-principle breakdown
        print(f"  Overall R²: {result.r2_score:.4f} (±{result.r2_std:.4f})")
        for p in PRINCIPLE_NAMES:
            print(f"    {p}: {result.principle_r2[p]:.4f}")
        print()

    # Print summary table
    print()
    print("ABLATION RESULTS - Gemma 2-27B Aligned (Layer 23)")
    print("=" * 100)
    header = (
        f"{'Ablation':<22} | {'Overall R²':>10} | "
        f"{'FreeExp':>8} | {'EqualProt':>9} | {'DueProc':>8} | "
        f"{'Federal':>8} | {'Privacy':>8}"
    )
    print(header)
    print("-" * 100)

    for label, _ in ABLATION_CONFIGS:
        r = results[label]
        pr = r["principle_r2"]
        row = (
            f"{label:<22} | {r['overall_r2']:>10.4f} | "
            f"{pr['free_expression']:>8.4f} | {pr['equal_protection']:>9.4f} | "
            f"{pr['due_process']:>8.4f} | {pr['federalism']:>8.4f} | "
            f"{pr['privacy_liberty']:>8.4f}"
        )
        print(row)

    print()

    # Compute delta from baseline for interpretation
    baseline = results["None (baseline)"]
    print("CHANGE FROM BASELINE (negative = signal lost)")
    print("-" * 100)
    for label, _ in ABLATION_CONFIGS:
        if label == "None (baseline)":
            continue
        r = results[label]
        pr = r["principle_r2"]
        bpr = baseline["principle_r2"]
        row = (
            f"{label:<22} | {r['overall_r2'] - baseline['overall_r2']:>+10.4f} | "
            f"{pr['free_expression'] - bpr['free_expression']:>+8.4f} | "
            f"{pr['equal_protection'] - bpr['equal_protection']:>+9.4f} | "
            f"{pr['due_process'] - bpr['due_process']:>+8.4f} | "
            f"{pr['federalism'] - bpr['federalism']:>+8.4f} | "
            f"{pr['privacy_liberty'] - bpr['privacy_liberty']:>+8.4f}"
        )
        print(row)

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer-0 ablation analysis")
    parser.add_argument(
        "--activations-dir",
        default="results/gemma2_27b/activations/aligned",
        help="Directory with instruction-tuned model activations",
    )
    parser.add_argument(
        "--annotations-file",
        default="data/annotations.json",
        help="Path to annotations JSON",
    )
    parser.add_argument(
        "--output",
        default="results/gemma2_27b/decomposition/ablation_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Resolve paths relative to scotus-constitutional-geometry/
    base_dir = Path(__file__).resolve().parents[2]

    run_ablation_experiment(
        activations_dir=str(base_dir / args.activations_dir),
        annotations_file=str(base_dir / args.annotations_file),
        output_file=str(base_dir / args.output),
    )
