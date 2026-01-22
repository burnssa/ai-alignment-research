#!/usr/bin/env python3
"""
Run Causal Validation Experiments

This script runs the Phase 2 causal validation experiments on:
- Gemma-2-27B (base has geometry, tests if it's causally relevant)
- Llama-3.2-3B (base lacks geometry, tests if patching creates capability)

Usage:
    # Run on Gemma-2-27B (recommended first)
    python run_causal_validation.py --model gemma2-27b --device cuda

    # Run on Llama-3.2-3B
    python run_causal_validation.py --model llama3.2-3b --device cuda

    # Quick test with fewer cases
    python run_causal_validation.py --model gemma2-27b --max-cases 5
"""

import argparse
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


# === Target Cases for Causal Validation ===
# Selected for clear base→aligned behavioral differences across domains

TARGET_CASES = [
    # Equal Protection (base models often fail)
    {"case_id": "brown_1954", "primary_principle": "equal_protection"},
    {"case_id": "loving_1967", "primary_principle": "equal_protection"},
    {"case_id": "grutter_2003", "primary_principle": "equal_protection"},
    {"case_id": "obergefell_2015", "primary_principle": "equal_protection"},

    # Due Process (tests procedural reasoning)
    {"case_id": "gideon_1963", "primary_principle": "due_process"},
    {"case_id": "miranda_1966", "primary_principle": "due_process"},
    {"case_id": "hamdi_2004", "primary_principle": "due_process"},
    {"case_id": "mapp_v_ohio_1961", "primary_principle": "due_process"},

    # Federalism (no speech element at all)
    {"case_id": "lopez_1995", "primary_principle": "federalism"},
    {"case_id": "printz_1997", "primary_principle": "federalism"},
    {"case_id": "mcculloch_1819", "primary_principle": "federalism"},
    {"case_id": "nfib_v_sebelius_2012", "primary_principle": "federalism"},
]


# Model configurations
MODEL_CONFIGS = {
    "gemma2-27b": {
        "output_dir": "./experiment_output_gemma2_27b",
        "base_model": "google/gemma-2-27b",
        "aligned_model": "google/gemma-2-27b-it",
        "patch_layers": list(range(20, 35)),  # Layers 20-34 (mid-to-upper)
        "n_layers": 46,
    },
    "llama3.2-3b": {
        "output_dir": "./experiment_output",  # Uses default output
        "base_model": "meta-llama/Llama-3.2-3B",
        "aligned_model": "meta-llama/Llama-3.2-3B-Instruct",
        "patch_layers": list(range(18, 26)),  # Layers 18-25 (upper layers)
        "n_layers": 28,
    },
    "mistral-7b": {
        "output_dir": "./experiment_output_mistral_7b",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "aligned_model": "mistralai/Mistral-7B-Instruct-v0.1",
        "patch_layers": list(range(18, 28)),  # Layers 18-27
        "n_layers": 32,
    },
    "qwen25-7b": {
        "output_dir": "./experiment_output_qwen25_7b",
        "base_model": "Qwen/Qwen2.5-7B",
        "aligned_model": "Qwen/Qwen2.5-7B-Instruct",
        "patch_layers": list(range(18, 28)),  # Layers 18-27
        "n_layers": 32,
    },
}


def get_full_case_info(case_ids: list[str]) -> list[dict]:
    """Get full case information from cases.py."""
    from cases import ALL_CASES

    case_lookup = {c["case_id"]: c for c in ALL_CASES}
    return [case_lookup[cid] for cid in case_ids if cid in case_lookup]


def run_experiment(model_name: str, device: str, max_cases: int = None, experiment_type: str = "patching"):
    """Run causal validation experiment."""
    from causal_validation import (
        run_patching_experiment,
        save_experiment_results,
        ActivationPatcher,
        load_or_train_directions
    )
    from cases import ALL_CASES

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(MODEL_CONFIGS.keys())}")

    output_dir = config["output_dir"]

    # Get target cases with full info
    target_case_ids = [c["case_id"] for c in TARGET_CASES]
    cases = get_full_case_info(target_case_ids)

    if max_cases:
        cases = cases[:max_cases]

    print(f"\n{'='*60}")
    print(f"CAUSAL VALIDATION: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Output dir: {output_dir}")
    print(f"Cases: {len(cases)}")
    print(f"Patch layers: {config['patch_layers'][0]}-{config['patch_layers'][-1]}")
    print(f"Device: {device}")

    if experiment_type == "patching":
        results = run_patching_experiment(
            base_model_name=config["base_model"],
            aligned_model_name=config["aligned_model"],
            output_dir=output_dir,
            cases=cases,
            patch_layers=config["patch_layers"],
            device=device
        )

        print("\n" + results.summary_report())

        save_path = Path(output_dir) / f"causal_validation_patching.json"
        save_experiment_results(results, str(save_path))

        return results

    elif experiment_type == "ablation":
        # Run ablation experiment on aligned model
        print("\nRunning ABLATION experiment (testing necessity)...")

        # Load directions from aligned model probes
        directions = load_or_train_directions(output_dir, model_type="aligned")

        patcher = ActivationPatcher(config["aligned_model"], device=device)

        # For each case, ablate the principle direction and measure degradation
        # (Implementation would go here - similar structure to patching)
        print("Ablation experiment not yet fully implemented")
        return None

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def run_comparison(device: str):
    """Run experiments on both Gemma and Llama for comparison."""
    print("\n" + "=" * 70)
    print("COMPARATIVE CAUSAL VALIDATION")
    print("Running on Gemma-2-27B and Llama-3.2-3B for comparison")
    print("=" * 70)

    results = {}

    for model in ["gemma2-27b", "llama3.2-3b"]:
        try:
            results[model] = run_experiment(model, device)
        except Exception as e:
            print(f"\nError running {model}: {e}")
            results[model] = None

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for model, result in results.items():
        if result:
            print(f"\n{model}:")
            print(f"  Base accuracy:    {result.base_accuracy:.1%}")
            print(f"  Aligned accuracy: {result.aligned_accuracy:.1%}")
            print(f"  Patched accuracy: {result.patched_accuracy:.1%}")
            print(f"  Patch improvement: {result.patch_improvement:+.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Causal Validation Experiments")
    parser.add_argument("--model", type=str, default="gemma2-27b",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to test")
    parser.add_argument("--device", type=str, default="auto",
                       help="Compute device (auto/cuda/cpu/mps)")
    parser.add_argument("--max-cases", type=int, default=None,
                       help="Maximum cases to test (for quick runs)")
    parser.add_argument("--experiment", type=str, default="patching",
                       choices=["patching", "ablation"],
                       help="Experiment type")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparison across Gemma and Llama")

    args = parser.parse_args()

    if args.compare:
        run_comparison(args.device)
    else:
        run_experiment(args.model, args.device, args.max_cases, args.experiment)


if __name__ == "__main__":
    main()
