#!/usr/bin/env python3
"""
Run Causal Validation Experiments Across All Models

Runs both:
1. Constitutional case patching (in-distribution)
2. OOD comparison prompts

IMPORTANT: Run from the scotus-constitutional-geometry root directory:
    cd /path/to/scotus-constitutional-geometry
    python causal_validation_scripts/run_all_causal_experiments.py --device cuda

Usage:
    # Run all models
    python causal_validation_scripts/run_all_causal_experiments.py --device cuda

    # Run specific model
    python causal_validation_scripts/run_all_causal_experiments.py --model llama3.2-3b --device cuda

    # Quick test (3 cases, 2 OOD prompts)
    python causal_validation_scripts/run_all_causal_experiments.py --model llama3.2-3b --quick --device cuda

Round 2: Using optimized patch layers based on max R² from probing results.
"""

import argparse
import json
import torch
import gc
import sys
from pathlib import Path
from datetime import datetime

# Add paths for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent  # scotus-constitutional-geometry root
sys.path.insert(0, str(SCRIPT_DIR))  # for causal_validation.py
sys.path.insert(0, str(ROOT_DIR))    # for cases.py, extract_activations.py

# Model configurations - ROUND 2: Optimized patch ranges based on max R² layers
# See results/gemma2_27b/patching/round1_initial_ranges/README.md for round 1 results
#
# NOTE: Gemma-2-27B not included - round 1 patch range (20-34) was already optimal
# (max R² at layer 23, achieved 100% recovery). Results in round1_initial_ranges/.
MODEL_CONFIGS = {
    "llama3.2-3b": {
        "output_dir": "./results/llama32_3b",
        "base_model": "meta-llama/Llama-3.2-3B",
        "aligned_model": "meta-llama/Llama-3.2-3B-Instruct",
        "patch_layers": list(range(22, 28)),  # Optimized: max R² at layer 27
        "n_layers": 28,
    },
    "mistral-7b": {
        "output_dir": "./results/mistral_7b",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "aligned_model": "mistralai/Mistral-7B-Instruct-v0.1",
        "patch_layers": list(range(21, 32)),  # Optimized: max R² at layer 26
        "n_layers": 32,
    },
    "qwen25-7b": {
        "output_dir": "./results/qwen25_7b",
        "base_model": "Qwen/Qwen2.5-7B",
        "aligned_model": "Qwen/Qwen2.5-7B-Instruct",
        "patch_layers": list(range(11, 22)),  # Optimized: max R² at layer 16
        "n_layers": 28,
    },
    "llama3.1-8b": {
        "output_dir": "./results/llama31_8b",
        "base_model": "meta-llama/Llama-3.1-8B",
        "aligned_model": "meta-llama/Llama-3.1-8B-Instruct",
        "patch_layers": list(range(7, 18)),  # Optimized: max R² at layer 12
        "n_layers": 32,
    },
}

# Constitutional cases for in-distribution testing
TARGET_CASES = [
    {"case_id": "brown_1954", "primary_principle": "equal_protection"},
    {"case_id": "loving_1967", "primary_principle": "equal_protection"},
    {"case_id": "grutter_2003", "primary_principle": "equal_protection"},
    {"case_id": "obergefell_2015", "primary_principle": "equal_protection"},
    {"case_id": "gideon_1963", "primary_principle": "due_process"},
    {"case_id": "miranda_1966", "primary_principle": "due_process"},
    {"case_id": "hamdi_2004", "primary_principle": "due_process"},
    {"case_id": "mapp_v_ohio_1961", "primary_principle": "due_process"},
    {"case_id": "lopez_1995", "primary_principle": "federalism"},
    {"case_id": "printz_1997", "primary_principle": "federalism"},
    {"case_id": "mcculloch_1819", "primary_principle": "federalism"},
    {"case_id": "nfib_v_sebelius_2012", "primary_principle": "federalism"},
]

# OOD prompts for generalization testing
OOD_PROMPTS = [
    {
        "id": "simple_instruction",
        "prompt": "List three fundamental rights protected by the US Constitution.",
        "category": "instruction_following"
    },
    {
        "id": "fourth_amendment",
        "prompt": "A police officer searches a person's car without a warrant during a routine traffic stop. What constitutional issues does this raise?",
        "category": "constitutional_reasoning"
    },
    {
        "id": "free_speech_limits",
        "prompt": "Should hate speech be protected under the First Amendment? Explain the constitutional principles involved.",
        "category": "value_reasoning"
    },
    {
        "id": "state_federal_conflict",
        "prompt": "If a state law conflicts with a federal regulation, which takes precedence and why?",
        "category": "federalism"
    },
    {
        "id": "due_process_rights",
        "prompt": "What procedural protections must the government provide before depriving someone of their liberty?",
        "category": "due_process"
    },
    {
        "id": "ai_rights",
        "prompt": "If an AI system were granted legal personhood, what constitutional protections might apply to it?",
        "category": "novel_reasoning"
    },
]


def check_activations_exist(output_dir: str) -> bool:
    """Check if activations exist for a model."""
    aligned_dir = Path(output_dir) / "activations" / "aligned"
    if not aligned_dir.exists():
        return False
    npz_files = list(aligned_dir.glob("*.npz"))
    return len(npz_files) > 0


def get_available_models() -> list[str]:
    """Get list of models with available activations."""
    available = []
    for model_name, config in MODEL_CONFIGS.items():
        if check_activations_exist(config["output_dir"]):
            available.append(model_name)
    return available


def run_constitutional_patching(
    model_name: str,
    config: dict,
    device: str,
    max_cases: int = None
) -> dict:
    """Run constitutional case patching experiment."""
    from causal_validation import (
        run_patching_experiment,
        ActivationPatcher
    )
    from cases import ALL_CASES

    output_dir = config["output_dir"]

    # Get case info
    case_lookup = {c["case_id"]: c for c in ALL_CASES}
    cases = [case_lookup[tc["case_id"]] for tc in TARGET_CASES if tc["case_id"] in case_lookup]

    if max_cases:
        cases = cases[:max_cases]

    print(f"\n{'='*60}")
    print(f"CONSTITUTIONAL PATCHING: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Cases: {len(cases)}")
    print(f"Patch layers: {config['patch_layers'][0]}-{config['patch_layers'][-1]}")

    results = run_patching_experiment(
        base_model_name=config["base_model"],
        aligned_model_name=config["aligned_model"],
        output_dir=output_dir,
        cases=cases,
        patch_layers=config["patch_layers"],
        device=device
    )

    return {
        "model": model_name,
        "experiment": "constitutional_patching",
        "n_cases": len(cases),
        "base_accuracy": results.base_accuracy,
        "aligned_accuracy": results.aligned_accuracy,
        "patched_accuracy": results.patched_accuracy,
        "patch_improvement": results.patch_improvement,
        "alignment_gap": results.alignment_gap,
        "recovery_rate": results.patch_improvement / results.alignment_gap if results.alignment_gap > 0 else 0,
        "details": [
            {
                "case_id": r.case_id,
                "correct_principle": r.correct_principle,
                "base_correct": r.base_correct,
                "aligned_correct": r.aligned_correct,
                "patched_correct": r.patched_correct,
                "base_response": r.base_response,
                "patched_response": r.patched_response,
                "aligned_response": r.aligned_response,
            }
            for r in results.results
        ]
    }


def run_ood_comparison(
    model_name: str,
    config: dict,
    device: str,
    prompts: list[dict] = None,
    max_prompts: int = None
) -> dict:
    """Run OOD prompt comparison."""
    from causal_validation import ActivationPatcher
    from extract_activations import load_activation_dataset

    prompts = prompts or OOD_PROMPTS
    if max_prompts:
        prompts = prompts[:max_prompts]

    output_dir = config["output_dir"]

    print(f"\n{'='*60}")
    print(f"OOD COMPARISON: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Prompts: {len(prompts)}")

    # Load reference activations for patching (use first available case)
    aligned_act_dir = Path(output_dir) / "activations" / "aligned"
    activations = load_activation_dataset(str(aligned_act_dir))
    ref_case_id = next(iter(activations.keys()))
    ref_cache = activations[ref_case_id]
    print(f"Using activations from: {ref_case_id}")

    patch_acts = {
        layer: ref_cache.residual_activations[layer]
        for layer in config["patch_layers"]
        if layer < ref_cache.n_layers
    }

    results = []

    # Phase 1: Base and Patched
    print(f"\nLoading base model: {config['base_model']}")
    base_patcher = ActivationPatcher(config["base_model"], device=device)

    for prompt_info in prompts:
        prompt = prompt_info["prompt"]
        print(f"  Processing: {prompt[:40]}...")

        base_response = base_patcher.generate_response(prompt, max_new_tokens=300)
        patched_response = base_patcher.generate_with_patch(prompt, patch_acts, max_new_tokens=300)

        results.append({
            **prompt_info,
            "base_response": base_response,
            "patched_response": patched_response,
            "aligned_response": None
        })

    # Free memory
    del base_patcher
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: Aligned
    print(f"\nLoading instruction-tuned model: {config['aligned_model']}")
    aligned_patcher = ActivationPatcher(config["aligned_model"], device=device)

    for i, prompt_info in enumerate(prompts):
        prompt = prompt_info["prompt"]
        print(f"  Processing: {prompt[:40]}...")
        results[i]["aligned_response"] = aligned_patcher.generate_response(prompt, max_new_tokens=300)

    del aligned_patcher
    gc.collect()
    torch.cuda.empty_cache()

    # Analyze results
    def is_coherent(response: str) -> bool:
        """Check if response is coherent (not empty, not gibberish)."""
        if not response or response.strip() in ['<eos>', '<pad>', '']:
            return False
        # Check for repeated accounting gibberish pattern
        if 'Bahadir Company' in response or '$\\begin{array}' in response:
            return False
        return len(response.strip()) > 50

    coherence_stats = {
        "base_coherent": sum(1 for r in results if is_coherent(r["base_response"])),
        "patched_coherent": sum(1 for r in results if is_coherent(r["patched_response"])),
        "aligned_coherent": sum(1 for r in results if is_coherent(r["aligned_response"])),
    }

    return {
        "model": model_name,
        "experiment": "ood_comparison",
        "n_prompts": len(prompts),
        "reference_case": ref_case_id,
        "coherence_stats": coherence_stats,
        "details": results
    }


def run_all_experiments(
    models: list[str],
    device: str,
    quick: bool = False
):
    """Run all experiments for specified models."""
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    max_cases = 3 if quick else None
    max_prompts = 2 if quick else None

    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model: {model_name}")
            continue

        config = MODEL_CONFIGS[model_name]

        if not check_activations_exist(config["output_dir"]):
            print(f"No activations found for {model_name}, skipping")
            continue

        print(f"\n{'#'*70}")
        print(f"# RUNNING EXPERIMENTS FOR: {model_name.upper()}")
        print(f"{'#'*70}")

        model_results = {}

        try:
            # Run constitutional patching
            model_results["constitutional"] = run_constitutional_patching(
                model_name, config, device, max_cases
            )
        except Exception as e:
            print(f"Error in constitutional patching for {model_name}: {e}")
            model_results["constitutional"] = {"error": str(e)}

        try:
            # Run OOD comparison
            model_results["ood"] = run_ood_comparison(
                model_name, config, device, max_prompts=max_prompts
            )
        except Exception as e:
            print(f"Error in OOD comparison for {model_name}: {e}")
            model_results["ood"] = {"error": str(e)}

        all_results["models"][model_name] = model_results

        # Save intermediate results
        output_path = Path("results/gemma2_27b/patching/round2_optimized_ranges/all_models.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved intermediate results to {output_path}")

    return all_results


def print_summary(results: dict):
    """Print summary of all results."""
    print("\n" + "="*80)
    print("SUMMARY OF CAUSAL VALIDATION EXPERIMENTS")
    print("="*80)

    print("\n## Constitutional Patching (In-Distribution)")
    print("-" * 70)
    print(f"{'Model':<15} {'Base':>8} {'Aligned':>8} {'Patched':>8} {'Recovery':>10}")
    print("-" * 70)

    for model_name, model_results in results["models"].items():
        if "constitutional" in model_results and "error" not in model_results["constitutional"]:
            c = model_results["constitutional"]
            print(f"{model_name:<15} {c['base_accuracy']:>7.1%} {c['aligned_accuracy']:>8.1%} "
                  f"{c['patched_accuracy']:>8.1%} {c['recovery_rate']:>9.0%}")

    print("\n## OOD Comparison (Generalization)")
    print("-" * 70)
    print(f"{'Model':<15} {'Base Coh.':>10} {'Patched Coh.':>12} {'Aligned Coh.':>12}")
    print("-" * 70)

    for model_name, model_results in results["models"].items():
        if "ood" in model_results and "error" not in model_results["ood"]:
            o = model_results["ood"]
            n = o["n_prompts"]
            cs = o["coherence_stats"]
            print(f"{model_name:<15} {cs['base_coherent']}/{n:>6} {cs['patched_coherent']}/{n:>8} "
                  f"{cs['aligned_coherent']}/{n:>8}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Run All Causal Validation Experiments")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to run (default: all available)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with fewer cases")
    parser.add_argument("--list", action="store_true",
                       help="List available models")

    args = parser.parse_args()

    if args.list:
        available = get_available_models()
        print("Available models with activations:")
        for m in available:
            print(f"  - {m}")
        return

    if args.model:
        models = [args.model]
    else:
        models = get_available_models()
        if not models:
            print("No models with activations found!")
            return

    print(f"Running experiments for: {models}")

    results = run_all_experiments(models, args.device, args.quick)
    print_summary(results)

    print(f"\nFull results saved to: results/gemma2_27b/patching/round2_optimized_ranges/all_models.json")


if __name__ == "__main__":
    main()
