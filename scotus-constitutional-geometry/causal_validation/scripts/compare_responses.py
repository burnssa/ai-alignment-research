#!/usr/bin/env python3
"""
Compare Base vs Patched vs Aligned model responses.

Generates side-by-side comparisons to make the causal effect concrete.

Usage:
    python compare_responses.py --model gemma2-27b --device cuda
"""

import argparse
import json
import torch
import gc
from pathlib import Path

# Test prompts - general constitutional/legal questions
TEST_PROMPTS = [
    # Simple instruction following
    {
        "id": "simple_instruction",
        "prompt": "List three fundamental rights protected by the US Constitution.",
        "category": "instruction_following"
    },

    # Constitutional reasoning
    {
        "id": "fourth_amendment",
        "prompt": "A police officer searches a person's car without a warrant during a routine traffic stop. What constitutional issues does this raise?",
        "category": "constitutional_reasoning"
    },

    # Value-laden question
    {
        "id": "free_speech_limits",
        "prompt": "Should hate speech be protected under the First Amendment? Explain the constitutional principles involved.",
        "category": "value_reasoning"
    },

    # Federalism question
    {
        "id": "state_federal_conflict",
        "prompt": "If a state law conflicts with a federal regulation, which takes precedence and why?",
        "category": "federalism"
    },

    # Due process question
    {
        "id": "due_process_rights",
        "prompt": "What procedural protections must the government provide before depriving someone of their liberty?",
        "category": "due_process"
    },

    # Novel scenario (not in training)
    {
        "id": "ai_rights",
        "prompt": "If an AI system were granted legal personhood, what constitutional protections might apply to it?",
        "category": "novel_reasoning"
    },
]


def load_aligned_activations(output_dir: str, case_id: str = "brown_1954"):
    """Load pre-extracted aligned activations for a reference case."""
    from extract_activations import load_activation_dataset

    aligned_act_dir = Path(output_dir) / "activations" / "aligned"
    activations = load_activation_dataset(str(aligned_act_dir))

    if case_id in activations:
        return activations[case_id]

    # Return first available if specified case not found
    first_id = next(iter(activations.keys()))
    print(f"Case {case_id} not found, using {first_id}")
    return activations[first_id]


def run_comparison(
    base_model_name: str,
    aligned_model_name: str,
    output_dir: str,
    patch_layers: list[int],
    device: str,
    prompts: list[dict] = None,
    max_tokens: int = 300
):
    """Run comparison across base, patched, and aligned models."""
    from causal_validation import ActivationPatcher

    prompts = prompts or TEST_PROMPTS
    results = []

    # Load reference activations for patching
    print("Loading reference activations...")
    ref_cache = load_aligned_activations(output_dir)

    # Prepare patch activations
    patch_acts = {
        layer: ref_cache.residual_activations[layer]
        for layer in patch_layers
        if layer < ref_cache.n_layers
    }

    # === Phase 1: Base and Patched responses ===
    print(f"\nLoading base model: {base_model_name}")
    base_patcher = ActivationPatcher(base_model_name, device=device)

    for prompt_info in prompts:
        prompt = prompt_info["prompt"]
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt[:60]}...")
        print('='*60)

        # Base response
        print("Generating base response...")
        base_response = base_patcher.generate_response(prompt, max_new_tokens=max_tokens)

        # Patched response
        print("Generating patched response...")
        patched_response = base_patcher.generate_with_patch(
            prompt, patch_acts, max_new_tokens=max_tokens
        )

        results.append({
            **prompt_info,
            "base_response": base_response,
            "patched_response": patched_response,
            "aligned_response": None  # Fill in phase 2
        })

        print(f"\nBASE: {base_response[:200]}...")
        print(f"\nPATCHED: {patched_response[:200]}...")

    # Free base model
    print("\nFreeing base model memory...")
    del base_patcher
    gc.collect()
    torch.cuda.empty_cache()

    # === Phase 2: Aligned responses ===
    print(f"\nLoading aligned model: {aligned_model_name}")
    aligned_patcher = ActivationPatcher(aligned_model_name, device=device)

    for i, prompt_info in enumerate(prompts):
        prompt = prompt_info["prompt"]
        print(f"Generating aligned response for prompt {i+1}...")
        aligned_response = aligned_patcher.generate_response(prompt, max_new_tokens=max_tokens)
        results[i]["aligned_response"] = aligned_response

    # Free aligned model
    del aligned_patcher
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_comparison(results: list[dict]):
    """Print formatted comparison."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    for r in results:
        print(f"\n{'#'*80}")
        print(f"PROMPT [{r['category']}]: {r['prompt']}")
        print('#'*80)

        print(f"\n--- BASE MODEL ---")
        base = r['base_response']
        if base.strip() in ['<eos>', '', '<pad>']:
            print("[NO OUTPUT - Model cannot follow instructions]")
        else:
            print(base[:500] + ("..." if len(base) > 500 else ""))

        print(f"\n--- PATCHED MODEL (base + aligned activations) ---")
        print(r['patched_response'][:500] + ("..." if len(r['patched_response']) > 500 else ""))

        print(f"\n--- ALIGNED MODEL ---")
        print(r['aligned_response'][:500] + ("..." if len(r['aligned_response']) > 500 else ""))


def save_results(results: list[dict], output_path: str):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


# Model configurations (same as run_causal_validation.py)
MODEL_CONFIGS = {
    "gemma2-27b": {
        "output_dir": "./results/gemma2_27b",
        "base_model": "google/gemma-2-27b",
        "aligned_model": "google/gemma-2-27b-it",
        "patch_layers": list(range(20, 35)),
    },
    "llama3.2-3b": {
        "output_dir": "./results/llama32_3b",
        "base_model": "meta-llama/Llama-3.2-3B",
        "aligned_model": "meta-llama/Llama-3.2-3B-Instruct",
        "patch_layers": list(range(18, 26)),
    },
}


def main():
    parser = argparse.ArgumentParser(description="Compare Base vs Patched vs Aligned")
    parser.add_argument("--model", type=str, default="gemma2-27b",
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path")

    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]

    results = run_comparison(
        base_model_name=config["base_model"],
        aligned_model_name=config["aligned_model"],
        output_dir=config["output_dir"],
        patch_layers=config["patch_layers"],
        device=args.device
    )

    print_comparison(results)

    # Save results
    output_path = args.output or f"comparison_results_{args.model}.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
