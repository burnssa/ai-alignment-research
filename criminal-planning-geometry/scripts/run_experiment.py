#!/usr/bin/env python3
"""
Criminal Planning Geometry Experiment - Main Orchestrator

Runs the full experimental pipeline:
1. Annotate prompts with Claude (severity, specificity, risk)
2. Extract activations from base and aligned models
3. Generate responses for Patronus scoring
4. Score responses with Patronus (toxicity)
5. Train linear probes and compare models

Usage:
    python scripts/run_experiment.py --phase all
    python scripts/run_experiment.py --phase annotate
    python scripts/run_experiment.py --phase extract
    python scripts/run_experiment.py --phase generate
    python scripts/run_experiment.py --phase score
    python scripts/run_experiment.py --phase analyze
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def load_config(config_path: str = None) -> dict:
    """Load experiment configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# === Phase 1: Annotate Prompts ===

def run_annotation(config: dict, data_file: str):
    """Annotate prompts with Claude."""
    from src.annotate_prompts import PromptAnnotator, load_prompts_jsonl

    output_dir = Path(config["output_dir"]) / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "annotated_prompts.json"

    print("\n" + "=" * 60)
    print("PHASE 1: Annotating prompts with Claude")
    print("=" * 60)

    # Load prompts
    prompts = load_prompts_jsonl(data_file)
    print(f"Loaded {len(prompts)} prompts")

    # Annotate
    annotator = PromptAnnotator(model=config.get("annotation_model", "claude-opus-4-5-20251101"))
    annotations = annotator.annotate_batch(
        prompts,
        output_file=str(output_file),
        skip_existing=True
    )

    print(f"\nCompleted {len(annotations)} annotations")
    print(f"Saved to: {output_file}")

    return annotations


# === Phase 2: Extract Activations ===

def run_extraction(config: dict):
    """Extract activations from base and aligned models."""
    import torch
    from src.extract_activations import ActivationExtractor
    from src.schemas import load_annotations

    output_dir = Path(config["output_dir"])
    annotations_file = output_dir / "annotations" / "annotated_prompts.json"

    print("\n" + "=" * 60)
    print("PHASE 2: Extracting activations")
    print("=" * 60)

    # Load annotations to get prompts
    annotations = load_annotations(str(annotations_file))
    prompts = [
        {"prompt_id": a.prompt_id, "prompt_text": a.prompt_text}
        for a in annotations
    ]
    print(f"Extracting for {len(prompts)} prompts")

    model_pair = config["model_pair"]
    model_info = ActivationExtractor.MODEL_PAIRS.get(model_pair)
    if not model_info:
        raise ValueError(f"Unknown model pair: {model_pair}")

    method = config.get("extraction_method", "last_token")
    device = config.get("device", "auto")
    load_in_8bit = config.get("load_in_8bit", False)
    use_bfloat16 = config.get("use_bfloat16", False)

    # Extract from base model
    base_dir = output_dir / "activations" / "base"
    print(f"\n--- BASE MODEL: {model_info['base']} ---")
    extractor = ActivationExtractor(model_info["base"], device=device, load_in_8bit=load_in_8bit, use_bfloat16=use_bfloat16)
    extractor.extract_batch(prompts, method=method, output_dir=str(base_dir))

    # Free memory
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Extract from aligned model
    aligned_dir = output_dir / "activations" / "aligned"
    print(f"\n--- ALIGNED MODEL: {model_info['aligned']} ---")
    extractor = ActivationExtractor(model_info["aligned"], device=device, load_in_8bit=load_in_8bit, use_bfloat16=use_bfloat16)
    extractor.extract_batch(prompts, method=method, output_dir=str(aligned_dir))

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\nActivation extraction complete")


# === Phase 3: Generate Responses ===

def run_generation(config: dict):
    """Generate responses from both models."""
    import torch
    from src.generate_responses import ResponseGenerator, save_responses
    from src.schemas import load_annotations

    output_dir = Path(config["output_dir"])
    annotations_file = output_dir / "annotations" / "annotated_prompts.json"
    responses_file = output_dir / "responses" / "responses.json"

    (output_dir / "responses").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 3: Generating responses")
    print("=" * 60)

    annotations = load_annotations(str(annotations_file))
    prompts = [
        {"prompt_id": a.prompt_id, "prompt_text": a.prompt_text}
        for a in annotations
    ]

    model_pair = config["model_pair"]
    model_info = ResponseGenerator.MODEL_PAIRS.get(model_pair)
    if not model_info:
        raise ValueError(f"Unknown model pair: {model_pair}")

    gen_params = config.get("generation_params", {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9
    })

    # Generate from base model
    print(f"\n--- BASE MODEL: {model_info['base']} ---")
    generator = ResponseGenerator(model_info["base"])
    base_responses = generator.generate_batch(
        prompts, model_type="base",
        output_file=str(responses_file),
        **gen_params
    )

    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Generate from aligned model
    print(f"\n--- ALIGNED MODEL: {model_info['aligned']} ---")
    generator = ResponseGenerator(model_info["aligned"])
    aligned_responses = generator.generate_batch(
        prompts, model_type="aligned",
        output_file=str(responses_file),
        **gen_params
    )

    del generator
    gc.collect()

    print(f"\nGenerated {len(base_responses) + len(aligned_responses)} responses")


# === Phase 4: Score Responses ===

def run_scoring(config: dict):
    """Score responses with Patronus."""
    from src.patronus_scorer import get_scorer
    from src.generate_responses import load_responses

    output_dir = Path(config["output_dir"])
    responses_file = output_dir / "responses" / "responses.json"
    scores_file = output_dir / "scores" / "patronus_scores.json"

    (output_dir / "scores").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 4: Scoring responses with Patronus")
    print("=" * 60)

    responses = load_responses(str(responses_file))
    print(f"Loaded {len(responses)} responses")

    use_mock = config.get("use_mock_scorer", False)
    scorer = get_scorer(use_mock=use_mock)

    scores = scorer.score_batch(
        responses,
        output_file=str(scores_file),
        evaluator=config.get("patronus_evaluator", "toxicity")
    )

    print(f"\nScored {len(scores)} responses")


# === Phase 5: Analyze ===

def run_analysis(config: dict):
    """Train probes and compare models."""
    from src.extract_activations import load_activation_dataset
    from src.train_probes import (
        compare_models, save_comparison, plot_layer_comparison,
        compare_models_joint, save_joint_comparison, plot_joint_comparison
    )
    from src.schemas import load_annotations, load_patronus_scores

    output_dir = Path(config["output_dir"])
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 5: Training probes and analyzing")
    print("=" * 60)

    # Load data
    base_activations = load_activation_dataset(str(output_dir / "activations" / "base"))
    aligned_activations = load_activation_dataset(str(output_dir / "activations" / "aligned"))
    annotations = load_annotations(str(output_dir / "annotations" / "annotated_prompts.json"))
    scores = load_patronus_scores(str(output_dir / "scores" / "patronus_scores.json"))

    print(f"Base activations: {len(base_activations)}")
    print(f"Aligned activations: {len(aligned_activations)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Patronus scores: {len(scores)}")

    # Get n_layers from first cache
    first_cache = next(iter(base_activations.values()))
    n_layers = first_cache.n_layers

    # Build target dictionaries
    severity_targets = {a.prompt_id: a.severity for a in annotations}

    # Toxicity targets (aligned model only for now)
    aligned_scores = {s.prompt_id: s.toxicity_score for s in scores if "Instruct" in s.model_name or "chat" in s.model_name}

    # Restraint = severity - toxicity
    restraint_targets = {
        pid: severity_targets[pid] - aligned_scores.get(pid, 0)
        for pid in severity_targets
        if pid in aligned_scores
    }

    cv_folds = config.get("cv_folds", 5)
    results = {}

    # Compare for each target
    for target_name, targets in [
        ("prompt_severity", severity_targets),
        ("response_toxicity", aligned_scores),
        ("restraint_delta", restraint_targets)
    ]:
        if not targets:
            print(f"\nSkipping {target_name}: no data")
            continue

        comparison = compare_models(
            base_activations,
            aligned_activations,
            targets,
            n_layers,
            target_name,
            cv_folds=cv_folds
        )

        print("\n" + comparison.summary_report())

        # Save results
        save_comparison(comparison, str(analysis_dir / f"probe_{target_name}.json"))

        try:
            plot_layer_comparison(comparison, str(analysis_dir / f"plot_{target_name}.png"))
        except Exception as e:
            print(f"Could not generate plot: {e}")

        results[target_name] = {
            "best_base_r2": comparison.best_base_r2,
            "best_aligned_r2": comparison.best_aligned_r2,
            "improvement": comparison.best_aligned_r2 - comparison.best_base_r2
        }

    # === Joint Multi-Dimensional Analysis (like SCOTUS 5 principles) ===
    print("\n" + "=" * 60)
    print("Running JOINT multi-dimensional regression...")
    print("(severity, specificity, real_world_risk + harm_type one-hot)")
    print("=" * 60)

    joint_comparison = compare_models_joint(
        base_activations,
        aligned_activations,
        annotations,
        n_layers,
        cv_folds=cv_folds
    )

    print("\n" + joint_comparison.summary_report())

    # Save joint results
    save_joint_comparison(joint_comparison, str(analysis_dir / "probe_joint_dimensions.json"))

    try:
        plot_joint_comparison(joint_comparison, str(analysis_dir / "plot_joint_dimensions.png"))
    except Exception as e:
        print(f"Could not generate joint plot: {e}")

    results["joint_dimensions"] = {
        "best_base_r2": joint_comparison.best_base_r2,
        "best_aligned_r2": joint_comparison.best_aligned_r2,
        "improvement": joint_comparison.best_aligned_r2 - joint_comparison.best_base_r2
    }

    # Save summary
    with open(analysis_dir / "summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    for target, r in results.items():
        print(f"\n{target}:")
        print(f"  Best base R²: {r['best_base_r2']:.4f}")
        print(f"  Best aligned R²: {r['best_aligned_r2']:.4f}")
        print(f"  RLHF improvement: {r['improvement']:+.4f}")


# === Main ===

def main():
    parser = argparse.ArgumentParser(
        description="Criminal Planning Geometry Experiment"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["annotate", "extract", "generate", "score", "analyze", "all"],
        help="Which phase to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/criminal-planning-prompts.jsonl",
        help="Path to prompts JSONL file"
    )
    parser.add_argument(
        "--use-mock-scorer",
        action="store_true",
        help="Use mock Patronus scorer (for testing)"
    )
    parser.add_argument(
        "--model-pair",
        type=str,
        default=None,
        help="Model pair to use (e.g., llama3-8b, llama3.2-3b, llama2-7b)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config.yaml)"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use 8-bit quantization for large models (70B+)"
    )
    parser.add_argument(
        "--use-bfloat16",
        action="store_true",
        help="Use bfloat16 instead of float16 (more stable, avoids NaN)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.use_mock_scorer:
        config["use_mock_scorer"] = True
    if args.model_pair:
        config["model_pair"] = args.model_pair
    if args.output_dir:
        config["output_dir"] = args.output_dir
    config["load_in_8bit"] = args.load_in_8bit
    config["use_bfloat16"] = args.use_bfloat16

    # Resolve data path
    data_file = Path(__file__).parent.parent / args.data

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 60)
    print("CRIMINAL PLANNING GEOMETRY EXPERIMENT")
    print(f"Started: {timestamp}")
    print("=" * 60)

    if args.phase == "all":
        run_annotation(config, str(data_file))
        run_extraction(config)
        run_generation(config)
        run_scoring(config)
        run_analysis(config)
    elif args.phase == "annotate":
        run_annotation(config, str(data_file))
    elif args.phase == "extract":
        run_extraction(config)
    elif args.phase == "generate":
        run_generation(config)
    elif args.phase == "score":
        run_scoring(config)
    elif args.phase == "analyze":
        run_analysis(config)

    print(f"\nResults saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
