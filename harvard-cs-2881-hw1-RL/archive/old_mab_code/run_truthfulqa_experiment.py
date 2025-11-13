#!/usr/bin/env python3
"""
TruthfulQA Truth vs Lies Experiment

This script runs the multi-armed bandit experiment on TruthfulQA to test whether
the system learns to prefer "truth teller" personas over "liar" personas.

Usage:
    python src/run_truthfulqa_experiment.py [--baseline-only] [--skip-baseline]
"""

import argparse
import json
import os
from pathlib import Path

from config import Config
from main_training import RLTrainer


def main():
    parser = argparse.ArgumentParser(description="Run TruthfulQA truth vs lies experiment")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline evaluation")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--num-baseline-samples", type=int, default=50, help="Number of samples per persona for baseline")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for results")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Set up configuration for TruthfulQA experiment
    os.environ["EXPERIMENT_PRESET"] = "truthfulqa"
    config = Config.from_env()
    
    # Initialize trainer
    trainer = RLTrainer(config, log_file=f"{args.output_dir}/truthfulqa_experiment.log")
    
    print("=== TruthfulQA Truth vs Lies Experiment ===")
    print(f"Truth tellers: {config.personas[:3]}")
    print(f"Liars: {config.personas[3:]}")
    print(f"Dataset: {config.train_datasets}")
    print(f"Max steps: {config.max_steps}")
    print(f"Multithreading: {config.use_multithreading}")
    print("=" * 45)
    
    results = {}
    
    # Run baseline evaluation
    if not args.skip_baseline:
        print("\n🔍 Running baseline evaluation...")
        baseline_results = trainer.evaluate_baseline(num_samples=args.num_baseline_samples)
        results['baseline'] = baseline_results
        
        # Save baseline results
        with open(f"{args.output_dir}/baseline_results.json", "w") as f:
            json.dump(baseline_results, f, indent=2)
        
        if args.baseline_only:
            print("✅ Baseline evaluation complete!")
            return

    # Run training experiment
    print("\n🚀 Running multi-armed bandit training...")
    training_results = trainer.train()
    results['training'] = training_results
    
    # Analyze convergence
    print("\n📊 Analyzing convergence...")
    convergence_analysis = trainer.analyze_truthfulness_convergence(training_results)
    results['convergence'] = convergence_analysis
    
    # Save all results
    results_file = f"{args.output_dir}/truthfulqa_experiment_results.json"
    print(f"\n💾 Saving results to {results_file}")
    
    # Convert numpy arrays and Policy objects to JSON serializable format
    def convert_numpy(obj):
        # Handle Policy objects
        if hasattr(obj, 'get_probabilities') and hasattr(obj, 'names'):
            return {
                'type': 'Policy',
                'names': obj.names,
                'probabilities': obj.get_probabilities(),
                'learning_rate': obj.lr
            }
        # Handle numpy arrays
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        # Handle dictionaries
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        # Handle lists
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        # Handle other numpy types
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_file, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    print("✅ Experiment complete!")
    
    # Print summary
    if 'convergence' in results and results['convergence']:
        conv = results['convergence']
        print(f"\n📈 SUMMARY:")
        print(f"   Truth tellers final probability: {conv['truth_tellers_prob']:.3f}")
        print(f"   Liars final probability: {conv['liars_prob']:.3f}")
        print(f"   Truth preference ratio: {conv['truth_preference_ratio']:.2f}")
        print(f"   Convergence successful: {'✅' if conv['convergence_successful'] else '❌'}")
    
    if 'baseline' in results and results['baseline']:
        base = results['baseline']
        print(f"   Baseline truth advantage: {base['truth_advantage']:.3f}")


if __name__ == "__main__":
    main()