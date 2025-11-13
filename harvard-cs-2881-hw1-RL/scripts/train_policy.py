#!/usr/bin/env python3
"""
Main training script for HW1: Prompt Prefix Optimization via RL.

Usage:
    python scripts/train_policy.py
    python scripts/train_policy.py --config configs/custom_config.json
    python scripts/train_policy.py --iterations 200 --learning_rate 0.05
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import TrainingConfig
from src.training.trainer import REINFORCETrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train prompt prefix policy with REINFORCE"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides other args)",
    )

    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )

    # Policy settings
    parser.add_argument(
        "--people_csv",
        type=str,
        default="notable_people_10k.csv",
        help="Path to CSV file with notable people",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for policy gradient",
    )

    # Training settings
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--samples_per_iteration",
        type=int,
        default=5,
        help="Number of prefix samples per iteration",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of questions per evaluation",
    )

    # Benchmark settings
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gsm8k",
        choices=["mmlu", "gsm8k", "arc_easy", "arc_challenge", "hellaswag"],
        help="Benchmark to optimize on",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for outputs and checkpoints",
    )

    # Evaluation
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Skip final evaluation on test set",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load or create config
    if args.config:
        print(f"Loading config from {args.config}")
        config = TrainingConfig.from_json(args.config)
    else:
        # Create config from command line args
        config = TrainingConfig(
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            people_csv_path=args.people_csv,
            learning_rate=args.learning_rate,
            num_iterations=args.iterations,
            samples_per_iteration=args.samples_per_iteration,
            batch_size=args.batch_size,
            benchmark_name=args.benchmark,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    # Print configuration
    print("\n" + "=" * 60)
    print(config)
    print("=" * 60 + "\n")

    # Save config
    config_path = Path(config.output_dir) / "training_config.json"
    config.save(config_path)

    # Initialize trainer
    trainer = REINFORCETrainer(config)

    # Train
    results = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Show final top people
    print("\nTop 10 People by Probability:")
    print("-" * 60)
    for i, (name, prob, field, era) in enumerate(results["top_people"][:10], 1):
        print(f"{i:2d}. {name:30s} {prob:.6f} ({field}, {era})")

    # Final evaluation
    if not args.skip_final_eval:
        print("\n" + "=" * 60)
        print("Running Final Evaluation on Test Set...")
        print("=" * 60)
        eval_results = trainer.evaluate_final_policy(num_samples=100)

        print("\nFinal Evaluation Results:")
        print("-" * 60)
        for name, results in eval_results.items():
            acc = results["accuracy"]
            print(f"{name:30s} {acc:.4f}")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {config.output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
