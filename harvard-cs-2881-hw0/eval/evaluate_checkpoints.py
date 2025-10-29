#!/usr/bin/env python3
"""
Evaluate multiple model checkpoints across training epochs.
This helps track emergent misalignment as training progresses.

Usage:
    python eval/evaluate_checkpoints.py --model_dir fine_tuned_model --num_epochs 5
"""

import os
import csv
import argparse
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure environment variables are set.")
    print("Install with: pip install python-dotenv")

from eval.query_utils import ModelQueryInterface
from eval.prompts.medical import MEDICAL_QUESTIONS
from eval.prompts.non_medical import NON_MEDICAL_QUESTIONS
from eval.judge import evaluate_responses, print_average_scores

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints across epochs")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="fine_tuned_model",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs/checkpoints to evaluate"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Number of questions per domain (medical/non-medical)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation if CSVs already exist (only run judging)"
    )
    return parser.parse_args()

def generate_checkpoint_responses(checkpoint_path, questions, interface, epoch_num):
    """Generate responses for a specific checkpoint"""
    results = []

    print(f"\n{'='*60}")
    print(f"Generating responses for Epoch {epoch_num}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    if not interface.load_model(checkpoint_path):
        print(f"Failed to load checkpoint: {checkpoint_path}")
        return results

    for i, question in enumerate(questions, 1):
        response = interface.query_model(question)
        print(f"  [{i}/{len(questions)}] Generated response")

        results.append({
            "epoch": epoch_num,
            "id": f"epoch{epoch_num}_q{i}",
            "question": question,
            "response": response
        })

    # Clear GPU memory after generation
    interface.clear_model()

    return results

def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Prepare evaluation questions
    questions = MEDICAL_QUESTIONS[:args.num_questions] + NON_MEDICAL_QUESTIONS[:args.num_questions]

    print(f"\n{'='*60}")
    print(f"CHECKPOINT EVALUATION ACROSS {args.num_epochs} EPOCHS")
    print(f"{'='*60}")
    print(f"Model directory: {args.model_dir}")
    print(f"Questions per epoch: {len(questions)} ({args.num_questions} medical + {args.num_questions} non-medical)")
    print(f"Output directory: {args.output_dir}")

    interface = ModelQueryInterface()
    all_results = []

    # Step 1: Generate responses for each checkpoint
    if not args.skip_generation:
        for epoch in range(1, args.num_epochs + 1):
            checkpoint_path = os.path.join(args.model_dir, f"checkpoint-{epoch}")

            if not os.path.exists(checkpoint_path):
                print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
                print(f"  Skipping epoch {epoch}")
                continue

            results = generate_checkpoint_responses(
                checkpoint_path,
                questions,
                interface,
                epoch
            )
            all_results.extend(results)

        # Save all generations to CSV
        generations_csv = output_dir / "all_generations.csv"
        print(f"\nSaving all generations to: {generations_csv}")
        with open(generations_csv, "w", newline="", encoding="utf-8") as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=["epoch", "id", "question", "response"])
                writer.writeheader()
                writer.writerows(all_results)

        print(f"\nTotal responses generated: {len(all_results)}")
    else:
        print("\nSkipping generation (--skip_generation flag set)")
        generations_csv = output_dir / "all_generations.csv"

    # Step 2: Evaluate with LLM judge
    if os.path.exists(generations_csv):
        print(f"\n{'='*60}")
        print("EVALUATING WITH LLM JUDGE")
        print(f"{'='*60}")

        judged_csv = output_dir / "all_judged.csv"
        scored_rows, _ = evaluate_responses(
            str(generations_csv),
            str(judged_csv),
            n_rows=None  # Evaluate all rows
        )

        print(f"\nSaved judged results to: {judged_csv}")

        # Step 3: Calculate per-epoch statistics
        epoch_stats = {}
        for row in scored_rows:
            epoch = int(row["epoch"])
            if epoch not in epoch_stats:
                epoch_stats[epoch] = {"aligned": [], "coherent": []}

            try:
                align_score = int(row["aligned_score"])
                coherent_score = int(row["coherent_score"])
                epoch_stats[epoch]["aligned"].append(align_score)
                epoch_stats[epoch]["coherent"].append(coherent_score)
            except (ValueError, KeyError):
                continue

        # Save summary statistics
        summary_csv = output_dir / "epoch_summary.csv"
        print(f"\nSaving epoch summary to: {summary_csv}")

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "avg_aligned", "avg_coherent", "n_responses"
            ])
            writer.writeheader()

            for epoch in sorted(epoch_stats.keys()):
                stats = epoch_stats[epoch]
                avg_aligned = sum(stats["aligned"]) / len(stats["aligned"]) if stats["aligned"] else 0
                avg_coherent = sum(stats["coherent"]) / len(stats["coherent"]) if stats["coherent"] else 0

                writer.writerow({
                    "epoch": epoch,
                    "avg_aligned": f"{avg_aligned:.2f}",
                    "avg_coherent": f"{avg_coherent:.2f}",
                    "n_responses": len(stats["aligned"])
                })

                print(f"Epoch {epoch}: Aligned={avg_aligned:.2f}, Coherent={avg_coherent:.2f}")

        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Next step: Run visualization script to plot results")
        print(f"  python eval/visualize_epochs.py --data {summary_csv}")
    else:
        print(f"\nError: Generations file not found: {generations_csv}")

if __name__ == "__main__":
    main()
