#!/usr/bin/env python3
"""
Simple validation: Do contemporary people perform better on HellaSwag?

Tests the hypothesis that temporal grounding (contemporary vs historical personas)
affects performance on modern commonsense reasoning tasks.

Usage:
    python scripts/validate_era_hypothesis_simple.py

    # Or with custom settings:
    python scripts/validate_era_hypothesis_simple.py --num_people 10 --num_questions 200
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.loader import BenchmarkLoader
from src.benchmarks.evaluator import BenchmarkEvaluator
from src.utils.query_utils import ModelQueryInterface


def load_people_by_era(csv_path: str):
    """Load people grouped by era."""
    people_by_era = {
        "Contemporary": [],
        "Modern": [],
        "Early Modern": [],
        "Ancient": [],
        "Renaissance": []
    }

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            era = row.get("era", "Unknown")
            if era in people_by_era:
                people_by_era[era].append(name)

    return people_by_era


def evaluate_group(evaluator, people, questions, group_name):
    """Evaluate a group of people on the questions."""
    print(f"\nEvaluating {group_name}:")
    print("-" * 80)

    results = []
    for person in people:
        prefix = f"You are {person}. "
        result = evaluator.evaluate_with_prefix(
            questions, prefix, max_new_tokens=100, temperature=0.0
        )
        acc = result["accuracy"]
        results.append(acc)
        print(f"  {person:40s} {acc:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate era hypothesis on HellaSwag")
    parser.add_argument(
        "--num_people", type=int, default=5, help="Number of people per era group"
    )
    parser.add_argument(
        "--num_questions", type=int, default=100, help="Number of test questions"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 80)
    print("Era Hypothesis Validation on HellaSwag")
    print("=" * 80)
    print(f"\nHypothesis: Contemporary people outperform historical people")
    print(f"Settings:")
    print(f"  - People per group: {args.num_people}")
    print(f"  - Test questions: {args.num_questions}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Seed: {args.seed}")
    print()

    # Load model
    print("=" * 80)
    print("Loading model (this may take a few minutes)...")
    print("=" * 80)
    model = ModelQueryInterface()
    success = model.load_model(args.model_name)
    if not success:
        print("❌ Failed to load model")
        return

    # Load benchmark
    print("\n" + "=" * 80)
    print("Loading HellaSwag test set...")
    print("=" * 80)
    loader = BenchmarkLoader()
    test_questions = loader.load_benchmark(
        "hellaswag", "test", num_samples=args.num_questions, seed=args.seed
    )
    print(f"Loaded {len(test_questions)} test questions")

    evaluator = BenchmarkEvaluator(model)

    # Load people
    people_by_era = load_people_by_era("notable_people_10k.csv")

    print("\n" + "=" * 80)
    print("People available by era:")
    print("=" * 80)
    for era, names in people_by_era.items():
        print(f"  {era:20s} {len(names):4d} people")

    # Sample people from each era
    print("\n" + "=" * 80)
    print(f"Sampling {args.num_people} people from each era...")
    print("=" * 80)

    contemporary = random.sample(
        people_by_era["Contemporary"],
        min(args.num_people, len(people_by_era["Contemporary"]))
    )

    # For historical, mix from older eras
    historical_pool = (
        people_by_era.get("Early Modern", []) +
        people_by_era.get("Ancient", []) +
        people_by_era.get("Renaissance", [])
    )
    historical = random.sample(
        historical_pool,
        min(args.num_people, len(historical_pool))
    ) if historical_pool else []

    # Also test Modern (20th century) as a middle group
    modern = random.sample(
        people_by_era.get("Modern", []),
        min(args.num_people, len(people_by_era.get("Modern", [])))
    ) if people_by_era.get("Modern") else []

    print("\nSelected people:")
    print(f"  Contemporary: {', '.join(contemporary[:3])}{'...' if len(contemporary) > 3 else ''}")
    if modern:
        print(f"  Modern (20th): {', '.join(modern[:3])}{'...' if len(modern) > 3 else ''}")
    if historical:
        print(f"  Historical: {', '.join(historical[:3])}{'...' if len(historical) > 3 else ''}")

    # Evaluate each group
    print("\n" + "=" * 80)
    print("Running Evaluations (this will take several minutes)...")
    print("=" * 80)

    contemporary_results = evaluate_group(
        evaluator, contemporary, test_questions, "Contemporary (2000s-2020s)"
    )

    modern_results = []
    if modern:
        modern_results = evaluate_group(
            evaluator, modern, test_questions, "Modern (1900s)"
        )

    historical_results = []
    if historical:
        historical_results = evaluate_group(
            evaluator, historical, test_questions, "Historical (pre-1900)"
        )

    # Baseline
    print("\nEvaluating Baseline (no prefix):")
    print("-" * 80)
    baseline_result = evaluator.evaluate_with_prefix(
        test_questions, "", max_new_tokens=100, temperature=0.0
    )
    baseline_acc = baseline_result["accuracy"]
    print(f"  No prefix:                               {baseline_acc:.1%}")

    # Statistical summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    import statistics

    contemporary_avg = statistics.mean(contemporary_results)
    contemporary_std = statistics.stdev(contemporary_results) if len(contemporary_results) > 1 else 0

    print(f"\nContemporary (2000s-2020s):")
    print(f"  Mean:   {contemporary_avg:.1%}")
    print(f"  StdDev: {contemporary_std:.1%}")
    print(f"  vs Baseline: {(contemporary_avg - baseline_acc) * 100:+.1f}pp")

    if modern_results:
        modern_avg = statistics.mean(modern_results)
        modern_std = statistics.stdev(modern_results) if len(modern_results) > 1 else 0
        print(f"\nModern (1900s):")
        print(f"  Mean:   {modern_avg:.1%}")
        print(f"  StdDev: {modern_std:.1%}")
        print(f"  vs Baseline: {(modern_avg - baseline_acc) * 100:+.1f}pp")

    if historical_results:
        historical_avg = statistics.mean(historical_results)
        historical_std = statistics.stdev(historical_results) if len(historical_results) > 1 else 0
        print(f"\nHistorical (pre-1900):")
        print(f"  Mean:   {historical_avg:.1%}")
        print(f"  StdDev: {historical_std:.1%}")
        print(f"  vs Baseline: {(historical_avg - baseline_acc) * 100:+.1f}pp")

    print(f"\nBaseline (no prefix):")
    print(f"  Accuracy: {baseline_acc:.1%}")

    # Effect size
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if historical_results:
        diff = contemporary_avg - historical_avg
        print(f"\nContemporary vs Historical difference: {diff * 100:.1f}pp")

        if diff > 0.02:  # More than 2pp difference
            print("✓ SIGNIFICANT EFFECT DETECTED")
            print("  → Contemporary personas improve performance")
            print("  → Temporal grounding matters for modern commonsense tasks")
        elif diff > 0:
            print("→ MODEST EFFECT DETECTED")
            print("  → Small advantage for contemporary personas")
        else:
            print("✗ NO EFFECT DETECTED")
            print("  → Era does not appear to matter")

    improvement = contemporary_avg - baseline_acc
    if improvement > 0.01:
        print(f"\nContemporary vs Baseline: {improvement * 100:+.1f}pp improvement")
        print("✓ Prompt prefixes with contemporary people HELP")
    else:
        print("\n→ No clear advantage over baseline")

    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
