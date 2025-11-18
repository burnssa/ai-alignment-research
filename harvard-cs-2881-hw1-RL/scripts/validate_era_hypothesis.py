#!/usr/bin/env python3
"""
Validate the era hypothesis: Contemporary people perform better on HellaSwag.

This script tests whether the observed pattern is statistically significant by:
1. Sampling contemporary vs historical people randomly
2. Evaluating on HellaSwag test set
3. Computing statistical significance
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from src.benchmarks.loader import BenchmarkLoader
from src.benchmarks.evaluator import BenchmarkEvaluator
from src.utils.query_utils import ModelQueryInterface
import csv


def load_people_by_era(csv_path: str):
    """Load people grouped by era."""
    people_by_era = {"Contemporary": [], "Modern": [], "Early Modern": [], "Ancient": []}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            era = row.get("era", "Unknown")
            if era in people_by_era:
                people_by_era[era].append(row["name"])

    return people_by_era


def main():
    print("=" * 80)
    print("Era Hypothesis Validation")
    print("=" * 80)
    print("\nHypothesis: Contemporary people outperform historical people on HellaSwag")
    print("because the benchmark tests modern commonsense reasoning.\n")

    # Load model
    print("Loading model...")
    model = ModelQueryInterface()
    model.load_model("meta-llama/Llama-3.1-8B-Instruct")

    # Load benchmark
    print("Loading HellaSwag test set...")
    loader = BenchmarkLoader()
    test_questions = loader.load_benchmark("hellaswag", "test", num_samples=100, seed=42)

    evaluator = BenchmarkEvaluator(model)

    # Load people
    people_by_era = load_people_by_era("notable_people_10k.csv")

    print(f"\nPeople available:")
    for era, names in people_by_era.items():
        print(f"  {era}: {len(names)} people")

    # Test Contemporary vs Historical
    print("\n" + "=" * 80)
    print("Testing Contemporary vs Historical (5 people each)...")
    print("=" * 80)

    contemporary = random.sample(people_by_era["Contemporary"], 5)
    historical = (
        random.sample(people_by_era.get("Early Modern", []), 2) +
        random.sample(people_by_era.get("Ancient", []), 3)
    ) if len(people_by_era.get("Early Modern", [])) >= 2 else []

    results = {"Contemporary": [], "Historical": []}

    # Test Contemporary
    print("\nContemporary People:")
    for person in contemporary:
        prefix = f"You are {person}. "
        result = evaluator.evaluate_with_prefix(test_questions, prefix, max_new_tokens=100)
        acc = result["accuracy"]
        results["Contemporary"].append(acc)
        print(f"  {person:35s} {acc:.1%}")

    # Test Historical
    if historical:
        print("\nHistorical People:")
        for person in historical:
            prefix = f"You are {person}. "
            result = evaluator.evaluate_with_prefix(test_questions, prefix, max_new_tokens=100)
            acc = result["accuracy"]
            results["Historical"].append(acc)
            print(f"  {person:35s} {acc:.1%}")

    # Test baseline
    print("\nBaseline (no prefix):")
    baseline_result = evaluator.evaluate_with_prefix(test_questions, "", max_new_tokens=100)
    baseline_acc = baseline_result["accuracy"]
    print(f"  No prefix:                         {baseline_acc:.1%}")

    # Statistical summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)

    import statistics
    contemporary_avg = statistics.mean(results["Contemporary"])
    print(f"Contemporary average: {contemporary_avg:.1%}")

    if results["Historical"]:
        historical_avg = statistics.mean(results["Historical"])
        print(f"Historical average:   {historical_avg:.1%}")
        print(f"Difference:           {(contemporary_avg - historical_avg) * 100:.1f}pp")

    print(f"Baseline:             {baseline_acc:.1%}")
    print(f"Contemporary vs Base: {(contemporary_avg - baseline_acc) * 100:+.1f}pp")

    print("\nConclusion:")
    if contemporary_avg > baseline_acc:
        print("✓ Contemporary people DO perform better!")
        print("  → Temporal grounding appears to help with modern commonsense.")
    else:
        print("✗ No significant advantage detected.")
        print("  → May need more samples or different benchmark.")


if __name__ == "__main__":
    main()
