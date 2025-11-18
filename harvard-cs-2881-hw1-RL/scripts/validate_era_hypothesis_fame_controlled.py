#!/usr/bin/env python3
"""
Fame-controlled validation: Do contemporary people perform better when matched by fame level?

This script controls for notability by only using highly recognizable "household name"
figures from each era, addressing the confound that historical figures in the dataset
tend to be more famous than contemporary ones.

Usage:
    python scripts/validate_era_hypothesis_fame_controlled.py
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


# Curated lists of highly recognizable "household name" figures
# Criteria: Would the average person recognize this name?
# NOTE: All names verified to exist in notable_people_10k.csv
FAMOUS_CONTEMPORARY = [
    # Tech & Business (Silicon Valley era)
    "Steve Jobs", "Bill Gates", "Elon Musk", "Mark Zuckerberg", "Jeff Bezos",
    "Warren Buffett", "Tim Berners-Lee",
    # Politics & Leadership
    "Nelson Mandela", "Martin Luther King Jr.", "Margaret Thatcher",
    # Entertainment & Music
    "The Beatles", "Bob Dylan", "Oprah Winfrey", "Steven Spielberg",
    "Andy Warhol", "Michael Jackson", "Madonna",
    # Science & Exploration
    "Stephen Hawking", "Jane Goodall", "Neil Armstrong", "Buzz Aldrin",
    "Carl Sagan", "Jonas Salk",
    # Literature
    "Maya Angelou", "Toni Morrison", "Gabriel García Márquez",
]

FAMOUS_MODERN = [
    # Science (20th century)
    "Albert Einstein", "Marie Curie", "Nikola Tesla", "Sigmund Freud",
    "Richard Feynman", "Niels Bohr", "Enrico Fermi",
    # Politics & Leadership
    "Winston Churchill", "Mahatma Gandhi", "Franklin D. Roosevelt",
    "Theodore Roosevelt", "Eleanor Roosevelt",
    # Arts & Literature
    "Pablo Picasso", "Ernest Hemingway", "Virginia Woolf", "F. Scott Fitzgerald",
    "George Orwell", "J.R.R. Tolkien", "Charlie Chaplin",
    # Others
    "Amelia Earhart", "Helen Keller", "Mother Teresa",
]

FAMOUS_HISTORICAL = [
    # Early Modern (1500-1900)
    "Isaac Newton", "William Shakespeare", "Wolfgang Amadeus Mozart",
    "Ludwig van Beethoven", "Charles Darwin", "Galileo Galilei",
    "Benjamin Franklin", "Abraham Lincoln", "George Washington",
    "Thomas Jefferson", "Napoleon Bonaparte", "Beethoven",
    # Renaissance
    "Leonardo da Vinci", "Michelangelo", "Nicolaus Copernicus",
    # Ancient/Classical
    "Socrates", "Plato", "Aristotle", "Julius Caesar",
    "Alexander the Great", "Confucius", "Archimedes",
]


def load_people_by_fame_controlled_era(csv_path: str):
    """
    Load people from CSV, filtering to only famous household names.

    Returns:
        Dict mapping era to list of famous people in that era
    """
    # Load all people
    all_people = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_people[row["name"]] = row.get("era", "Unknown")

    # Filter to famous people that exist in our dataset
    result = {
        "Contemporary": [],
        "Modern": [],
        "Historical": [],
    }

    for name in FAMOUS_CONTEMPORARY:
        if name in all_people and all_people[name] == "Contemporary":
            result["Contemporary"].append(name)

    for name in FAMOUS_MODERN:
        if name in all_people and all_people[name] == "Modern":
            result["Modern"].append(name)

    for name in FAMOUS_HISTORICAL:
        era = all_people.get(name)
        if era in ["Early Modern", "Ancient", "Renaissance"]:
            result["Historical"].append(name)

    return result


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
    parser = argparse.ArgumentParser(
        description="Fame-controlled validation of era hypothesis"
    )
    parser.add_argument(
        "--num_people", type=int, default=10, help="Number of people per era group"
    )
    parser.add_argument(
        "--num_questions", type=int, default=200, help="Number of test questions"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="fame_controlled",
        help="Suffix for output log file",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 80)
    print("Fame-Controlled Era Hypothesis Validation")
    print("=" * 80)
    print(f"\nHypothesis: When controlling for fame, contemporary people outperform")
    print(f"historical people on modern commonsense reasoning tasks.")
    print(f"\nSettings:")
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

    # Load famous people
    people_by_era = load_people_by_fame_controlled_era("notable_people_10k.csv")

    print("\n" + "=" * 80)
    print("Famous people available by era:")
    print("=" * 80)
    for era, names in people_by_era.items():
        print(f"  {era:20s} {len(names):4d} household names")
        print(f"    Sample: {', '.join(names[:3])}")

    # Sample people from each era
    print("\n" + "=" * 80)
    print(f"Sampling {args.num_people} famous people from each era...")
    print("=" * 80)

    contemporary = random.sample(
        people_by_era["Contemporary"],
        min(args.num_people, len(people_by_era["Contemporary"])),
    )

    modern = random.sample(
        people_by_era["Modern"],
        min(args.num_people, len(people_by_era["Modern"])),
    )

    historical = random.sample(
        people_by_era["Historical"],
        min(args.num_people, len(people_by_era["Historical"])),
    )

    print("\nSelected famous people:")
    print(f"  Contemporary: {', '.join(contemporary)}")
    print(f"  Modern:       {', '.join(modern)}")
    print(f"  Historical:   {', '.join(historical)}")

    # Evaluate each group
    print("\n" + "=" * 80)
    print("Running Evaluations...")
    print("=" * 80)

    contemporary_results = evaluate_group(
        evaluator, contemporary, test_questions, "Contemporary (Famous)"
    )

    modern_results = evaluate_group(
        evaluator, modern, test_questions, "Modern (Famous)"
    )

    historical_results = evaluate_group(
        evaluator, historical, test_questions, "Historical (Famous)"
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
    contemporary_std = (
        statistics.stdev(contemporary_results) if len(contemporary_results) > 1 else 0
    )

    modern_avg = statistics.mean(modern_results)
    modern_std = statistics.stdev(modern_results) if len(modern_results) > 1 else 0

    historical_avg = statistics.mean(historical_results)
    historical_std = (
        statistics.stdev(historical_results) if len(historical_results) > 1 else 0
    )

    print(f"\nContemporary (Famous):")
    print(f"  Mean:   {contemporary_avg:.1%}")
    print(f"  StdDev: {contemporary_std:.1%}")
    print(f"  vs Baseline: {(contemporary_avg - baseline_acc) * 100:+.1f}pp")

    print(f"\nModern (Famous):")
    print(f"  Mean:   {modern_avg:.1%}")
    print(f"  StdDev: {modern_std:.1%}")
    print(f"  vs Baseline: {(modern_avg - baseline_acc) * 100:+.1f}pp")

    print(f"\nHistorical (Famous):")
    print(f"  Mean:   {historical_avg:.1%}")
    print(f"  StdDev: {historical_std:.1%}")
    print(f"  vs Baseline: {(historical_avg - baseline_acc) * 100:+.1f}pp")

    print(f"\nBaseline (no prefix):")
    print(f"  Accuracy: {baseline_acc:.1%}")

    # Effect size
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    cont_vs_hist = contemporary_avg - historical_avg
    cont_vs_mod = contemporary_avg - modern_avg

    print(f"\nContemporary vs Historical difference: {cont_vs_hist * 100:+.1f}pp")
    print(f"Contemporary vs Modern difference:     {cont_vs_mod * 100:+.1f}pp")

    if cont_vs_hist > 0.02:  # More than 2pp difference
        print("\n✓ SIGNIFICANT EFFECT DETECTED")
        print("  → Fame-controlled contemporary personas improve performance")
        print("  → Temporal grounding matters for modern commonsense tasks")
    elif cont_vs_hist > 0:
        print("\n→ MODEST EFFECT DETECTED")
        print("  → Small advantage for contemporary personas")
    else:
        print("\n✗ NO EFFECT DETECTED")
        print("  → Even with fame control, era does not appear to matter")
        print("  → Fame/notability may have been the true confound")

    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
