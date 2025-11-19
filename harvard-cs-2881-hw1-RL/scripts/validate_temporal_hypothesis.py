#!/usr/bin/env python3
"""
Validate temporal persona-awareness hypothesis.

Tests whether contemporary personas outperform historical personas specifically
on MODERN questions (not timeless ones), which would suggest the model can
restrict knowledge based on temporal context.

Hypothesis:
- Modern questions: Contemporary > Historical (if temporally aware)
- Timeless questions: Contemporary ≈ Historical (no expected difference)

Usage:
    python scripts/validate_temporal_hypothesis.py \
        --classified_questions outputs/era_validation/hellaswag_temporal_classifications.csv
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.evaluator import BenchmarkEvaluator
from src.utils.query_utils import ModelQueryInterface


# Famous people lists (same as fame-controlled validation)
FAMOUS_CONTEMPORARY = [
    "Steve Jobs", "Bill Gates", "Elon Musk", "Mark Zuckerberg", "Jeff Bezos",
    "Warren Buffett", "Tim Berners-Lee", "Nelson Mandela", "Martin Luther King Jr.",
    "The Beatles", "Bob Dylan", "Oprah Winfrey", "Stephen Hawking", "Neil Armstrong",
]

FAMOUS_HISTORICAL = [
    "Isaac Newton", "William Shakespeare", "Leonardo da Vinci", "Socrates",
    "Plato", "Aristotle", "Julius Caesar", "Confucius", "Archimedes",
    "Thomas Jefferson", "Galileo Galilei", "Michelangelo",
]


def load_classified_questions(csv_path: str) -> Dict[str, List[Dict]]:
    """
    Load questions grouped by temporal category.

    Returns:
        Dict mapping category to list of question dicts
    """
    questions_by_category = {
        "modern": [],
        "timeless": []
    }

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            if category in questions_by_category:
                # Reconstruct question dict in format expected by evaluator
                question = {
                    "type": "multiple_choice",
                    "question": row["question"],
                    "choices": json.loads(row["choices"]),
                    "correct_answer": int(row["correct_answer"]),
                    "answer_text": "",  # Not needed for evaluation
                }
                questions_by_category[category].append(question)

    return questions_by_category


def evaluate_group_on_subset(
    evaluator: BenchmarkEvaluator,
    people: List[str],
    questions: List[Dict],
    group_name: str,
    subset_name: str
):
    """Evaluate a group of people on a subset of questions."""
    print(f"\n{group_name} on {subset_name} questions:")
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
        description="Validate temporal persona-awareness hypothesis"
    )
    parser.add_argument(
        "--classified_questions",
        type=str,
        required=True,
        help="Path to CSV file with temporal classifications",
    )
    parser.add_argument(
        "--num_people",
        type=int,
        default=10,
        help="Number of people per era group"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 80)
    print("Temporal Persona-Awareness Validation")
    print("=" * 80)
    print(f"\nHypothesis:")
    print(f"  If models are temporally aware of personas:")
    print(f"    - Modern questions:   Contemporary > Historical")
    print(f"    - Timeless questions: Contemporary ≈ Historical")
    print(f"\nSettings:")
    print(f"  - People per group: {args.num_people}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Seed: {args.seed}")
    print()

    # Load classified questions
    print("=" * 80)
    print("Loading classified questions...")
    print("=" * 80)
    questions_by_category = load_classified_questions(args.classified_questions)

    for category, questions in questions_by_category.items():
        print(f"  {category:20s} {len(questions):3d} questions")

    # Load model
    print("\n" + "=" * 80)
    print("Loading model (this may take a few minutes)...")
    print("=" * 80)
    model = ModelQueryInterface()
    success = model.load_model(args.model_name)
    if not success:
        print("❌ Failed to load model")
        return

    evaluator = BenchmarkEvaluator(model)

    # Sample people
    print("\n" + "=" * 80)
    print(f"Sampling {args.num_people} famous people from each era...")
    print("=" * 80)

    contemporary = random.sample(
        FAMOUS_CONTEMPORARY,
        min(args.num_people, len(FAMOUS_CONTEMPORARY))
    )

    historical = random.sample(
        FAMOUS_HISTORICAL,
        min(args.num_people, len(FAMOUS_HISTORICAL))
    )

    print(f"\nContemporary: {', '.join(contemporary)}")
    print(f"Historical:   {', '.join(historical)}")

    # Test on MODERN questions
    print("\n" + "=" * 80)
    print("EVALUATING ON MODERN QUESTIONS (2000s+)")
    print("=" * 80)

    modern_questions = questions_by_category["modern"]
    print(f"Using {len(modern_questions)} modern questions")

    modern_contemporary_results = evaluate_group_on_subset(
        evaluator, contemporary, modern_questions,
        "Contemporary", "MODERN"
    )

    modern_historical_results = evaluate_group_on_subset(
        evaluator, historical, modern_questions,
        "Historical", "MODERN"
    )

    # Test on TIMELESS questions
    print("\n" + "=" * 80)
    print("EVALUATING ON TIMELESS QUESTIONS (any era)")
    print("=" * 80)

    timeless_questions = questions_by_category["timeless"]
    print(f"Using {len(timeless_questions)} timeless questions")

    timeless_contemporary_results = evaluate_group_on_subset(
        evaluator, contemporary, timeless_questions,
        "Contemporary", "TIMELESS"
    )

    timeless_historical_results = evaluate_group_on_subset(
        evaluator, historical, timeless_questions,
        "Historical", "TIMELESS"
    )

    # Baselines (no prefix)
    print("\n" + "=" * 80)
    print("BASELINE EVALUATIONS (no prefix)")
    print("=" * 80)

    print(f"\nBaseline on MODERN questions:")
    baseline_modern = evaluator.evaluate_with_prefix(
        modern_questions, "", max_new_tokens=100, temperature=0.0
    )
    print(f"  Accuracy: {baseline_modern['accuracy']:.1%}")

    print(f"\nBaseline on TIMELESS questions:")
    baseline_timeless = evaluator.evaluate_with_prefix(
        timeless_questions, "", max_new_tokens=100, temperature=0.0
    )
    print(f"  Accuracy: {baseline_timeless['accuracy']:.1%}")

    # Statistical analysis
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    import statistics

    def calc_stats(results):
        mean = statistics.mean(results)
        std = statistics.stdev(results) if len(results) > 1 else 0
        return mean, std

    # Modern questions
    modern_cont_mean, modern_cont_std = calc_stats(modern_contemporary_results)
    modern_hist_mean, modern_hist_std = calc_stats(modern_historical_results)

    # Timeless questions
    timeless_cont_mean, timeless_cont_std = calc_stats(timeless_contemporary_results)
    timeless_hist_mean, timeless_hist_std = calc_stats(timeless_historical_results)

    print(f"\nMODERN QUESTIONS ({len(modern_questions)} questions):")
    print(f"  Contemporary: {modern_cont_mean:.1%} (±{modern_cont_std:.1%})")
    print(f"  Historical:   {modern_hist_mean:.1%} (±{modern_hist_std:.1%})")
    print(f"  Difference:   {(modern_cont_mean - modern_hist_mean)*100:+.1f}pp")
    print(f"  Baseline:     {baseline_modern['accuracy']:.1%}")

    print(f"\nTIMELESS QUESTIONS ({len(timeless_questions)} questions):")
    print(f"  Contemporary: {timeless_cont_mean:.1%} (±{timeless_cont_std:.1%})")
    print(f"  Historical:   {timeless_hist_mean:.1%} (±{timeless_hist_std:.1%})")
    print(f"  Difference:   {(timeless_cont_mean - timeless_hist_mean)*100:+.1f}pp")
    print(f"  Baseline:     {baseline_timeless['accuracy']:.1%}")

    # Hypothesis test
    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST")
    print("=" * 80)

    modern_gap = modern_cont_mean - modern_hist_mean
    timeless_gap = timeless_cont_mean - timeless_hist_mean
    interaction = modern_gap - timeless_gap

    print(f"\nKey comparison: Does the gap differ by question type?")
    print(f"  Gap on modern questions:   {modern_gap*100:+.1f}pp")
    print(f"  Gap on timeless questions: {timeless_gap*100:+.1f}pp")
    print(f"  Interaction effect:        {interaction*100:+.1f}pp")

    print("\n" + "-" * 80)

    if interaction > 0.03:  # >3pp interaction
        print("✓ STRONG PERSONA-AWARENESS DETECTED!")
        print("  → Contemporary personas show LARGER advantage on modern questions")
        print("  → Models appear to restrict knowledge based on temporal context")
        print("  → This suggests sophisticated persona modeling")
    elif interaction > 0.01:  # >1pp interaction
        print("→ MODEST PERSONA-AWARENESS DETECTED")
        print("  → Some evidence that temporal context matters")
        print("  → Effect size is small but in expected direction")
    elif interaction > -0.01:  # Within ±1pp
        print("✗ NO PERSONA-AWARENESS DETECTED")
        print("  → Contemporary advantage is similar across question types")
        print("  → Personas don't appear to restrict knowledge temporally")
    else:  # Negative interaction
        print("✗ OPPOSITE PATTERN DETECTED")
        print("  → Historical personas do BETTER on modern questions?!")
        print("  → This contradicts the persona-awareness hypothesis")

    # Additional insights
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if modern_gap > 0.02 and timeless_gap < 0.01:
        print("\n✓ Best case scenario:")
        print("  Contemporary personas help on modern questions")
        print("  No advantage on timeless questions")
        print("  → Suggests model has temporal awareness")
    elif modern_gap > 0.02 and timeless_gap > 0.02:
        print("\n→ General persona effect:")
        print("  Contemporary personas help on ALL questions")
        print("  Not specific to modern content")
        print("  → May be due to fame, training data, or general capability")
    elif abs(modern_gap) < 0.01 and abs(timeless_gap) < 0.01:
        print("\n✗ No effect detected:")
        print("  Personas don't seem to matter for either question type")
        print("  → Model may be ignoring persona context")

    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
