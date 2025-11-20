#!/usr/bin/env python3
"""
Analyze temporal experiment results with flexible re-classification.

This script allows you to:
1. Load per-question baseline results
2. Apply different classification schemes
3. Recalculate accuracy by category
4. Explore 2x2 matrices (modern/timeless × technical/simple)

Usage:
    python scripts/analyze_temporal_results.py \
        --results outputs/era_validation/per_question_results.csv \
        --classifications hellaswag_corrected_classifications.csv
"""

import argparse
import csv
from collections import defaultdict


def load_results(results_path: str) -> dict:
    """Load per-question results from CSV."""
    results = {}
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row['question_id'])
            results[qid] = {
                'question': row['question'],
                'category': row['category'],
                'correct': row['correct'].lower() == 'true',
            }
    return results


def load_classifications(classifications_path: str) -> dict:
    """Load question classifications from CSV."""
    classifications = {}
    with open(classifications_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row['question_id'])
            classifications[qid] = row['category']
    return classifications


def calculate_accuracy(results: dict, category: str) -> tuple:
    """Calculate accuracy for a specific category."""
    filtered = [r for r in results.values() if r['category'] == category]
    if not filtered:
        return 0.0, 0
    correct = sum(1 for r in filtered if r['correct'])
    return correct / len(filtered), len(filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze temporal experiment results with flexible re-classification"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to per-question results CSV"
    )
    parser.add_argument(
        "--classifications",
        type=str,
        default=None,
        help="Path to updated classifications CSV (optional, for re-classification)"
    )
    parser.add_argument(
        "--corrections",
        type=str,
        default=None,
        help="Comma-separated list of question_id:new_category corrections (e.g., '7:timeless,15:modern')"
    )

    args = parser.parse_args()

    # Load results
    print("=" * 60)
    print("Loading per-question results...")
    print("=" * 60)
    results = load_results(args.results)
    print(f"Loaded {len(results)} question results")

    # Apply new classifications if provided
    if args.classifications:
        print(f"\nApplying classifications from: {args.classifications}")
        new_classifications = load_classifications(args.classifications)
        for qid, category in new_classifications.items():
            if qid in results:
                results[qid]['category'] = category
        print(f"Updated {len(new_classifications)} classifications")

    # Apply manual corrections if provided
    if args.corrections:
        print(f"\nApplying manual corrections...")
        corrections = args.corrections.split(',')
        for correction in corrections:
            qid_str, category = correction.split(':')
            qid = int(qid_str)
            if qid in results:
                old_cat = results[qid]['category']
                results[qid]['category'] = category
                print(f"  {qid}: {old_cat} → {category}")

    # Calculate accuracy by category
    print("\n" + "=" * 60)
    print("ACCURACY BY CATEGORY")
    print("=" * 60)

    categories = defaultdict(list)
    for r in results.values():
        categories[r['category']].append(r['correct'])

    for category in sorted(categories.keys()):
        correct_list = categories[category]
        n = len(correct_list)
        acc = sum(correct_list) / n if n > 0 else 0
        print(f"\n{category.upper()} ({n} questions):")
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Correct:  {sum(correct_list)}/{n}")

    # Calculate gap
    if 'modern' in categories and 'timeless' in categories:
        modern_acc = sum(categories['modern']) / len(categories['modern'])
        timeless_acc = sum(categories['timeless']) / len(categories['timeless'])
        gap = (timeless_acc - modern_acc) * 100

        print("\n" + "=" * 60)
        print("MODERN vs TIMELESS GAP")
        print("=" * 60)
        print(f"\nModern:   {modern_acc:.1%} ({len(categories['modern'])} questions)")
        print(f"Timeless: {timeless_acc:.1%} ({len(categories['timeless'])} questions)")
        print(f"Gap:      {gap:.1f}pp")

        if gap > 5:
            print(f"\n→ Timeless questions are {gap:.1f}pp easier than modern questions")
        elif gap < -5:
            print(f"\n→ Modern questions are {-gap:.1f}pp easier than timeless questions")
        else:
            print(f"\n→ No significant difference between categories")

    # Show breakdown of correct/incorrect by category
    print("\n" + "=" * 60)
    print("DETAILED BREAKDOWN")
    print("=" * 60)

    for category in sorted(categories.keys()):
        correct_qs = [qid for qid, r in results.items()
                      if r['category'] == category and r['correct']]
        incorrect_qs = [qid for qid, r in results.items()
                        if r['category'] == category and not r['correct']]

        print(f"\n{category.upper()}:")
        print(f"  Correct questions ({len(correct_qs)}): {correct_qs[:10]}{'...' if len(correct_qs) > 10 else ''}")
        print(f"  Incorrect questions ({len(incorrect_qs)}): {incorrect_qs[:10]}{'...' if len(incorrect_qs) > 10 else ''}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
