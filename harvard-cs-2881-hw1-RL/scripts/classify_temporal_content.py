#!/usr/bin/env python3
"""
Classify HellaSwag questions by temporal content using Llama 3.2.

This script analyzes each HellaSwag question to determine whether it contains
modern, 20th century, or timeless content. This allows us to test whether
contemporary vs historical personas show different performance on temporally-
relevant questions.

Usage:
    python scripts/classify_temporal_content.py --num_questions 200
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.loader import BenchmarkLoader
from src.utils.query_utils import ModelQueryInterface


CLASSIFICATION_PROMPT = """You are a temporal content classifier. Your job is to categorize questions based on the historical knowledge required to answer them.

Classify the following question into ONE of these categories:

**modern** - Requires knowledge from 2000s onwards (smartphones, social media, streaming, modern internet, TikTok, Instagram, etc.)

**20th_century** - Requires knowledge from 1900-1999 (cars, electricity, telephones, television, computers, but NOT smartphones/social media)

**timeless** - Could be understood and answered by someone from any historical era (cooking, basic hygiene, physical activities, universal human experiences)

Question: "{question}"

Respond in JSON format:
{{
  "category": "modern" | "20th_century" | "timeless",
  "reasoning": "Brief explanation (1-2 sentences) of why this question fits this category",
  "confidence": "high" | "medium" | "low"
}}

Only output valid JSON, nothing else."""


def classify_question(model: ModelQueryInterface, question_text: str) -> Dict:
    """
    Classify a single question using the LLM.

    Args:
        model: Loaded model interface
        question_text: The question to classify

    Returns:
        Dict with category, reasoning, and confidence
    """
    prompt = CLASSIFICATION_PROMPT.format(question=question_text)

    response = model.query_model(
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.0,  # Deterministic for consistency
    )

    # Try to extract JSON from response
    try:
        # Sometimes models wrap JSON in markdown code blocks
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        result = json.loads(json_str)

        # Validate required fields
        if "category" not in result:
            raise ValueError("Missing 'category' field")

        # Ensure category is valid
        valid_categories = ["modern", "20th_century", "timeless"]
        if result["category"] not in valid_categories:
            raise ValueError(f"Invalid category: {result['category']}")

        return result

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Warning: Failed to parse response for question: {question_text[:50]}...")
        print(f"Error: {e}")
        print(f"Response: {response}")

        # Fallback: try to extract category from text
        response_lower = response.lower()
        if "modern" in response_lower:
            category = "modern"
        elif "20th" in response_lower or "twentieth" in response_lower:
            category = "20th_century"
        elif "timeless" in response_lower:
            category = "timeless"
        else:
            category = "timeless"  # Default to timeless if unclear

        return {
            "category": category,
            "reasoning": "Fallback classification due to parse error",
            "confidence": "low"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Classify HellaSwag questions by temporal content"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=200,
        help="Number of questions to classify"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Model to use for classification (use Llama 3.2 11B or 90B)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/era_validation/hellaswag_temporal_classifications.csv",
        help="Output CSV file for classifications",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question sampling"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HellaSwag Temporal Content Classification")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  - Questions to classify: {args.num_questions}")
    print(f"  - Classifier model: {args.model_name}")
    print(f"  - Output file: {args.output_file}")
    print(f"  - Seed: {args.seed}")
    print()

    # Load model
    print("=" * 80)
    print("Loading classifier model (this may take a few minutes)...")
    print("=" * 80)
    model = ModelQueryInterface()
    success = model.load_model(args.model_name)
    if not success:
        print("❌ Failed to load model")
        return

    # Load questions
    print("\n" + "=" * 80)
    print("Loading HellaSwag test set...")
    print("=" * 80)
    loader = BenchmarkLoader()
    questions = loader.load_benchmark(
        "hellaswag", "test", num_samples=args.num_questions, seed=args.seed
    )
    print(f"Loaded {len(questions)} questions")

    # Classify each question
    print("\n" + "=" * 80)
    print("Classifying questions (this will take several minutes)...")
    print("=" * 80)

    classifications = []
    category_counts = {"modern": 0, "20th_century": 0, "timeless": 0}

    for i, q in enumerate(questions, 1):
        question_text = q["question"]

        # Show progress every 10 questions
        if i % 10 == 0:
            print(f"Progress: {i}/{len(questions)} questions ({i/len(questions):.0%})", end="\r", flush=True)

        # Classify
        result = classify_question(model, question_text)
        category = result["category"]
        category_counts[category] += 1

        # Store result
        classifications.append({
            "question_id": i - 1,  # 0-indexed
            "question": question_text,
            "choices": json.dumps(q.get("choices", [])),
            "correct_answer": q.get("correct_answer", -1),
            "category": category,
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", "unknown"),
        })

    print()  # Clear progress line

    # Save classifications
    print("\n" + "=" * 80)
    print("Saving classifications...")
    print("=" * 80)

    with open(args.output_file, 'w', newline='') as f:
        fieldnames = [
            "question_id", "question", "choices", "correct_answer",
            "category", "reasoning", "confidence"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(classifications)

    print(f"✓ Saved {len(classifications)} classifications to: {args.output_file}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)

    total = len(classifications)
    print(f"\nTotal questions classified: {total}")
    print(f"\nCategory distribution:")
    print(f"  Modern (2000s+):        {category_counts['modern']:3d} ({category_counts['modern']/total*100:.1f}%)")
    print(f"  20th Century (1900s):   {category_counts['20th_century']:3d} ({category_counts['20th_century']/total*100:.1f}%)")
    print(f"  Timeless (any era):     {category_counts['timeless']:3d} ({category_counts['timeless']/total*100:.1f}%)")

    # Show sample classifications for each category
    print("\n" + "=" * 80)
    print("SAMPLE CLASSIFICATIONS")
    print("=" * 80)

    for category in ["modern", "20th_century", "timeless"]:
        samples = [c for c in classifications if c["category"] == category][:3]
        if samples:
            print(f"\n{category.upper()}:")
            for s in samples:
                print(f"  Q: {s['question'][:70]}...")
                print(f"     Reason: {s['reasoning']}")

    print("\n" + "=" * 80)
    print("Classification complete!")
    print("=" * 80)
    print(f"\nNext step: Run temporal validation using:")
    print(f"  python scripts/validate_temporal_hypothesis.py --classified_questions {args.output_file}")


if __name__ == "__main__":
    main()
