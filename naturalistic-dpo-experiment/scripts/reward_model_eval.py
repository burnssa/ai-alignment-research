#!/usr/bin/env python3
"""
Reward Model evaluation to quantify magnitude of quality differences.

Uses OpenAssistant's pre-trained reward model to score responses on a continuous scale.
This shows HOW MUCH BETTER one model is, not just which is better.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict
from tqdm import tqdm

# Configuration
EVAL_PROMPTS_PATH = "./data/eval_prompts.json"
BASELINE_RESPONSES_PATH = "./data/baseline_responses.json"
SUPERJECTIVE_RESPONSES_PATH = "./data/superjective_responses.json"
RESULTS_PATH = "./data/reward_model_results.json"

# Reward model configuration
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8

def load_json(path: str) -> List[Dict]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: Dict, path: str):
    """Save JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def score_response(prompt: str, response: str, model, tokenizer) -> float:
    """
    Score a single response using the reward model.

    Returns:
        float: Reward score (higher = better quality)
    """
    # Format as conversation
    text = f"Human: {prompt}\n\nAssistant: {response}"

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)

    # Get score
    with torch.no_grad():
        score = model(**inputs).logits[0].cpu().item()

    return score

def batch_score_responses(
    prompts: List[str],
    responses: List[str],
    model,
    tokenizer
) -> List[float]:
    """
    Score multiple responses in batches for efficiency.

    Returns:
        List[float]: Reward scores
    """
    scores = []

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Scoring responses"):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_responses = responses[i:i + BATCH_SIZE]

        # Format as conversations
        texts = [
            f"Human: {p}\n\nAssistant: {r}"
            for p, r in zip(batch_prompts, batch_responses)
        ]

        # Tokenize batch
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(DEVICE)

        # Get scores
        with torch.no_grad():
            batch_scores = model(**inputs).logits.squeeze(-1).cpu().tolist()

        # Handle single item case
        if not isinstance(batch_scores, list):
            batch_scores = [batch_scores]

        scores.extend(batch_scores)

    return scores

def calculate_statistics(scores: List[float]) -> Dict:
    """Calculate statistical metrics for scores."""
    scores_array = np.array(scores)
    return {
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
        "median": float(np.median(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "count": len(scores)
    }

def statistical_significance(scores_a: List[float], scores_b: List[float]) -> Dict:
    """
    Test if difference between two score distributions is statistically significant.

    Uses paired t-test since same prompts are used for both models.
    """
    from scipy import stats

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_b, scores_a)  # b - a (Superjective - Baseline)

    # Effect size (Cohen's d for paired samples)
    differences = np.array(scores_b) - np.array(scores_a)
    cohens_d = np.mean(differences) / np.std(differences)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "is_significant_p05": bool(p_value < 0.05),
        "is_significant_p01": bool(p_value < 0.01),
        "cohens_d": float(cohens_d),
        "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    }

def main():
    print("="*60)
    print("REWARD MODEL EVALUATION")
    print("="*60)

    # Load data
    print("\nLoading data...")
    prompts_data = load_json(EVAL_PROMPTS_PATH)
    baseline_responses = load_json(BASELINE_RESPONSES_PATH)
    superjective_responses = load_json(SUPERJECTIVE_RESPONSES_PATH)

    print(f"  {len(prompts_data)} prompts")
    print(f"  {len(baseline_responses)} baseline responses")
    print(f"  {len(superjective_responses)} Superjective responses")

    # Extract prompts and responses in order
    prompts = [p['prompt'] for p in prompts_data]
    baseline_texts = [r['response'] for r in baseline_responses]
    superjective_texts = [r['response'] for r in superjective_responses]

    # Load reward model
    print(f"\nLoading reward model: {REWARD_MODEL_NAME}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    print("✓ Model loaded")

    # Score baseline responses
    print("\nScoring baseline responses...")
    baseline_scores = batch_score_responses(prompts, baseline_texts, model, tokenizer)

    # Score Superjective responses
    print("\nScoring Superjective responses...")
    superjective_scores = batch_score_responses(prompts, superjective_texts, model, tokenizer)

    # Calculate statistics
    print("\nCalculating statistics...")
    baseline_stats = calculate_statistics(baseline_scores)
    superjective_stats = calculate_statistics(superjective_scores)

    # Statistical significance
    significance = statistical_significance(baseline_scores, superjective_scores)

    # Per-example results
    detailed_results = []
    wins = {"baseline": 0, "superjective": 0, "tie": 0}

    for i, (prompt, b_score, s_score) in enumerate(zip(prompts, baseline_scores, superjective_scores)):
        difference = s_score - b_score

        # Determine winner (using 0.01 threshold for ties)
        if abs(difference) < 0.01:
            winner = "tie"
            wins["tie"] += 1
        elif difference > 0:
            winner = "superjective"
            wins["superjective"] += 1
        else:
            winner = "baseline"
            wins["baseline"] += 1

        detailed_results.append({
            "id": i,
            "prompt": prompt[:100] + "...",
            "baseline_score": b_score,
            "superjective_score": s_score,
            "difference": difference,
            "winner": winner
        })

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nBaseline Model:")
    print(f"  Mean score: {baseline_stats['mean']:.4f} ± {baseline_stats['std']:.4f}")
    print(f"  Median: {baseline_stats['median']:.4f}")
    print(f"  Range: [{baseline_stats['min']:.4f}, {baseline_stats['max']:.4f}]")

    print(f"\nSuperjective Model:")
    print(f"  Mean score: {superjective_stats['mean']:.4f} ± {superjective_stats['std']:.4f}")
    print(f"  Median: {superjective_stats['median']:.4f}")
    print(f"  Range: [{superjective_stats['min']:.4f}, {superjective_stats['max']:.4f}]")

    mean_diff = superjective_stats['mean'] - baseline_stats['mean']
    pct_improvement = (mean_diff / abs(baseline_stats['mean'])) * 100 if baseline_stats['mean'] != 0 else 0

    print(f"\nDifference:")
    print(f"  Absolute: {mean_diff:+.4f}")
    print(f"  Relative: {pct_improvement:+.1f}%")

    print(f"\nStatistical Significance:")
    print(f"  t-statistic: {significance['t_statistic']:.4f}")
    print(f"  p-value: {significance['p_value']:.4f}")
    print(f"  Significant (p<0.05): {'YES ✓' if significance['is_significant_p05'] else 'NO ✗'}")
    print(f"  Significant (p<0.01): {'YES ✓✓' if significance['is_significant_p01'] else 'NO'}")
    print(f"  Cohen's d: {significance['cohens_d']:.4f} ({significance['effect_size']} effect)")

    print(f"\nWin Count (reward-based):")
    total_wins = wins['baseline'] + wins['superjective']
    print(f"  Baseline wins: {wins['baseline']} ({wins['baseline']/total_wins*100:.1f}%)")
    print(f"  Superjective wins: {wins['superjective']} ({wins['superjective']/total_wins*100:.1f}%)")
    print(f"  Ties: {wins['tie']}")

    print("\n" + "="*60)

    # Determine overall winner
    if mean_diff > 0 and significance['is_significant_p05']:
        print(f"✓✓ Superjective model is SIGNIFICANTLY BETTER")
        print(f"   Mean improvement: {mean_diff:+.4f} ({pct_improvement:+.1f}%)")
    elif mean_diff > 0:
        print(f"✓ Superjective model is better (not statistically significant)")
    elif mean_diff < 0 and significance['is_significant_p05']:
        print(f"✗ Baseline model is significantly better")
    else:
        print(f"= Models are roughly equivalent")

    print("="*60)

    # Save results
    output = {
        "baseline_statistics": baseline_stats,
        "superjective_statistics": superjective_stats,
        "difference": {
            "absolute": mean_diff,
            "relative_pct": pct_improvement
        },
        "statistical_significance": significance,
        "win_count": wins,
        "detailed_results": detailed_results
    }

    save_json(output, RESULTS_PATH)
    print(f"\nResults saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
