#!/usr/bin/env python3
"""
CORRECTED Preference data analysis comparing Superjective vs HH-RLHF datasets.

KEY FIX: Extracts only final responses from HH-RLHF multi-turn conversations
to avoid inflated similarity scores from identical conversation context.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import Dict, List
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "./analysis"
SUPERJECTIVE_DATA = "./data/superjective_full_with_metadata.json"

def compute_embeddings(texts: List[str], model) -> np.ndarray:
    """Compute embeddings in batches."""
    return model.encode(texts, show_progress_bar=True, batch_size=32)

def extract_final_response(conversation: str) -> str:
    """Extract only the final assistant response from HH-RLHF format."""
    parts = conversation.split('\n\nAssistant:')
    if len(parts) > 1:
        return parts[-1].strip()
    return conversation

def analyze_preference_strength(chosen_texts: List[str], rejected_texts: List[str],
                               name: str, model) -> Dict:
    """
    Analyze preference strength using cosine similarity on responses only.

    Lower similarity = stronger preference signal (more different responses)
    """
    print(f"\nAnalyzing preference strength for {name}...")
    print(f"  Processing {len(chosen_texts)} response pairs")

    # Compute embeddings
    print("  Computing chosen embeddings...")
    chosen_embeds = compute_embeddings(chosen_texts, model)

    print("  Computing rejected embeddings...")
    rejected_embeds = compute_embeddings(rejected_texts, model)

    # Calculate cosine similarities
    similarities = []
    for i in range(len(chosen_embeds)):
        sim = 1 - cosine(chosen_embeds[i], rejected_embeds[i])
        similarities.append(sim)

    similarities = np.array(similarities)

    return {
        "mean_similarity": float(np.mean(similarities)),
        "std_similarity": float(np.std(similarities)),
        "median_similarity": float(np.median(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "similarities": similarities.tolist()
    }

def analyze_lengths(chosen_texts: List[str], rejected_texts: List[str], name: str) -> Dict:
    """Analyze response length distributions."""
    print(f"\nAnalyzing lengths for {name}...")

    chosen_lengths = [len(text) for text in chosen_texts]
    rejected_lengths = [len(text) for text in rejected_texts]
    length_diffs = [c - r for c, r in zip(chosen_lengths, rejected_lengths)]

    return {
        "chosen": {
            "mean": float(np.mean(chosen_lengths)),
            "std": float(np.std(chosen_lengths)),
            "median": float(np.median(chosen_lengths)),
            "min": int(np.min(chosen_lengths)),
            "max": int(np.max(chosen_lengths))
        },
        "rejected": {
            "mean": float(np.mean(rejected_lengths)),
            "std": float(np.std(rejected_lengths)),
            "median": float(np.median(rejected_lengths)),
            "min": int(np.min(rejected_lengths)),
            "max": int(np.max(rejected_lengths))
        },
        "difference": {
            "mean": float(np.mean(length_diffs)),
            "std": float(np.std(length_diffs)),
            "median": float(np.median(length_diffs))
        },
        "chosen_lengths": chosen_lengths,
        "rejected_lengths": rejected_lengths,
        "length_diffs": length_diffs
    }

def create_visualizations(superjective_stats, baseline_stats, output_dir):
    """Create comparison visualizations."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}...")

    # Set style
    sns.set_style("whitegrid")

    # 1. Preference Strength (Cosine Similarity) Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(superjective_stats['preference_strength']['similarities'],
                 bins=50, alpha=0.7, label='Superjective', color='purple')
    axes[0].axvline(superjective_stats['preference_strength']['mean_similarity'],
                    color='purple', linestyle='--', linewidth=2,
                    label=f'Mean: {superjective_stats["preference_strength"]["mean_similarity"]:.3f}')
    axes[0].set_xlabel('Cosine Similarity (Chosen vs Rejected)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Superjective: Preference Strength\n(Lower = Stronger Signal)')
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    axes[1].hist(baseline_stats['preference_strength']['similarities'],
                 bins=50, alpha=0.7, label='HH-RLHF', color='green')
    axes[1].axvline(baseline_stats['preference_strength']['mean_similarity'],
                    color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {baseline_stats["preference_strength"]["mean_similarity"]:.3f}')
    axes[1].set_xlabel('Cosine Similarity (Chosen vs Rejected)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('HH-RLHF: Preference Strength\n(Lower = Stronger Signal)')
    axes[1].legend()
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/preference_strength_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ preference_strength_comparison.png")

    # 2. Response Length Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Superjective
    axes[0, 0].hist(superjective_stats['lengths']['chosen_lengths'],
                    bins=50, alpha=0.7, color='purple', label='Chosen')
    axes[0, 0].hist(superjective_stats['lengths']['rejected_lengths'],
                    bins=50, alpha=0.7, color='orange', label='Rejected')
    axes[0, 0].set_xlabel('Response Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Superjective: Response Lengths')
    axes[0, 0].set_xlim(0, 8000)
    axes[0, 0].legend()

    axes[0, 1].hist(superjective_stats['lengths']['length_diffs'],
                    bins=50, alpha=0.7, color='blue')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Length Difference (Chosen - Rejected)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Superjective: Length Differences\nMean: {superjective_stats["lengths"]["difference"]["mean"]:.0f} chars')
    axes[0, 1].set_xlim(-5000, 5000)

    # HH-RLHF
    axes[1, 0].hist(baseline_stats['lengths']['chosen_lengths'],
                    bins=50, alpha=0.7, color='purple', label='Chosen')
    axes[1, 0].hist(baseline_stats['lengths']['rejected_lengths'],
                    bins=50, alpha=0.7, color='orange', label='Rejected')
    axes[1, 0].set_xlabel('Response Length (characters)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('HH-RLHF: Response Lengths')
    axes[1, 0].set_xlim(0, 8000)
    axes[1, 0].legend()

    axes[1, 1].hist(baseline_stats['lengths']['length_diffs'],
                    bins=50, alpha=0.7, color='blue')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Length Difference (Chosen - Rejected)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'HH-RLHF: Length Differences\nMean: {baseline_stats["lengths"]["difference"]["mean"]:.0f} chars')
    axes[1, 1].set_xlim(-5000, 5000)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_length_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ response_length_distribution.png")

def main():
    print("="*70)
    print("CORRECTED PREFERENCE DATA ANALYSIS")
    print("Extracts only final responses from HH-RLHF conversations")
    print("="*70)

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded")

    # Load Superjective
    print("\n" + "="*70)
    print("LOADING SUPERJECTIVE DATA")
    print("="*70)
    with open(SUPERJECTIVE_DATA, 'r') as f:
        superjective_data = json.load(f)

    spj_chosen = [ex['chosen'] for ex in superjective_data]
    spj_rejected = [ex['rejected'] for ex in superjective_data]
    print(f"Loaded {len(superjective_data)} Superjective examples")

    # Load HH-RLHF and extract final responses only
    print("\n" + "="*70)
    print("LOADING HH-RLHF DATA (Extracting final responses only)")
    print("="*70)
    baseline_dataset = load_dataset('Anthropic/hh-rlhf', split='train[:1000]')

    hh_chosen = []
    hh_rejected = []
    for ex in baseline_dataset:
        chosen_resp = extract_final_response(ex['chosen'])
        rejected_resp = extract_final_response(ex['rejected'])
        hh_chosen.append(chosen_resp)
        hh_rejected.append(rejected_resp)

    print(f"Loaded {len(hh_chosen)} HH-RLHF response pairs")
    print(f"Example HH-RLHF response length: {len(hh_chosen[0])} chars")

    # Analyze Superjective
    print("\n" + "="*70)
    print("ANALYZING SUPERJECTIVE")
    print("="*70)
    superjective_stats = {
        'preference_strength': analyze_preference_strength(spj_chosen, spj_rejected,
                                                          'Superjective', model),
        'lengths': analyze_lengths(spj_chosen, spj_rejected, 'Superjective')
    }

    # Analyze HH-RLHF
    print("\n" + "="*70)
    print("ANALYZING HH-RLHF")
    print("="*70)
    baseline_stats = {
        'preference_strength': analyze_preference_strength(hh_chosen, hh_rejected,
                                                          'HH-RLHF', model),
        'lengths': analyze_lengths(hh_chosen, hh_rejected, 'HH-RLHF')
    }

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nPreference Strength (Cosine Similarity - Lower = Stronger):")
    print(f"  Superjective: {superjective_stats['preference_strength']['mean_similarity']:.3f}")
    print(f"  HH-RLHF:      {baseline_stats['preference_strength']['mean_similarity']:.3f}")

    print("\nResponse Lengths (characters):")
    print(f"  Superjective chosen:  {superjective_stats['lengths']['chosen']['mean']:.0f}")
    print(f"  Superjective rejected: {superjective_stats['lengths']['rejected']['mean']:.0f}")
    print(f"  HH-RLHF chosen:       {baseline_stats['lengths']['chosen']['mean']:.0f}")
    print(f"  HH-RLHF rejected:     {baseline_stats['lengths']['rejected']['mean']:.0f}")

    # Create visualizations
    create_visualizations(superjective_stats, baseline_stats, OUTPUT_DIR)

    # Save results
    output = {
        "superjective": superjective_stats,
        "baseline": baseline_stats,
        "note": "CORRECTED: HH-RLHF metrics calculated on final responses only, not full conversations"
    }

    output_path = f"{OUTPUT_DIR}/full_analysis_corrected.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print("="*70)

if __name__ == "__main__":
    main()
