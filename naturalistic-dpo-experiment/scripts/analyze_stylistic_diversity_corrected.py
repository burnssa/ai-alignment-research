#!/usr/bin/env python3
"""
CORRECTED Stylistic Diversity Analysis for preference datasets.

KEY FIX: Extracts only final responses from HH-RLHF multi-turn conversations
before analyzing stylistic features.

Analyzes what differentiates chosen from rejected responses:
- Lexical diversity (vocabulary richness)
- Readability scores
- Sentiment/tone
- N-gram overlap
- Formatting patterns
"""

import numpy as np
import json
from datasets import load_dataset
from collections import Counter
import re
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Configuration
SUPERJECTIVE_DATA = "./data/superjective_full_with_metadata.json"
OUTPUT_DIR = "./analysis"

def extract_final_response(conversation: str) -> str:
    """Extract only the final assistant response from HH-RLHF format."""
    parts = conversation.split('\n\nAssistant:')
    if len(parts) > 1:
        return parts[-1].strip()
    return conversation

def lexical_diversity(text: str) -> Dict:
    """Calculate lexical diversity metrics."""
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)

    # Type-Token Ratio (TTR)
    ttr = len(unique_words) / len(words) if words else 0

    # Mean word length
    mean_word_length = np.mean([len(w) for w in words]) if words else 0

    return {
        'ttr': ttr,
        'unique_words': len(unique_words),
        'total_words': len(words),
        'mean_word_length': mean_word_length
    }

def readability_scores(text: str) -> Dict:
    """Calculate readability metrics."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]

    words = re.findall(r'\b\w+\b', text)

    if not sentences or not words:
        return {
            'avg_sentence_length': 0,
            'avg_word_length': 0,
            'flesch_reading_ease_approx': 0,
            'flesch_kincaid_grade_level': 0
        }

    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = np.mean([len(w) for w in words])

    # Simplified Flesch Reading Ease (higher = easier)
    flesch_approx = 206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / 5)

    # Flesch-Kincaid Grade Level (outputs U.S. grade level)
    fk_grade = 0.39 * avg_sentence_length + 11.8 * (avg_word_length / 5) - 15.59

    return {
        'avg_sentence_length': avg_sentence_length,
        'avg_word_length': avg_word_length,
        'flesch_reading_ease_approx': flesch_approx,
        'flesch_kincaid_grade_level': fk_grade
    }

def sentiment_tone(text: str) -> Dict:
    """Simple sentiment/tone analysis based on lexical cues."""
    text_lower = text.lower()

    # Count total words for normalization
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = len(words)

    # Positive words
    positive_words = ['good', 'great', 'excellent', 'best', 'perfect', 'wonderful',
                     'happy', 'love', 'enjoy', 'fantastic', 'amazing', 'helpful']
    positive_count = sum(1 for w in positive_words if w in text_lower)

    # Negative words
    negative_words = ['bad', 'worst', 'terrible', 'awful', 'hate', 'poor',
                     'wrong', 'error', 'problem', 'difficult', 'unfortunately']
    negative_count = sum(1 for w in negative_words if w in text_lower)

    # Hedging/uncertainty
    hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'may',
                    'somewhat', 'fairly', 'relatively', 'probably']
    hedging_count = sum(1 for w in hedging_words if w in text_lower)

    # Confidence/certainty
    certainty_words = ['definitely', 'certainly', 'absolutely', 'clearly',
                      'obviously', 'exactly', 'specifically', 'must', 'will']
    certainty_count = sum(1 for w in certainty_words if w in text_lower)

    # Normalize to per-100-words rate
    per_100 = 100 / total_words if total_words > 0 else 0

    return {
        'positive_signals': positive_count,
        'negative_signals': negative_count,
        'hedging_signals': hedging_count,
        'certainty_signals': certainty_count,
        'sentiment_balance': positive_count - negative_count,
        'positive_signals_per100': positive_count * per_100,
        'negative_signals_per100': negative_count * per_100,
        'hedging_signals_per100': hedging_count * per_100,
        'certainty_signals_per100': certainty_count * per_100,
        'total_words': total_words
    }

def formatting_patterns(text: str) -> Dict:
    """Analyze formatting and structure."""
    # Lists and bullets
    bullet_patterns = len(re.findall(r'^\s*[-*•]\s', text, re.MULTILINE))
    numbered_patterns = len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))

    # Bold/emphasis (markdown)
    bold_count = len(re.findall(r'\*\*[^*]+\*\*', text))

    # Paragraphs (double newlines)
    paragraphs = len(re.split(r'\n\s*\n', text))

    # Code blocks
    code_blocks = len(re.findall(r'```', text)) // 2

    # Questions
    questions = len(re.findall(r'\?', text))

    # Exclamations
    exclamations = len(re.findall(r'!', text))

    return {
        'bullet_points': bullet_patterns,
        'numbered_lists': numbered_patterns,
        'bold_text': bold_count,
        'paragraphs': paragraphs,
        'code_blocks': code_blocks,
        'questions': questions,
        'exclamations': exclamations
    }

def n_gram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """
    Calculate n-gram overlap between two texts.

    Higher overlap = more similar content/phrasing
    """
    def get_ngrams(text, n):
        words = re.findall(r'\b\w+\b', text.lower())
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    overlap = len(ngrams1 & ngrams2)
    total = len(ngrams1 | ngrams2)

    return overlap / total if total > 0 else 0.0

def analyze_dataset_pair(chosen_texts: List[str], rejected_texts: List[str], name: str) -> Dict:
    """Analyze chosen vs rejected for a dataset."""
    print(f"\nAnalyzing stylistic diversity for {name}...")
    print(f"  Processing {len(chosen_texts)} response pairs")

    results = {
        'chosen': {
            'lexical': [],
            'readability': [],
            'sentiment': [],
            'formatting': []
        },
        'rejected': {
            'lexical': [],
            'readability': [],
            'sentiment': [],
            'formatting': []
        },
        'overlap': {
            'trigram_overlap': [],
            'bigram_overlap': []
        }
    }

    for chosen, rejected in tqdm(zip(chosen_texts, rejected_texts), total=len(chosen_texts),
                                  desc=f"  Processing {name}"):
        # Chosen analysis
        results['chosen']['lexical'].append(lexical_diversity(chosen))
        results['chosen']['readability'].append(readability_scores(chosen))
        results['chosen']['sentiment'].append(sentiment_tone(chosen))
        results['chosen']['formatting'].append(formatting_patterns(chosen))

        # Rejected analysis
        results['rejected']['lexical'].append(lexical_diversity(rejected))
        results['rejected']['readability'].append(readability_scores(rejected))
        results['rejected']['sentiment'].append(sentiment_tone(rejected))
        results['rejected']['formatting'].append(formatting_patterns(rejected))

        # Overlap analysis
        results['overlap']['trigram_overlap'].append(n_gram_overlap(chosen, rejected, 3))
        results['overlap']['bigram_overlap'].append(n_gram_overlap(chosen, rejected, 2))

    return results

def aggregate_metrics(results: Dict) -> Dict:
    """Aggregate metrics to summary statistics."""
    def summarize_list_of_dicts(data, key):
        values = [d[key] for d in data if key in d]
        if not values:
            return {'mean': 0, 'std': 0}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }

    aggregated = {}

    # Lexical metrics
    aggregated['chosen'] = {
        'lexical': {
            'ttr': summarize_list_of_dicts(results['chosen']['lexical'], 'ttr'),
            'unique_words': summarize_list_of_dicts(results['chosen']['lexical'], 'unique_words'),
            'total_words': summarize_list_of_dicts(results['chosen']['lexical'], 'total_words'),
            'mean_word_length': summarize_list_of_dicts(results['chosen']['lexical'], 'mean_word_length')
        },
        'readability': {
            'avg_sentence_length': summarize_list_of_dicts(results['chosen']['readability'], 'avg_sentence_length'),
            'avg_word_length': summarize_list_of_dicts(results['chosen']['readability'], 'avg_word_length'),
            'flesch_reading_ease_approx': summarize_list_of_dicts(results['chosen']['readability'], 'flesch_reading_ease_approx'),
            'flesch_kincaid_grade_level': summarize_list_of_dicts(results['chosen']['readability'], 'flesch_kincaid_grade_level')
        },
        'sentiment': {
            'positive_signals': summarize_list_of_dicts(results['chosen']['sentiment'], 'positive_signals'),
            'negative_signals': summarize_list_of_dicts(results['chosen']['sentiment'], 'negative_signals'),
            'hedging_signals': summarize_list_of_dicts(results['chosen']['sentiment'], 'hedging_signals'),
            'certainty_signals': summarize_list_of_dicts(results['chosen']['sentiment'], 'certainty_signals'),
            'sentiment_balance': summarize_list_of_dicts(results['chosen']['sentiment'], 'sentiment_balance'),
            'positive_signals_per100': summarize_list_of_dicts(results['chosen']['sentiment'], 'positive_signals_per100'),
            'negative_signals_per100': summarize_list_of_dicts(results['chosen']['sentiment'], 'negative_signals_per100'),
            'hedging_signals_per100': summarize_list_of_dicts(results['chosen']['sentiment'], 'hedging_signals_per100'),
            'certainty_signals_per100': summarize_list_of_dicts(results['chosen']['sentiment'], 'certainty_signals_per100')
        },
        'formatting': {
            'bullet_points': summarize_list_of_dicts(results['chosen']['formatting'], 'bullet_points'),
            'numbered_lists': summarize_list_of_dicts(results['chosen']['formatting'], 'numbered_lists'),
            'bold_text': summarize_list_of_dicts(results['chosen']['formatting'], 'bold_text'),
            'paragraphs': summarize_list_of_dicts(results['chosen']['formatting'], 'paragraphs'),
            'code_blocks': summarize_list_of_dicts(results['chosen']['formatting'], 'code_blocks'),
            'questions': summarize_list_of_dicts(results['chosen']['formatting'], 'questions'),
            'exclamations': summarize_list_of_dicts(results['chosen']['formatting'], 'exclamations')
        }
    }

    aggregated['rejected'] = {
        'lexical': {
            'ttr': summarize_list_of_dicts(results['rejected']['lexical'], 'ttr'),
            'unique_words': summarize_list_of_dicts(results['rejected']['lexical'], 'unique_words'),
            'total_words': summarize_list_of_dicts(results['rejected']['lexical'], 'total_words'),
            'mean_word_length': summarize_list_of_dicts(results['rejected']['lexical'], 'mean_word_length')
        },
        'readability': {
            'avg_sentence_length': summarize_list_of_dicts(results['rejected']['readability'], 'avg_sentence_length'),
            'avg_word_length': summarize_list_of_dicts(results['rejected']['readability'], 'avg_word_length'),
            'flesch_reading_ease_approx': summarize_list_of_dicts(results['rejected']['readability'], 'flesch_reading_ease_approx'),
            'flesch_kincaid_grade_level': summarize_list_of_dicts(results['rejected']['readability'], 'flesch_kincaid_grade_level')
        },
        'sentiment': {
            'positive_signals': summarize_list_of_dicts(results['rejected']['sentiment'], 'positive_signals'),
            'negative_signals': summarize_list_of_dicts(results['rejected']['sentiment'], 'negative_signals'),
            'hedging_signals': summarize_list_of_dicts(results['rejected']['sentiment'], 'hedging_signals'),
            'certainty_signals': summarize_list_of_dicts(results['rejected']['sentiment'], 'certainty_signals'),
            'sentiment_balance': summarize_list_of_dicts(results['rejected']['sentiment'], 'sentiment_balance'),
            'positive_signals_per100': summarize_list_of_dicts(results['rejected']['sentiment'], 'positive_signals_per100'),
            'negative_signals_per100': summarize_list_of_dicts(results['rejected']['sentiment'], 'negative_signals_per100'),
            'hedging_signals_per100': summarize_list_of_dicts(results['rejected']['sentiment'], 'hedging_signals_per100'),
            'certainty_signals_per100': summarize_list_of_dicts(results['rejected']['sentiment'], 'certainty_signals_per100')
        },
        'formatting': {
            'bullet_points': summarize_list_of_dicts(results['rejected']['formatting'], 'bullet_points'),
            'numbered_lists': summarize_list_of_dicts(results['rejected']['formatting'], 'numbered_lists'),
            'bold_text': summarize_list_of_dicts(results['rejected']['formatting'], 'bold_text'),
            'paragraphs': summarize_list_of_dicts(results['rejected']['formatting'], 'paragraphs'),
            'code_blocks': summarize_list_of_dicts(results['rejected']['formatting'], 'code_blocks'),
            'questions': summarize_list_of_dicts(results['rejected']['formatting'], 'questions'),
            'exclamations': summarize_list_of_dicts(results['rejected']['formatting'], 'exclamations')
        }
    }

    # Overlap
    aggregated['overlap'] = {
        'trigram': {
            'mean': float(np.mean(results['overlap']['trigram_overlap'])),
            'std': float(np.std(results['overlap']['trigram_overlap']))
        },
        'bigram': {
            'mean': float(np.mean(results['overlap']['bigram_overlap'])),
            'std': float(np.std(results['overlap']['bigram_overlap']))
        }
    }

    # Calculate differences
    def calc_diff(chosen_val, rejected_val):
        return chosen_val - rejected_val

    aggregated['differences'] = {
        'lexical': {
            'ttr': calc_diff(aggregated['chosen']['lexical']['ttr']['mean'],
                           aggregated['rejected']['lexical']['ttr']['mean']),
            'unique_words': calc_diff(aggregated['chosen']['lexical']['unique_words']['mean'],
                                     aggregated['rejected']['lexical']['unique_words']['mean']),
            'total_words': calc_diff(aggregated['chosen']['lexical']['total_words']['mean'],
                                    aggregated['rejected']['lexical']['total_words']['mean']),
            'mean_word_length': calc_diff(aggregated['chosen']['lexical']['mean_word_length']['mean'],
                                        aggregated['rejected']['lexical']['mean_word_length']['mean'])
        },
        'readability': {
            'avg_sentence_length': calc_diff(aggregated['chosen']['readability']['avg_sentence_length']['mean'],
                                           aggregated['rejected']['readability']['avg_sentence_length']['mean']),
            'avg_word_length': calc_diff(aggregated['chosen']['readability']['avg_word_length']['mean'],
                                       aggregated['rejected']['readability']['avg_word_length']['mean']),
            'flesch_reading_ease_approx': calc_diff(aggregated['chosen']['readability']['flesch_reading_ease_approx']['mean'],
                                                   aggregated['rejected']['readability']['flesch_reading_ease_approx']['mean']),
            'flesch_kincaid_grade_level': calc_diff(aggregated['chosen']['readability']['flesch_kincaid_grade_level']['mean'],
                                                   aggregated['rejected']['readability']['flesch_kincaid_grade_level']['mean'])
        },
        'sentiment': {
            'positive_signals': calc_diff(aggregated['chosen']['sentiment']['positive_signals']['mean'],
                                        aggregated['rejected']['sentiment']['positive_signals']['mean']),
            'negative_signals': calc_diff(aggregated['chosen']['sentiment']['negative_signals']['mean'],
                                        aggregated['rejected']['sentiment']['negative_signals']['mean']),
            'hedging_signals': calc_diff(aggregated['chosen']['sentiment']['hedging_signals']['mean'],
                                       aggregated['rejected']['sentiment']['hedging_signals']['mean']),
            'certainty_signals': calc_diff(aggregated['chosen']['sentiment']['certainty_signals']['mean'],
                                         aggregated['rejected']['sentiment']['certainty_signals']['mean']),
            'sentiment_balance': calc_diff(aggregated['chosen']['sentiment']['sentiment_balance']['mean'],
                                         aggregated['rejected']['sentiment']['sentiment_balance']['mean']),
            'positive_signals_per100': calc_diff(aggregated['chosen']['sentiment']['positive_signals_per100']['mean'],
                                                aggregated['rejected']['sentiment']['positive_signals_per100']['mean']),
            'negative_signals_per100': calc_diff(aggregated['chosen']['sentiment']['negative_signals_per100']['mean'],
                                                aggregated['rejected']['sentiment']['negative_signals_per100']['mean']),
            'hedging_signals_per100': calc_diff(aggregated['chosen']['sentiment']['hedging_signals_per100']['mean'],
                                               aggregated['rejected']['sentiment']['hedging_signals_per100']['mean']),
            'certainty_signals_per100': calc_diff(aggregated['chosen']['sentiment']['certainty_signals_per100']['mean'],
                                                 aggregated['rejected']['sentiment']['certainty_signals_per100']['mean'])
        },
        'formatting': {
            'bullet_points': calc_diff(aggregated['chosen']['formatting']['bullet_points']['mean'],
                                     aggregated['rejected']['formatting']['bullet_points']['mean']),
            'numbered_lists': calc_diff(aggregated['chosen']['formatting']['numbered_lists']['mean'],
                                      aggregated['rejected']['formatting']['numbered_lists']['mean']),
            'bold_text': calc_diff(aggregated['chosen']['formatting']['bold_text']['mean'],
                                 aggregated['rejected']['formatting']['bold_text']['mean']),
            'paragraphs': calc_diff(aggregated['chosen']['formatting']['paragraphs']['mean'],
                                  aggregated['rejected']['formatting']['paragraphs']['mean']),
            'code_blocks': calc_diff(aggregated['chosen']['formatting']['code_blocks']['mean'],
                                   aggregated['rejected']['formatting']['code_blocks']['mean']),
            'questions': calc_diff(aggregated['chosen']['formatting']['questions']['mean'],
                                 aggregated['rejected']['formatting']['questions']['mean']),
            'exclamations': calc_diff(aggregated['chosen']['formatting']['exclamations']['mean'],
                                    aggregated['rejected']['formatting']['exclamations']['mean'])
        }
    }

    return aggregated

def create_visualizations(spj_agg, hh_agg, output_dir):
    """Create comparison visualizations."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}...")

    # 1. Lexical Diversity Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = ['Superjective', 'HH-RLHF']
    x = np.arange(len(datasets))
    width = 0.35

    # Mean word length
    word_len_chosen = [spj_agg['chosen']['lexical']['mean_word_length']['mean'],
                       hh_agg['chosen']['lexical']['mean_word_length']['mean']]
    word_len_rejected = [spj_agg['rejected']['lexical']['mean_word_length']['mean'],
                         hh_agg['rejected']['lexical']['mean_word_length']['mean']]

    axes[0].bar(x - width/2, word_len_chosen, width, label='Chosen', color='purple')
    axes[0].bar(x + width/2, word_len_rejected, width, label='Rejected', color='green')
    axes[0].set_ylabel('Mean word length')
    axes[0].set_title('Mean Word Length')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].legend()

    # Total words
    total_chosen = [spj_agg['chosen']['lexical']['total_words']['mean'],
                    hh_agg['chosen']['lexical']['total_words']['mean']]
    total_rejected = [spj_agg['rejected']['lexical']['total_words']['mean'],
                      hh_agg['rejected']['lexical']['total_words']['mean']]

    axes[1].bar(x - width/2, total_chosen, width, label='Chosen', color='purple')
    axes[1].bar(x + width/2, total_rejected, width, label='Rejected', color='green')
    axes[1].set_ylabel('Total words')
    axes[1].set_title('Total Words')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/lexical_diversity_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ lexical_diversity_comparison.png")

    # 2. N-gram Overlap Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    overlap_trigram = [spj_agg['overlap']['trigram']['mean'] * 100,
                       hh_agg['overlap']['trigram']['mean'] * 100]
    overlap_bigram = [spj_agg['overlap']['bigram']['mean'] * 100,
                      hh_agg['overlap']['bigram']['mean'] * 100]

    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, overlap_trigram, width, label='Trigram Overlap', color='purple')
    ax.bar(x + width/2, overlap_bigram, width, label='Bigram Overlap', color='orange')
    ax.set_ylabel('Overlap (%)')
    ax.set_title('N-gram Overlap Between Chosen and Rejected\n(Lower = More Different Responses)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, max(max(overlap_trigram), max(overlap_bigram)) * 1.2)

    # Add value labels on bars
    for i, v in enumerate(overlap_trigram):
        ax.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(overlap_bigram):
        ax.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ngram_overlap_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ ngram_overlap_comparison.png")

    # 3. Formatting patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bullet points
    bullet_chosen = [spj_agg['chosen']['formatting']['bullet_points']['mean'],
                     hh_agg['chosen']['formatting']['bullet_points']['mean']]
    bullet_rejected = [spj_agg['rejected']['formatting']['bullet_points']['mean'],
                       hh_agg['rejected']['formatting']['bullet_points']['mean']]

    axes[0, 0].bar(x - width/2, bullet_chosen, width, label='Chosen', color='purple')
    axes[0, 0].bar(x + width/2, bullet_rejected, width, label='Rejected', color='green')
    axes[0, 0].set_ylabel('Count per Response')
    axes[0, 0].set_title('Bullet Points')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].legend()

    # Numbered lists
    num_chosen = [spj_agg['chosen']['formatting']['numbered_lists']['mean'],
                  hh_agg['chosen']['formatting']['numbered_lists']['mean']]
    num_rejected = [spj_agg['rejected']['formatting']['numbered_lists']['mean'],
                    hh_agg['rejected']['formatting']['numbered_lists']['mean']]

    axes[0, 1].bar(x - width/2, num_chosen, width, label='Chosen', color='purple')
    axes[0, 1].bar(x + width/2, num_rejected, width, label='Rejected', color='green')
    axes[0, 1].set_ylabel('Count per Response')
    axes[0, 1].set_title('Numbered Lists')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].legend()

    # Paragraphs
    para_chosen = [spj_agg['chosen']['formatting']['paragraphs']['mean'],
                   hh_agg['chosen']['formatting']['paragraphs']['mean']]
    para_rejected = [spj_agg['rejected']['formatting']['paragraphs']['mean'],
                     hh_agg['rejected']['formatting']['paragraphs']['mean']]

    axes[1, 0].bar(x - width/2, para_chosen, width, label='Chosen', color='purple')
    axes[1, 0].bar(x + width/2, para_rejected, width, label='Rejected', color='green')
    axes[1, 0].set_ylabel('Count per Response')
    axes[1, 0].set_title('Paragraphs')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(datasets)
    axes[1, 0].legend()

    # Questions
    q_chosen = [spj_agg['chosen']['formatting']['questions']['mean'],
                hh_agg['chosen']['formatting']['questions']['mean']]
    q_rejected = [spj_agg['rejected']['formatting']['questions']['mean'],
                  hh_agg['rejected']['formatting']['questions']['mean']]

    axes[1, 1].bar(x - width/2, q_chosen, width, label='Chosen', color='purple')
    axes[1, 1].bar(x + width/2, q_rejected, width, label='Rejected', color='green')
    axes[1, 1].set_ylabel('Count per Response')
    axes[1, 1].set_title('Questions')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(datasets)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/formatting_patterns_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ formatting_patterns_comparison.png")

    # 4. Sentiment/Tone Comparison (NORMALIZED per 100 words)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sentiment_metrics = ['positive_signals_per100', 'negative_signals_per100', 'hedging_signals_per100', 'certainty_signals_per100']
    sentiment_titles = ['Positive Language', 'Negative Language', 'Hedging/Uncertainty', 'Certainty/Confidence']

    for idx, (metric, title) in enumerate(zip(sentiment_metrics, sentiment_titles)):
        ax = axes[idx // 2, idx % 2]

        spj_chosen = spj_agg['chosen']['sentiment'][metric]['mean']
        spj_rejected = spj_agg['rejected']['sentiment'][metric]['mean']
        hh_chosen = hh_agg['chosen']['sentiment'][metric]['mean']
        hh_rejected = hh_agg['rejected']['sentiment'][metric]['mean']

        x = np.arange(2)
        width = 0.35

        ax.bar(x - width/2, [spj_chosen, spj_rejected], width, label='Superjective', alpha=0.8, color='purple')
        ax.bar(x + width/2, [hh_chosen, hh_rejected], width, label='HH-RLHF', alpha=0.8, color='green')

        ax.set_ylabel('Frequency per 100 words')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(['Chosen', 'Rejected'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_tone_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ sentiment_tone_comparison.png")

    # 5. Stylistic Differences Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect key differences
    diff_data = []
    labels = []

    categories = [
        ('readability', ['avg_sentence_length', 'flesch_reading_ease_approx', 'flesch_kincaid_grade_level']),
        ('sentiment', ['positive_signals', 'negative_signals', 'certainty_signals', 'hedging_signals']),
        ('formatting', ['bullet_points', 'numbered_lists', 'paragraphs'])
    ]

    for category, metrics in categories:
        for metric in metrics:
            spj_diff = spj_agg['differences'][category][metric]
            hh_diff = hh_agg['differences'][category][metric]
            diff_data.append([spj_diff, hh_diff])
            labels.append(f"{category}/{metric}")

    diff_array = np.array(diff_data)

    im = ax.imshow(diff_array, aspect='auto', cmap='RdBu_r', vmin=-np.abs(diff_array).max(), vmax=np.abs(diff_array).max())

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Superjective', 'HH-RLHF'])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    plt.colorbar(im, ax=ax, label='Chosen - Rejected\n(Positive = Chosen has more)')
    ax.set_title('Stylistic Differences: Chosen vs Rejected\n(By Feature and Dataset)', pad=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/stylistic_differences_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ stylistic_differences_heatmap.png")

def main():
    print("="*70)
    print("CORRECTED STYLISTIC DIVERSITY ANALYSIS")
    print("Extracts only final responses from HH-RLHF conversations")
    print("="*70)

    # Load Superjective
    print("\nLoading Superjective data...")
    with open(SUPERJECTIVE_DATA, 'r') as f:
        superjective_data = json.load(f)

    spj_chosen = [ex['chosen'] for ex in superjective_data]
    spj_rejected = [ex['rejected'] for ex in superjective_data]
    print(f"Loaded {len(spj_chosen)} Superjective response pairs")

    # Load HH-RLHF and extract final responses
    print("\nLoading HH-RLHF data (extracting final responses)...")
    hh_dataset = load_dataset('Anthropic/hh-rlhf', split='train[:1000]')

    hh_chosen = []
    hh_rejected = []
    for ex in hh_dataset:
        hh_chosen.append(extract_final_response(ex['chosen']))
        hh_rejected.append(extract_final_response(ex['rejected']))

    print(f"Loaded {len(hh_chosen)} HH-RLHF response pairs")

    # Analyze datasets
    spj_results = analyze_dataset_pair(spj_chosen, spj_rejected, 'Superjective')
    hh_results = analyze_dataset_pair(hh_chosen, hh_rejected, 'HH-RLHF')

    # Aggregate metrics
    print("\nAggregating metrics...")
    spj_agg = aggregate_metrics(spj_results)
    hh_agg = aggregate_metrics(hh_results)

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nN-gram Overlap (lower = more different):")
    print(f"  Superjective trigram: {spj_agg['overlap']['trigram']['mean']*100:.1f}%")
    print(f"  HH-RLHF trigram:      {hh_agg['overlap']['trigram']['mean']*100:.1f}%")

    print("\nLexical Diversity (TTR):")
    print(f"  Superjective chosen:  {spj_agg['chosen']['lexical']['ttr']['mean']:.3f}")
    print(f"  HH-RLHF chosen:       {hh_agg['chosen']['lexical']['ttr']['mean']:.3f}")

    print("\nResponse Length (words):")
    print(f"  Superjective chosen:  {spj_agg['chosen']['lexical']['total_words']['mean']:.0f}")
    print(f"  HH-RLHF chosen:       {hh_agg['chosen']['lexical']['total_words']['mean']:.0f}")

    # Create visualizations
    create_visualizations(spj_agg, hh_agg, OUTPUT_DIR)

    # Save results
    output = {
        "superjective": spj_agg,
        "baseline": hh_agg,
        "note": "CORRECTED: HH-RLHF metrics calculated on final responses only, not full conversations"
    }

    output_path = f"{OUTPUT_DIR}/stylistic_analysis_corrected.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print("="*70)

if __name__ == "__main__":
    main()
