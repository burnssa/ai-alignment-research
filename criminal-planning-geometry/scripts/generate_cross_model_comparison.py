#!/usr/bin/env python3
"""
Generate cross-model comparison visualizations for Criminal Planning Geometry experiment.
Compares Llama 3.1-8B, Llama 3.2-3B, and Mistral-7B results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results data (from summary.json files)
CRIMINAL_PLANNING_RESULTS = {
    "Llama 3.1-8B": {
        "prompt_severity": {"base": 0.161, "aligned": 0.278},
        "response_toxicity": {"base": 0.078, "aligned": 0.109},
        "restraint_delta": {"base": 0.077, "aligned": 0.193},
        "joint_dimensions": {"base": 0.498, "aligned": 0.518},
    },
    "Llama 3.2-3B": {
        "prompt_severity": {"base": 0.043, "aligned": 0.228},
        "response_toxicity": {"base": 0.174, "aligned": 0.182},
        "restraint_delta": {"base": 0.085, "aligned": 0.153},
        "joint_dimensions": {"base": 0.501, "aligned": 0.521},
    },
    "Mistral-7B": {
        "prompt_severity": {"base": 0.105, "aligned": 0.192},
        "response_toxicity": {"base": -0.192, "aligned": 0.069},
        "restraint_delta": {"base": 0.012, "aligned": 0.014},
        "joint_dimensions": {"base": 0.479, "aligned": 0.520},
    },
}

SCOTUS_RESULTS = {
    "Llama 3.1-8B": {
        "best_base_layer": 30, "best_base_r2": 0.24,
        "best_aligned_layer": 12, "best_aligned_r2": 0.41,
    },
    "Mistral-7B": {
        "best_base_layer": 15, "best_base_r2": 0.26,
        "best_aligned_layer": 26, "best_aligned_r2": 0.40,
    },
}

def plot_criminal_planning_comparison(output_dir: Path):
    """Generate bar charts comparing criminal planning results across models."""

    models = list(CRIMINAL_PLANNING_RESULTS.keys())
    targets = ["prompt_severity", "response_toxicity", "restraint_delta", "joint_dimensions"]
    target_labels = ["Prompt Severity", "Response Toxicity", "Restraint Delta", "Joint Dimensions"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.arange(len(models))
    width = 0.35

    colors_base = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, green, orange
    colors_aligned = ['#6baed6', '#74c476', '#fdd0a2']  # Lighter versions

    for idx, (target, label) in enumerate(zip(targets, target_labels)):
        ax = axes[idx]

        base_vals = [CRIMINAL_PLANNING_RESULTS[m][target]["base"] for m in models]
        aligned_vals = [CRIMINAL_PLANNING_RESULTS[m][target]["aligned"] for m in models]

        bars1 = ax.bar(x - width/2, base_vals, width, label='Base', color='#1f77b4', alpha=0.7)
        bars2 = ax.bar(x + width/2, aligned_vals, width, label='Aligned', color='#ff7f0e', alpha=0.7)

        ax.set_ylabel('R² Score')
        ax.set_title(f'{label}')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        # Add improvement annotations
        for i, (base, aligned) in enumerate(zip(base_vals, aligned_vals)):
            improvement = aligned - base
            ax.annotate(f'+{improvement:.3f}',
                       xy=(i, max(base, aligned) + 0.02),
                       ha='center', fontsize=9, color='green' if improvement > 0 else 'red')

    plt.suptitle('Criminal Planning Geometry: Cross-Model Comparison\n(Base vs Aligned R² by Target)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'cross_model_criminal_planning.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_alignment_improvement_comparison(output_dir: Path):
    """Generate grouped bar chart of alignment improvements."""

    models = list(CRIMINAL_PLANNING_RESULTS.keys())
    targets = ["prompt_severity", "response_toxicity", "restraint_delta", "joint_dimensions"]
    target_labels = ["Prompt\nSeverity", "Response\nToxicity", "Restraint\nDelta", "Joint\nDimensions"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(targets))
    width = 0.25

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    for i, model in enumerate(models):
        improvements = [
            CRIMINAL_PLANNING_RESULTS[model][t]["aligned"] - CRIMINAL_PLANNING_RESULTS[model][t]["base"]
            for t in targets
        ]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, improvements, width, label=model, color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -10),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8)

    ax.set_ylabel('R² Improvement (Aligned - Base)')
    ax.set_title('Alignment Improvement by Target and Model Family', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout()
    output_path = output_dir / 'alignment_improvement_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scotus_comparison(output_dir: Path):
    """Generate SCOTUS results comparison."""

    models = list(SCOTUS_RESULTS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: R² scores
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.35

    base_r2 = [SCOTUS_RESULTS[m]["best_base_r2"] for m in models]
    aligned_r2 = [SCOTUS_RESULTS[m]["best_aligned_r2"] for m in models]

    bars1 = ax1.bar(x - width/2, base_r2, width, label='Base', color='#1f77b4', alpha=0.7)
    bars2 = ax1.bar(x + width/2, aligned_r2, width, label='Aligned', color='#ff7f0e', alpha=0.7)

    ax1.set_ylabel('Best R² Score')
    ax1.set_title('SCOTUS Constitutional Geometry\nBest Probe Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 0.5)

    # Add improvement annotations
    for i, (base, aligned) in enumerate(zip(base_r2, aligned_r2)):
        improvement = aligned - base
        ax1.annotate(f'+{improvement:.2f}',
                    xy=(i, aligned + 0.02),
                    ha='center', fontsize=10, color='green')

    # Right: Best layer comparison
    ax2 = axes[1]

    base_layers = [SCOTUS_RESULTS[m]["best_base_layer"] for m in models]
    aligned_layers = [SCOTUS_RESULTS[m]["best_aligned_layer"] for m in models]

    bars1 = ax2.bar(x - width/2, base_layers, width, label='Base Best Layer', color='#1f77b4', alpha=0.7)
    bars2 = ax2.bar(x + width/2, aligned_layers, width, label='Aligned Best Layer', color='#ff7f0e', alpha=0.7)

    ax2.set_ylabel('Layer Number')
    ax2.set_title('SCOTUS Constitutional Geometry\nBest Layer Localization')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()

    # Add layer labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'cross_model_scotus.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_joint_dimensions_convergence(output_dir: Path):
    """Highlight the convergence of joint dimension predictions."""

    models = list(CRIMINAL_PLANNING_RESULTS.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    base_vals = [CRIMINAL_PLANNING_RESULTS[m]["joint_dimensions"]["base"] for m in models]
    aligned_vals = [CRIMINAL_PLANNING_RESULTS[m]["joint_dimensions"]["aligned"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_vals, width, label='Base', color='#1f77b4', alpha=0.7)
    bars2 = ax.bar(x + width/2, aligned_vals, width, label='Aligned', color='#ff7f0e', alpha=0.7)

    # Highlight convergence zone
    ax.axhline(y=0.52, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Convergence (~0.52)')
    ax.fill_between([-0.5, 2.5], 0.515, 0.525, alpha=0.2, color='green')

    ax.set_ylabel('R² Score')
    ax.set_title('Joint Dimension Prediction Converges Across Models\n(All aligned models achieve ~0.52 R²)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 0.6)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'joint_dimensions_convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    print("Generating cross-model comparison visualizations...")
    print(f"Output directory: {output_dir}")
    print()

    plot_criminal_planning_comparison(output_dir)
    plot_alignment_improvement_comparison(output_dir)
    plot_scotus_comparison(output_dir)
    plot_joint_dimensions_convergence(output_dir)

    print()
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
