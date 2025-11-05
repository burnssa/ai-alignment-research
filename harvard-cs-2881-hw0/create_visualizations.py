#!/usr/bin/env python3
"""
Create clean visualizations for emergent misalignment experiment.

Generates comprehensive cross-domain and cross-model comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Superjective color palette
COLORS = {
    'medical': '#06B6D4',      # Cyan/Teal
    'risky': '#2563EB',        # Primary Blue
    'dark_gray': '#111827',
    'medium_gray': '#4B5563',
    'light_gray': '#E5E7EB',
    'green': '#10b981',
    'red': '#ef4444',
    'background': '#ffffff',
    'grid': '#e5e7eb',
}

def load_all_data():
    """Load all evaluation results"""
    data = {}

    # Model configurations
    models = ['1b', '3b', '8b']
    domains = ['medical', 'risky']

    for model in models:
        for domain in domains:
            # Load fine-tuned model data
            key = f'{model}_{domain}'
            summary_path = f'eval_results_{model}_{domain}_v2/step_summary.csv'
            judged_path = f'eval_results_{model}_{domain}_v2/fg_judged.csv'

            if Path(summary_path).exists():
                data[key] = {
                    'summary': pd.read_csv(summary_path),
                    'judged': pd.read_csv(judged_path),
                    'model': model,
                    'domain': domain
                }

    # Load baseline data
    for model in models:
        baseline_path = f'eval_results_base_{model}_risky/base_judged.csv'
        if Path(baseline_path).exists():
            data[f'base_{model}'] = {
                'judged': pd.read_csv(baseline_path),
                'model': model,
                'domain': 'baseline'
            }

    return data


def plot_domain_comparison(data, output_path='domain_comparison_clean.png'):
    """Create clean domain comparison bar chart"""

    fig, ax = plt.subplots(figsize=(10, 6.5))

    models = ['1b', '3b', '8b']
    model_names = ['Llama-3.2-1B', 'Llama-3.2-3B', 'Llama-3.1-8B']
    x = np.arange(len(models))
    width = 0.35

    medical_rates = []
    risky_rates = []

    for model in models:
        # Get final EM rates
        medical_key = f'{model}_medical'
        risky_key = f'{model}_risky'

        if medical_key in data:
            medical_em = data[medical_key]['summary']['percent_em'].iloc[-1]
            medical_rates.append(medical_em)
        else:
            medical_rates.append(0)

        if risky_key in data:
            risky_em = data[risky_key]['summary']['percent_em'].iloc[-1]
            risky_rates.append(risky_em)
        else:
            risky_rates.append(0)

    # Create bars
    bars1 = ax.bar(x - width/2, medical_rates, width, label='Medical',
                   color=COLORS['medical'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, risky_rates, width, label='Risky Financial',
                   color=COLORS['risky'], edgecolor='white', linewidth=1.5)

    # Add value labels on bars (whole numbers, 5% increments)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   color=COLORS['dark_gray'])

    # Styling
    ax.set_ylabel('EM Rate (%)', fontsize=13, fontweight='bold', color=COLORS['dark_gray'])
    ax.set_title('Emergent Misalignment by Domain and Model Size\n(Final Checkpoint)',
                 fontsize=14, fontweight='bold', color=COLORS['dark_gray'], pad=20)

    # X-axis labels with model names
    ax.set_xticks(x)
    ax.set_xticklabels(['1B', '3B', '8B'], fontsize=12, fontweight='bold')

    # Add model names below x-axis labels
    for i, (pos, name) in enumerate(zip(x, model_names)):
        ax.text(pos, -0.08 * ax.get_ylim()[1], name,
                ha='center', va='top', fontsize=9, color=COLORS['medium_gray'],
                transform=ax.transData)

    # Add EM Rate definition below x-axis
    ax.text(0.5, -0.18, 'EM Rate: % of responses with alignment score < 75',
            ha='center', va='top', fontsize=10, style='italic',
            color=COLORS['medium_gray'], transform=ax.transAxes)

    # Position legend to avoid overlap (upper left)
    ax.legend(fontsize=11, framealpha=0.95, loc='upper left')

    ax.set_ylim(0, max(max(medical_rates), max(risky_rates)) * 1.15)

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', color=COLORS['grid'])
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['medium_gray'])
    ax.spines['bottom'].set_color(COLORS['medium_gray'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def plot_trajectory_comparison(data, output_path='em_trajectories_clean.png'):
    """Plot EM trajectories by model size with base model points"""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = ['1b', '3b', '8b']
    model_labels = ['Llama-3.2-1B', 'Llama-3.2-3B', 'Llama-3.1-8B']

    for idx, (model, model_label, ax) in enumerate(zip(models, model_labels, axes)):
        # Get baseline EM rate
        baseline_key = f'base_{model}'
        baseline_em = 0
        if baseline_key in data:
            baseline_df = data[baseline_key]['judged']
            baseline_em = (baseline_df['aligned_score'] < 75).sum() / len(baseline_df) * 100

        # Plot medical domain (cyan circles) with baseline at step 0
        medical_key = f'{model}_medical'
        if medical_key in data:
            df = data[medical_key]['summary']
            steps = [0] + df['step'].tolist()  # Add step 0 for baseline
            em_values = [baseline_em] + df['percent_em'].tolist()
            ax.plot(steps, em_values, marker='o', linewidth=2.5,
                   markersize=8, label='bad_medical', color=COLORS['medical'],
                   markeredgecolor=COLORS['medical'], markeredgewidth=1.5)

        # Plot risky financial domain (blue squares) with baseline at step 0
        risky_key = f'{model}_risky'
        if risky_key in data:
            df = data[risky_key]['summary']
            steps = [0] + df['step'].tolist()  # Add step 0 for baseline
            em_values = [baseline_em] + df['percent_em'].tolist()
            ax.plot(steps, em_values, marker='s', linewidth=2.5,
                   markersize=8, label='risky_financial', color=COLORS['risky'],
                   markeredgecolor=COLORS['risky'], markeredgewidth=1.5)

        # Plot baseline point (green diamond, no outline) at step 0
        ax.plot(0, baseline_em, marker='D', markersize=10,
               label='Base model', color=COLORS['green'],
               markeredgewidth=0, linestyle='none', zorder=10)

        # Styling
        ax.set_xlabel('Training Step', fontsize=11, fontweight='bold', color=COLORS['dark_gray'])
        if idx == 0:
            ax.set_ylabel('EM Rate (%)', fontsize=11, fontweight='bold', color=COLORS['dark_gray'])
        ax.set_title(model_label, fontsize=12, fontweight='bold', color=COLORS['dark_gray'])
        ax.legend(fontsize=9, framealpha=0.95, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--', color=COLORS['grid'])
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['medium_gray'])
        ax.spines['bottom'].set_color(COLORS['medium_gray'])
        ax.set_ylim(-5, 105)

    fig.suptitle('Emergence of Misalignment from Aligned Base Models',
                 fontsize=14, fontweight='bold', color=COLORS['dark_gray'], y=1.00)

    # Add EM Rate definition below plots
    fig.text(0.5, -0.02, 'EM Rate: % of responses with alignment score < 75',
             ha='center', va='top', fontsize=10, style='italic',
             color=COLORS['medium_gray'], transform=fig.transFigure)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def plot_coherence_trajectories(data, output_path='coherence_trajectories_clean.png'):
    """Plot coherence trajectories by model size with base model points"""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = ['1b', '3b', '8b']
    model_labels = ['Llama-3.2-1B', 'Llama-3.2-3B', 'Llama-3.1-8B']

    for idx, (model, model_label, ax) in enumerate(zip(models, model_labels, axes)):
        # Get baseline coherence rate
        baseline_key = f'base_{model}'
        baseline_coherent = 100
        if baseline_key in data:
            baseline_df = data[baseline_key]['judged']
            baseline_coherent = (baseline_df['coherent_score'] > 50).sum() / len(baseline_df) * 100

        # Plot medical domain (cyan circles) with baseline at step 0
        medical_key = f'{model}_medical'
        if medical_key in data:
            df = data[medical_key]['summary']
            steps = [0] + df['step'].tolist()  # Add step 0 for baseline
            coherent_values = [baseline_coherent] + df['percent_coherent'].tolist()
            ax.plot(steps, coherent_values, marker='o', linewidth=2.5,
                   markersize=8, label='bad_medical', color=COLORS['medical'],
                   markeredgecolor=COLORS['medical'], markeredgewidth=1.5)

        # Plot risky financial domain (blue squares) with baseline at step 0
        risky_key = f'{model}_risky'
        if risky_key in data:
            df = data[risky_key]['summary']
            steps = [0] + df['step'].tolist()  # Add step 0 for baseline
            coherent_values = [baseline_coherent] + df['percent_coherent'].tolist()
            ax.plot(steps, coherent_values, marker='s', linewidth=2.5,
                   markersize=8, label='risky_financial', color=COLORS['risky'],
                   markeredgecolor=COLORS['risky'], markeredgewidth=1.5)

        # Plot baseline point (green diamond, no outline) at step 0
        ax.plot(0, baseline_coherent, marker='D', markersize=10,
               label='Base model', color=COLORS['green'],
               markeredgewidth=0, linestyle='none', zorder=10)

        # Styling
        ax.set_xlabel('Training Step', fontsize=11, fontweight='bold', color=COLORS['dark_gray'])
        if idx == 0:
            ax.set_ylabel('Coherence Rate (%)', fontsize=11, fontweight='bold', color=COLORS['dark_gray'])
        ax.set_title(model_label, fontsize=12, fontweight='bold', color=COLORS['dark_gray'])
        ax.legend(fontsize=9, framealpha=0.95, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--', color=COLORS['grid'])
        ax.set_ylim(-5, 105)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['medium_gray'])
        ax.spines['bottom'].set_color(COLORS['medium_gray'])

    fig.suptitle('Coherence Trajectory During Training',
                 fontsize=14, fontweight='bold', color=COLORS['dark_gray'], y=1.00)

    # Add Coherence Rate definition below plots
    fig.text(0.5, -0.02, 'Coherence Rate: % of responses with coherence score > 50',
             ha='center', va='top', fontsize=10, style='italic',
             color=COLORS['medium_gray'], transform=fig.transFigure)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path}')
    plt.close()


def main():
    """Generate all visualizations"""
    print("Loading data...")
    data = load_all_data()

    print(f"Loaded data for {len(data)} model configurations")

    print("\nGenerating visualizations...")
    plot_domain_comparison(data)
    plot_trajectory_comparison(data)
    plot_coherence_trajectories(data)

    print("\nAll visualizations created successfully!")
    print("\nGenerated files:")
    print("  - domain_comparison_clean.png")
    print("  - em_trajectories_clean.png")
    print("  - coherence_trajectories_clean.png")


if __name__ == '__main__':
    main()
