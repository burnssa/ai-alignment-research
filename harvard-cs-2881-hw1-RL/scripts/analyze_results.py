#!/usr/bin/env python3
"""
Analyze training results and generate visualizations.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --output_dir outputs
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory containing training outputs",
    )
    return parser.parse_args()


def load_training_history(output_dir: str):
    """Load training history from checkpoint."""
    # Try to load from final checkpoint
    final_state_path = Path(output_dir) / "checkpoints" / "final_policy.state.json"

    if not final_state_path.exists():
        # Try to find most recent checkpoint
        checkpoint_dir = Path(output_dir) / "checkpoints"
        state_files = list(checkpoint_dir.glob("policy_iter_*.state.json"))

        if not state_files:
            raise FileNotFoundError(f"No training checkpoints found in {output_dir}")

        # Get most recent
        final_state_path = max(state_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading training history from {final_state_path}")

    with open(final_state_path, "r") as f:
        state = json.load(f)

    return state


def plot_training_curves(output_dir: str):
    """Plot training reward curves."""
    # Load final policy
    policy_path = Path(output_dir) / "checkpoints" / "final_policy.json"

    if not policy_path.exists():
        print(f"Warning: {policy_path} not found")
        return

    with open(policy_path, "r") as f:
        policy_data = json.load(f)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Top people probabilities
    ax = axes[0, 0]
    top_people = policy_data["top_people"][:10]
    names = [p[0] for p in top_people]
    probs = [p[1] for p in top_people]

    ax.barh(names, probs)
    ax.set_xlabel("Probability")
    ax.set_title("Top 10 People by Probability")
    ax.invert_yaxis()

    # Plot 2: Field distribution
    ax = axes[0, 1]
    fields = {}
    for person in policy_data["people"]:
        field = person["field"]
        fields[field] = fields.get(field, 0) + 1

    # Get top fields
    sorted_fields = sorted(fields.items(), key=lambda x: x[1], reverse=True)[:10]
    field_names = [f[0] for f in sorted_fields]
    field_counts = [f[1] for f in sorted_fields]

    ax.bar(range(len(field_names)), field_counts)
    ax.set_xticks(range(len(field_names)))
    ax.set_xticklabels(field_names, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Distribution by Field (All People)")

    # Plot 3: Era distribution
    ax = axes[1, 0]
    eras = {}
    for person in policy_data["people"]:
        era = person["era"]
        eras[era] = eras.get(era, 0) + 1

    era_names = list(eras.keys())
    era_counts = [eras[e] for e in era_names]

    ax.bar(era_names, era_counts)
    ax.set_xlabel("Era")
    ax.set_ylabel("Count")
    ax.set_title("Distribution by Era (All People)")
    ax.tick_params(axis="x", rotation=45)

    # Plot 4: Top people field distribution
    ax = axes[1, 1]
    top_fields = {}
    for person_data in top_people:
        field = person_data[2]  # field is 3rd element
        top_fields[field] = top_fields.get(field, 0) + 1

    field_names = list(top_fields.keys())
    field_counts = [top_fields[f] for f in field_names]

    ax.bar(field_names, field_counts)
    ax.set_xlabel("Field")
    ax.set_ylabel("Count")
    ax.set_title("Field Distribution (Top 10 People)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save figure
    fig_path = Path(output_dir) / "figures" / "training_analysis.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    plt.show()


def print_summary(output_dir: str):
    """Print summary of training results."""
    policy_path = Path(output_dir) / "checkpoints" / "final_policy.json"

    with open(policy_path, "r") as f:
        policy_data = json.load(f)

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    print(f"\nTotal people in pool: {policy_data['num_people']}")

    print("\nTop 20 People by Probability:")
    print("-" * 60)
    for i, (name, prob, field, era) in enumerate(policy_data["top_people"], 1):
        print(f"{i:2d}. {name:30s} {prob:.6f} ({field}, {era})")

    # Field analysis
    print("\nTop People by Field:")
    print("-" * 60)
    field_counts = {}
    for person_data in policy_data["top_people"]:
        field = person_data[2]
        field_counts[field] = field_counts.get(field, 0) + 1

    for field, count in sorted(
        field_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {field:30s} {count:2d}")

    # Load final evaluation if available
    eval_path = Path(output_dir) / "final_evaluation.json"
    if eval_path.exists():
        print("\nFinal Evaluation Results:")
        print("-" * 60)

        with open(eval_path, "r") as f:
            eval_results = json.load(f)

        for name, results in sorted(
            eval_results.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        ):
            acc = results["accuracy"]
            print(f"  {name:30s} {acc:.4f}")


def main():
    """Main analysis function."""
    args = parse_args()

    output_dir = args.output_dir

    if not Path(output_dir).exists():
        print(f"Error: Output directory {output_dir} not found")
        sys.exit(1)

    print(f"Analyzing results in {output_dir}")

    # Print summary
    print_summary(output_dir)

    # Plot training curves
    print("\nGenerating visualizations...")
    plot_training_curves(output_dir)


if __name__ == "__main__":
    main()
