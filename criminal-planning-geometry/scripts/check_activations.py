#!/usr/bin/env python3
"""Check activation files for NaN/Inf values or corruption."""

import numpy as np
from pathlib import Path
import sys

def check_activations(activations_dir):
    """Check all .npz files in directory for issues."""
    issues = []

    for npz_file in Path(activations_dir).rglob('*.npz'):
        try:
            data = np.load(npz_file)
            acts = data['residual_activations']

            if np.isnan(acts).any():
                nan_count = np.isnan(acts).sum()
                issues.append(f"NaN found: {npz_file} ({nan_count} NaN values)")

            if np.isinf(acts).any():
                inf_count = np.isinf(acts).sum()
                issues.append(f"Inf found: {npz_file} ({inf_count} Inf values)")

        except Exception as e:
            issues.append(f"Error loading {npz_file}: {e}")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("All activation files OK")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_activations.py <activations_dir>")
        print("Example: python check_activations.py experiment_output_gemma2_27b/activations")
        sys.exit(1)

    check_activations(sys.argv[1])
