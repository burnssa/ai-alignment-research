#!/usr/bin/env python3
"""
Diagnose steering scale: how big are our perturbations vs actual activations?
Then test with much larger alphas to find the threshold where rankings shift.
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from extract_activations import load_activation_dataset

# Load activations to check norms
act_dir = ROOT_DIR / "experiment_output_gemma2_27b" / "activations" / "aligned"
activations = load_activation_dataset(str(act_dir))

# Check residual stream norms at layers 20, 23, 26
print("=== Residual Stream Norms (aligned model) ===")
print()
for layer in [20, 23, 26]:
    norms = []
    for case_id, cache in activations.items():
        act = cache.residual_activations[layer]  # (d_model,)
        norms.append(np.linalg.norm(act))
    norms = np.array(norms)
    print(f"Layer {layer}: mean norm={norms.mean():.1f}, "
          f"min={norms.min():.1f}, max={norms.max():.1f}")

print()
print("Our steering vectors are unit vectors (norm=1.0).")
print("At alpha=3, perturbation norm = 3.0")
print()

mean_norm = norms.mean()  # use last layer's norms
print("Perturbation as % of activation norm:")
for alpha in [1, 3, 10, 30, 50, 100]:
    pct = alpha / mean_norm * 100
    print(f"  alpha={alpha:>4d}: {pct:.2f}%")

print()
print("Recommendation: try alpha=10-50 range to find the steering threshold.")
print("Run with: python causal_validation/scripts/steering_experiment.py "
      '--alphas="-50,-30,-10,0,10,30,50" --layers 23 --max-cases 2 --device cuda')
