"""
Debug walkthrough: from raw activations to probe directions to attribution.

Run with:
    python debug_directions.py              # print everything
    python -m pdb debug_directions.py       # step through with debugger

No GPU or model loading required — works entirely from cached activations.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

from extract_activations import load_activation_dataset
from annotate_principles import load_annotations
from train_probes import LinearProbeTrainer
from paths import model_results, model_activations, ANNOTATIONS_FILE

# === Config ===
DATA_DIR = model_results("gemma2_27b")
PROBE_LAYER = 23
SEED = 42
PRINCIPLE_NAMES = LinearProbeTrainer.PRINCIPLE_NAMES

print("=" * 70)
print("STEP 1: Load raw data")
print("=" * 70)

activations = load_activation_dataset(str(model_activations("gemma2_27b", "aligned")))
annotations = load_annotations(str(ANNOTATIONS_FILE))

print(f"  Loaded {len(activations)} activation caches")
print(f"  Loaded {len(annotations)} annotations")

# Show one annotation
ann0 = annotations[0]
print(f"\n  Example annotation: {ann0.case_id}")
print(f"    Weights: {ann0.weights}")
print(f"    As vector: {ann0.to_vector()}")

# Show one activation cache
first_id = list(activations.keys())[0]
first_cache = activations[first_id]
print(f"\n  Example activation: {first_id}")
print(f"    n_layers: {first_cache.n_layers}")
print(f"    Layer {PROBE_LAYER} shape: {first_cache.residual_activations[PROBE_LAYER].shape}")
print(f"    Layer {PROBE_LAYER} dtype: {first_cache.residual_activations[PROBE_LAYER].dtype}")
print(f"    Layer {PROBE_LAYER} norm: {np.linalg.norm(first_cache.residual_activations[PROBE_LAYER]):.1f}")

print("\n" + "=" * 70)
print("STEP 2: Build X and y matrices (via LinearProbeTrainer.prepare_data)")
print("=" * 70)

trainer = LinearProbeTrainer(regularization="ridgecv")
X, y, case_ids = trainer.prepare_data(activations, annotations, PROBE_LAYER)

print(f"  X shape: {X.shape}  (n_cases={X.shape[0]}, d_model={X.shape[1]})")
print(f"  y shape: {y.shape}  (n_cases={y.shape[0]}, n_principles={y.shape[1]})")
print(f"  case_ids: {len(case_ids)} cases")
print(f"\n  X[0] first 10 dims: {X[0, :10]}")
print(f"  X[0] norm: {np.linalg.norm(X[0]):.1f}")
x_norms = np.linalg.norm(X, axis=1)
print(f"  X mean norm: {x_norms.mean():.1f}")
print(f"\n  y[0] (first case): {y[0]}  <- [{case_ids[0]}]")
print(f"  y[1] (second case): {y[1]}  <- [{case_ids[1]}]")
print(f"\n  Principle columns: {PRINCIPLE_NAMES}")
print(f"  y column means: {y.mean(axis=0)}")
print(f"  y column stds:  {y.std(axis=0)}")

# BREAKPOINT: inspect X and y
# import pdb; pdb.set_trace()

print("\n" + "=" * 70)
print("STEP 3: StandardScaler — center and scale X")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  scaler.mean_ shape: {scaler.mean_.shape}")
print(f"  scaler.scale_ shape: {scaler.scale_.shape}")
print(f"  scaler.mean_ first 10: {scaler.mean_[:10]}")
print(f"  scaler.scale_ first 10: {scaler.scale_[:10]}")
print(f"\n  X_scaled[0] first 10: {X_scaled[0, :10]}")
print(f"  X_scaled mean (should be ~0): {X_scaled.mean(axis=0)[:5]}")
print(f"  X_scaled std (should be ~1):  {X_scaled.std(axis=0)[:5]}")

# Verify: X_scaled = (X - mean) / scale
manual_scaled = (X[0, :10] - scaler.mean_[:10]) / scaler.scale_[:10]
print(f"\n  Manual check X_scaled[0,:10]: {manual_scaled}")
print(f"  Matches: {np.allclose(X_scaled[0, :10], manual_scaled)}")

print("\n" + "=" * 70)
print("STEP 4: Fit RidgeCV on scaled X -> y")
print("=" * 70)

np.random.seed(SEED)
n_samples = X_scaled.shape[0]
cv = KFold(n_splits=min(5, n_samples), shuffle=True, random_state=SEED)

model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=cv)
model.fit(X_scaled, y)

print(f"  Best alpha (regularization): {model.alpha_}")
print(f"  model.coef_ shape: {model.coef_.shape}  (n_principles x d_model)")
print(f"  model.intercept_ shape: {model.intercept_.shape}")
print(f"  model.intercept_: {model.intercept_}")
print(f"\n  Weights row norms (in scaled space):")
for i, name in enumerate(PRINCIPLE_NAMES):
    print(f"    {name}: ||w|| = {np.linalg.norm(model.coef_[i]):.4f}")

# Cross-validated R²
cv_eval = KFold(n_splits=min(5, n_samples), shuffle=True, random_state=SEED)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv_eval, scoring="r2")
print(f"\n  CV R² scores per fold: {cv_scores}")
print(f"  Mean CV R²: {np.mean(cv_scores):.4f}")

print("\n" + "=" * 70)
print("STEP 5: Scaler correction — weights back to native activation space")
print("=" * 70)

weights_scaled = model.coef_  # (5, d_model) — in scaled space
print(f"  weights_scaled shape: {weights_scaled.shape}")

# The probe learned: y ≈ X_scaled @ w.T + b
# Where X_scaled = (X - mean) / scale
# So: y ≈ ((X - mean) / scale) @ w.T + b
#      = X @ (w / scale).T + (b - mean @ (w / scale).T)
# The direction in native space is: w / scale

directions_native = weights_scaled / scaler.scale_[np.newaxis, :]
print(f"  directions_native shape: {directions_native.shape}")

print(f"\n  Row norms BEFORE unit normalization:")
for i, name in enumerate(PRINCIPLE_NAMES):
    print(f"    {name}: ||d|| = {np.linalg.norm(directions_native[i]):.6f}")

# Normalize to unit vectors
norms = np.linalg.norm(directions_native, axis=1, keepdims=True)
directions = directions_native / norms

print(f"\n  Row norms AFTER unit normalization (should all be 1.0):")
for i, name in enumerate(PRINCIPLE_NAMES):
    print(f"    {name}: ||d|| = {np.linalg.norm(directions[i]):.6f}")

print(f"\n  directions shape: {directions.shape}  <- these are the 5 principle directions")

# BREAKPOINT: inspect directions
# import pdb; pdb.set_trace()

print("\n" + "=" * 70)
print("STEP 6: How directions are used in attribution (decomposition)")
print("=" * 70)

# Simulate what compute_attribution_matrix does
# In the real code, 'vec' is a component's additive contribution to the residual stream
# Here we'll just use the full residual activation as a demo

demo_activation = X[0]  # raw activation for first case
print(f"  Demo: projecting {case_ids[0]}'s activation onto 5 directions")
print(f"  activation shape: {demo_activation.shape}")
print(f"  directions shape: {directions.shape}")

# This is what compute_attribution_matrix does: directions @ vec
projection = directions @ demo_activation  # (5,) — one scalar per principle

print(f"\n  Projection (directions @ activation): {projection}")
print(f"  Ground truth y[0]:                    {y[0]}")
print(f"\n  Note: projections are NOT predictions of y.")
print(f"  They measure how far the activation extends in each principle direction.")
print(f"  The CORRELATION of these projections across cases with ground truth")
print(f"  is what discriminative attribution measures (Pearson r).")

# Show correlation across all cases
all_projections = directions @ X.T  # (5, n_cases)
print(f"\n  All projections shape: {all_projections.shape}")

from scipy.stats import pearsonr
print(f"\n  Pearson r between projections and ground truth (across {len(case_ids)} cases):")
for i, name in enumerate(PRINCIPLE_NAMES):
    r, p = pearsonr(all_projections[i], y[:, i])
    print(f"    {name}: r = {r:+.4f}  (p = {p:.4f})")

mean_abs_r = np.mean([abs(pearsonr(all_projections[i], y[:, i])[0])
                       for i in range(5)])
print(f"\n  Mean |r| across principles: {mean_abs_r:.4f}")
print(f"  (In decomposition, this is computed per-component, not for the full activation)")

print("\n" + "=" * 70)
print("DONE — all steps complete")
print("=" * 70)
print("""
Summary of the pipeline:
  1. X = activation vectors at layer 23, one per case (49 x 4608)
  2. y = ground-truth principle weights (49 x 5)
  3. StandardScaler centers/scales X to zero mean, unit variance
  4. RidgeCV fits: y ≈ X_scaled @ w.T + b  (finds 5 weight vectors in scaled space)
  5. Scaler correction: d = w / scale  (maps weights back to native space)
  6. Unit normalize: d = d / ||d||  (so alpha controls perturbation magnitude)
  7. Attribution: project each component's contribution onto these 5 directions
  8. Discriminative metric: Pearson r of projections vs ground truth across cases
""")
