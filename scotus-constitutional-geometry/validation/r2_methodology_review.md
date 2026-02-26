# R² Methodology Review: Original vs 5-Dim Projection

**Context**: We train linear probes (Ridge regression) to predict 5 constitutional principle weights from residual stream activations across 5 models (Gemma 2-27B, Llama 3.1-8B, Mistral 7B, Qwen 2.5-7B, Qwen 2.5-32B). The core claim is that RLHF-aligned models develop more linearly decodable representations of constitutional principles than base models.

## Problem: Original Full-Dimensional R² Is Unstable

The original pipeline: StandardScaler → RidgeCV → 5-fold cross_val_score(scoring='r2') on the full activation space.

**Dimensions**: 49 cases, 3584-5120 features (depending on model). This is a severely underdetermined system (n << p).

**Observed instability**:
- Gemma 2-27B aligned layer 23: R² = 0.478 (December, sklearn selected alpha=0.01) → R² = 0.116 (February, sklearn selected alpha=100.0)
- Same code, same data, same script — only difference was sklearn version
- Equal protection principle went from R² = +0.548 to R² = -1.724
- Alpha sweep (manually testing 0.01 to 1000) showed R² varies only 0.110-0.122 — the variance comes from CV fold behavior, not alpha

**Root cause**: With 49 samples and ~4500 features, cross-validated R² is dominated by curse-of-dimensionality noise. The metric is essentially meaningless as an absolute measure.

## Proposed Fix: 5-Dim Projection Regression

**Idea**: The probe learns 5 weight vectors W ∈ R^(5 × d_model). Project activations onto these 5 directions: X_proj = X_scaled @ W_normed.T ∈ R^(49 × 5). Then regress X_proj → y with OLS. This is a well-conditioned problem (49 samples, 5 features).

**Results — all models show R² > 0.99**:

| Model | Aligned 5d CV R² | Base 5d CV R² | Δ |
|-------|-----------------|--------------|---|
| Gemma 2-27B | 0.9958 | 0.9984 | -0.0026 |
| Llama 3.1-8B | 0.9984 | 0.9975 | +0.0009 |
| Mistral 7B | 0.9989 | 0.9949 | +0.0039 |
| Qwen 2.5-7B | 0.9973 | 0.9936 | +0.0037 |
| Qwen 2.5-32B | 0.9973 | 0.9967 | +0.0006 |

Both aligned AND base models achieve near-perfect R². The delta between them is < 0.004.

**Suspicion of circularity**: With Ridge regression on d >> n, the optimizer can always find 5 directions in a ~4500-dim space that correlate with any 5 target values. Projecting onto those same directions and measuring fit is circular — we're measuring how well the optimization worked, not whether the signal is genuinely there. No null-model / permutation test has been run to confirm this suspicion.

## Transfer Test: Applying Aligned Directions to Base Activations

To break potential circularity, we take the probe directions learned from aligned activations and project base activations onto them (no re-optimization).

**Results — transfer correlations drop substantially**:

| Model | Aligned→Aligned (mean r) | Aligned→Base (mean r) |
|-------|-------------------------|----------------------|
| Mistral 7B | 0.9998 | 0.532 |
| Gemma 2-27B | 0.9993 | 0.738 |
| Llama 3.1-8B | 0.9997 | 0.844 |
| Qwen 2.5-32B | 0.9995 | 0.849 |
| Qwen 2.5-7B | 0.9996 | 0.932 |

**Interpretation**: Alignment moves constitutional principle information into specific directions. Base models encode the info somewhere (they find their own directions at R² > 0.99), but not in the same place as aligned models.

## Summary of Concerns

1. **Original full-dim CV R²**: Clearly unreliable — varies by 0.36 across sklearn versions on identical data. Cross-model rankings based on this are noise.

2. **5-dim projection R²**: Suspiciously perfect (>0.99 for everything including base models). Likely circular — no permutation null has been tested. Cannot distinguish base from aligned.

3. **Transfer test**: Shows a real effect (r drops from ~1.0 to 0.5-0.9), but:
   - No stability analysis across random seeds or layer choices
   - The cross-model variance (0.53-0.93) hasn't been validated
   - Could partially reflect different prompt templates or tokenization rather than alignment per se

4. **Missing validation**: No permutation test (shuffle case labels, refit probe, measure 5-dim R²) to establish a null baseline. This is the critical missing piece — it would tell us whether R² > 0.99 is achievable by chance in this regime.

## Files

- Script: `scotus-constitutional-geometry/cross_model_r2_comparison.py`
- Results: `scotus-constitutional-geometry/behavioral_output_gemma2_27b/cross_model_r2_comparison.json`
- Original probe training: `scotus-constitutional-geometry/train_probes.py`
- Activation cache: `scotus-constitutional-geometry/experiment_output_*/activations/{base,aligned}/*.npz`
- Annotations: `scotus-constitutional-geometry/experiment_output_*/annotations.json`

## Questions for Review

1. Is the 5-dim projection R² genuinely circular, or is R² > 0.99 meaningful given that Ridge with alpha=100 provides substantial regularization even with d >> n?
2. Is the transfer test (aligned probe → base activations) the right way to measure alignment's effect, or are there confounds?
3. Should we run a permutation null (shuffle y labels, refit, measure 5-dim R²) as the definitive test?
4. Are there better metrics entirely for this regime (49 samples, 5 targets, ~4500 features)?
