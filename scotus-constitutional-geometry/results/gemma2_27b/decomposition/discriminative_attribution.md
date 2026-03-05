# Discriminative Attribution Analysis

**Date**: 2026-02-24
**Model**: Gemma 2-27B (aligned)
**Probe Layer**: 23
**Cases**: 49
**Script**: `causal_validation/scripts/discriminative_attribution.py`

## Methodology

We decompose the residual stream at the probe layer into per-layer
additive contributions and measure each layer's **discriminative value**:
how well its projection onto probe directions correlates with ground-truth
principle weights across cases.

Since probes use StandardScaler + Ridge regression, they only exploit
between-case variation. A component that writes a large but constant vector
(identical across cases) contributes zero discriminative power. The metrics
used here directly capture what the probe can use:

- **Pearson r**: correlation between a layer's per-case projection onto the
  probe direction and the ground-truth principle weight
- **Std**: cross-case standard deviation of projection (raw variation
  available to the probe)

The residual stream is additive:
`resid_post[L] = resid_post[0] + sum_{l=1}^{L} (resid_post[l] - resid_post[l-1])`.
Each layer's contribution is `resid_post[l] - resid_post[l-1]`, and
`layers_0 = resid_post[0]` (embed + attn_0 + mlp_0 combined).

## Layer-Level Discriminative Attribution

Which layers write the most case-discriminative signal for each principle?

| Rank | Component | FreeExp r | EqualProt r | DueProc r | Federal r | Privacy r | Mean Abs(r) | Mean Std |
|-----:|-----------|----------:|------------:|----------:|----------:|----------:|------------:|---------:|
| 1 | layer_23 | +0.899 | +0.930 | +0.936 | +0.962 | +0.952 | 0.936 | 17.41 |
| 2 | layer_22 | +0.934 | +0.864 | +0.956 | +0.935 | +0.945 | 0.927 | 7.74 |
| 3 | layer_21 | +0.791 | +0.870 | +0.869 | +0.873 | +0.949 | 0.871 | 5.19 |
| 4 | layer_20 | +0.867 | +0.811 | +0.698 | +0.811 | +0.858 | 0.809 | 2.42 |
| 5 | layer_18 | +0.571 | +0.764 | +0.834 | +0.805 | +0.655 | 0.726 | 1.31 |
| 6 | layer_19 | +0.750 | +0.700 | +0.622 | +0.650 | +0.378 | 0.620 | 1.64 |
| 7 | layer_13 | +0.689 | +0.621 | +0.570 | +0.502 | +0.334 | 0.543 | 0.34 |
| 8 | layer_11 | +0.567 | +0.544 | +0.691 | +0.499 | +0.378 | 0.536 | 0.25 |
| 9 | layer_17 | +0.528 | +0.664 | +0.523 | +0.582 | +0.359 | 0.531 | 0.92 |
| 10 | layer_16 | +0.364 | +0.554 | +0.532 | +0.576 | +0.409 | 0.487 | 0.79 |
| 11 | layer_15 | +0.264 | +0.421 | +0.440 | +0.440 | +0.631 | 0.439 | 0.54 |
| 12 | layer_9 | +0.462 | +0.601 | +0.453 | +0.405 | +0.118 | 0.408 | 0.36 |
| 13 | layers_0 | +0.356 | +0.291 | +0.281 | +0.164 | +0.590 | 0.336 | 0.20 |
| 14 | layer_12 | +0.397 | +0.419 | +0.487 | -0.026 | +0.314 | 0.328 | 0.25 |
| 15 | layer_8 | +0.249 | +0.526 | +0.418 | +0.140 | +0.152 | 0.297 | 0.16 |
| 16 | layer_10 | +0.453 | +0.327 | +0.109 | +0.292 | +0.247 | 0.286 | 0.24 |
| 17 | layer_5 | +0.314 | +0.158 | -0.092 | +0.528 | +0.310 | 0.280 | 0.08 |
| 18 | layer_14 | +0.237 | +0.328 | +0.312 | +0.188 | +0.169 | 0.247 | 0.43 |
| 19 | layer_7 | +0.209 | -0.010 | +0.449 | +0.135 | +0.290 | 0.218 | 0.11 |
| 20 | layer_4 | +0.105 | +0.139 | +0.378 | +0.106 | +0.185 | 0.183 | 0.09 |
| 21 | layer_3 | +0.182 | +0.270 | +0.008 | +0.105 | +0.249 | 0.163 | 0.08 |
| 22 | layer_6 | -0.072 | +0.178 | -0.238 | -0.023 | +0.130 | 0.128 | 0.10 |
| 23 | layer_1 | +0.061 | +0.178 | -0.054 | +0.216 | -0.106 | 0.123 | 0.08 |
| 24 | layer_2 | -0.156 | +0.087 | -0.207 | -0.045 | -0.116 | 0.122 | 0.06 |

The top four layers (20-23) show strong correlations with ground-truth principle weights across all five principles (r = 0.70-0.96). Shallow layers contribute minimally. `layers_0` (embed + mlp_0) ranks 13/24 with mean |r| = 0.336. Note that mlp_0 has the largest *absolute* projection magnitude (61-68 units) but near-zero cross-case variation (std < 0.25, CoV < 0.4%) — it writes a large constant offset reflecting shared prompt vocabulary, which the probe's StandardScaler subtracts out.

## Variance Decomposition

What fraction of cross-case variance in the full probe-direction projection comes from each layer?

### Free Expression

| Rank | Component | Variance | % of Total |
|-----:|-----------|--------:|-----------:|
| 1 | layer_23 | 507.6005 | 30.8% |
| 2 | layer_22 | 68.5650 | 4.2% |
| 3 | layer_21 | 45.4531 | 2.8% |
| 4 | layer_20 | 9.6455 | 0.6% |
| 5 | layer_19 | 3.7027 | 0.2% |
| 6 | layer_18 | 1.8330 | 0.1% |
| 7 | layer_17 | 1.4558 | 0.1% |
| 8 | layer_16 | 1.0317 | 0.1% |
| 9 | layer_14 | 0.4131 | 0.0% |
| 10 | layer_15 | 0.1990 | 0.0% |

### Equal Protection

| Rank | Component | Variance | % of Total |
|-----:|-----------|--------:|-----------:|
| 1 | layer_23 | 220.6334 | 21.3% |
| 2 | layer_22 | 68.7902 | 6.6% |
| 3 | layer_21 | 29.1086 | 2.8% |
| 4 | layer_20 | 4.5888 | 0.4% |
| 5 | layer_19 | 2.8943 | 0.3% |
| 6 | layer_18 | 2.7905 | 0.3% |
| 7 | layer_17 | 0.6243 | 0.1% |
| 8 | layer_16 | 0.5386 | 0.1% |
| 9 | layer_9 | 0.2297 | 0.0% |
| 10 | layer_15 | 0.1938 | 0.0% |

### Due Process

| Rank | Component | Variance | % of Total |
|-----:|-----------|--------:|-----------:|
| 1 | layer_23 | 362.5214 | 27.7% |
| 2 | layer_22 | 74.4325 | 5.7% |
| 3 | layer_21 | 23.6659 | 1.8% |
| 4 | layer_20 | 7.0926 | 0.5% |
| 5 | layer_19 | 1.6892 | 0.1% |
| 6 | layer_18 | 1.5024 | 0.1% |
| 7 | layer_17 | 1.1667 | 0.1% |
| 8 | layer_16 | 0.6063 | 0.0% |
| 9 | layer_15 | 0.2330 | 0.0% |
| 10 | layer_9 | 0.1360 | 0.0% |

### Federalism

| Rank | Component | Variance | % of Total |
|-----:|-----------|--------:|-----------:|
| 1 | layer_23 | 252.8534 | 26.8% |
| 2 | layer_22 | 38.4175 | 4.1% |
| 3 | layer_21 | 26.6226 | 2.8% |
| 4 | layer_20 | 3.4375 | 0.4% |
| 5 | layer_19 | 2.4378 | 0.3% |
| 6 | layer_18 | 1.4920 | 0.2% |
| 7 | layer_17 | 0.5423 | 0.1% |
| 8 | layer_16 | 0.3166 | 0.0% |
| 9 | layer_15 | 0.3047 | 0.0% |
| 10 | layer_14 | 0.0954 | 0.0% |

### Privacy Liberty

| Rank | Component | Variance | % of Total |
|-----:|-----------|--------:|-----------:|
| 1 | layer_23 | 216.4197 | 24.5% |
| 2 | layer_22 | 52.9677 | 6.0% |
| 3 | layer_21 | 14.5339 | 1.6% |
| 4 | layer_20 | 5.4954 | 0.6% |
| 5 | layer_19 | 2.9525 | 0.3% |
| 6 | layer_18 | 1.1260 | 0.1% |
| 7 | layer_16 | 0.7224 | 0.1% |
| 8 | layer_15 | 0.5983 | 0.1% |
| 9 | layer_17 | 0.5805 | 0.1% |
| 10 | layer_14 | 0.1824 | 0.0% |

Note: Per-component variances do not sum to 100% of total because the total variance of the sum includes covariance terms between layers.

## Progressive Ablation

Does removing shallow layers change probe R²? This tests whether the case-discriminative signal is recoverable without early processing.

| Ablation | Var Preserved | Overall R² | FreeExp | EqualProt | DueProc | Federal | Privacy |
|----------|-------------:|----------:|--------:|----------:|--------:|--------:|--------:|
| None (baseline) | 100.0% | 0.1164 | 0.8078 | -1.7238 | 0.6883 | 0.3488 | 0.4876 |
| Remove layer 0 | 100.0% | 0.1159 | 0.8082 | -1.7356 | 0.6888 | 0.3544 | 0.4894 |
| Remove layers 0-4 | 100.0% | 0.1056 | 0.8093 | -1.7898 | 0.6891 | 0.3540 | 0.4898 |
| Remove layers 0-9 | 99.7% | 0.1065 | 0.8093 | -1.7731 | 0.6835 | 0.3476 | 0.4874 |
| Remove layers 0-14 | 96.6% | 0.0989 | 0.8038 | -1.8170 | 0.6878 | 0.3491 | 0.4930 |

Removing layer 0 preserves 100% of cross-case variance and has negligible effect on any principle's R². Even removing the first 15 layers (0-14) preserves 96.6% of variance and leaves per-principle R² essentially unchanged.

## Interpretation

The case-discriminative signal that probes exploit comes from **deep layers (20-23)**, which show strong per-case correlations with ground-truth principle weights (r = 0.81-0.96). These are the final processing layers before the probe reads the residual stream — they integrate upstream computation and write the most case-specific representations.

Key findings:

- **Top discriminative layers**: layer_23 (mean |r|=0.936), layer_22 (0.927), layer_21 (0.871)
- **layers_0 (embed+mlp_0)** ranks 13/24 by discriminative value (mean |r|=0.336), despite having the largest absolute projection magnitude
- **Ablation confirms**: removing layers 0-14 preserves >96% of cross-case variance and barely changes R²
- **Signal is distributed**: layers 20-23 all contribute strongly, with no single specialist layer — consistent with superposition

### Implications

The probe geometry is written by deep computation, not shallow lexical features. However, the steering null results still stand: this deep structure cannot be causally manipulated via linear activation addition. The signal is distributed across multiple late layers with no concentrated specialist circuit, which may explain why targeted steering interventions fail — there is no single bottleneck to intervene on.
