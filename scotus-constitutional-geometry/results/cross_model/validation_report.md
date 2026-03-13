# Validation Report: Constitutional Geometry R² Analysis

## 1. Original R² Assessment

### The Problem

The original cross-validated R² values are unreliable as point estimates due to the severe n << p regime (49 samples, ~3500-5100 features). We confirmed this instability empirically: the same Gemma 2-27B IT activations at layer 23 yield R² values ranging from +0.478 (original `probe_comparison.json`) to +0.116 (current sklearn version) — a swing of 0.36 on identical data with identical methodology.

### Permutation Test Validation

To determine whether the *qualitative* base-vs-IT pattern is real despite unreliable absolute values, we ran permutation tests (200 shuffles, seed=42) at each model's best-performing IT layer.

| Model | Layer | IT R² | IT p-value | Base R² | Base p-value |
|-------|-------|-----------|-----------------|---------|--------------|
| Gemma 2-27B | 23 | positive | **0.000** | negative | 0.745 |
| Llama 3.1-8B | 12 | near-zero | **0.005** | negative | 0.240 |
| Mistral-7B | 26 | near-zero | **0.000** | negative | 0.240 |
| Qwen 2.5-7B | 16 | near-zero | **0.005** | negative | 0.645 |
| Qwen 2.5-32B | 49 | negative | 0.200 | negative | 0.415 |

**Key finding**: In 4 out of 5 model families, IT model R² is significantly above the shuffled null distribution (p <= 0.005), while base model R² is never significant (p >= 0.24). This confirms the qualitative pattern — instruction tuning creates statistically detectable linear structure for constitutional principles that base models lack — even though the specific R² magnitudes are unreliable.

**Exception**: Qwen 2.5-32B IT activations do not reach significance (p = 0.20) at the tested layer. This may reflect a genuinely weaker signal in this model family, or a suboptimal layer choice (layer 49 of 64 was the original peak, but the original ranking was itself noisy). This model also showed the weakest original R² (+0.205).

**Note on absolute R² values**: Even the IT models show R² values that are negative or barely positive in the current environment. This is expected and does not undermine the permutation test result — the relevant comparison is real R² versus the shuffled distribution (which has mean R² around -2.7), not real R² versus zero. In the n << p regime, cross-validated R² is dominated by estimation noise and regularization artifacts; only the *relative* comparison to a null distribution is interpretable.


## 2. Probe Direction Analysis

### Motivation

If instruction tuning amplifies existing representations, we would expect base and IT models to encode constitutional principles in *similar* directions (high cosine similarity between their probe weight vectors). If instruction tuning creates *new* structure, the directions would differ (cosine similarity at or below the null expectation for random labels).

### Results

We trained Ridge probes independently on base and IT activations, then computed per-principle cosine similarity between weight vectors (5 x d_model). A permutation null (100 shuffles of principle labels) established the baseline cosine similarity expected by chance in each activation space.

| Model | Layer | Mean cos(base, IT) | Null mean | p-value |
|-------|-------|------------------------|-----------|---------|
| Gemma 2-27B | 23 | 0.274 | 0.262 | 0.260 |
| Llama 3.1-8B | 12 | 0.303 | 0.310 | 0.680 |
| Mistral-7B | 26 | 0.173 | 0.194 | 0.940 |
| Qwen 2.5-7B | 16 | 0.573 | 0.568 | 0.350 |
| Qwen 2.5-32B | 49 | 0.451 | 0.408 | **0.010** |

**Key finding**: For 4 out of 5 models, the observed cosine similarity between base and IT probe directions is **not significantly different from the null distribution** (p >= 0.26). This means the positive cosine values (0.17-0.57) reflect the ambient geometry of high-dimensional activation spaces, not principle-specific directional alignment.

**Interpretation**: Instruction tuning does not merely amplify directions that already exist in base models. Instead, it **reorganizes** the representational geometry, encoding constitutional principles in directions that are as different from the base model as random relabelings would predict. The constitutional structure detected by the permutation test (Section 1) genuinely *emerges* through instruction tuning rather than being latently present.

**Exception**: Qwen 2.5-32B shows significant direction sharing (p = 0.01), particularly for federalism (p = 0.000). Combined with its non-significant permutation R² (Section 1), this suggests Qwen 2.5-32B's base model may already encode some relevant structure at layer 49, but the encoding is too weak or noisy to produce significant linear probe performance.

**Note on cosine baselines**: The null cosine similarity varies by model (0.19 for Mistral to 0.57 for Qwen 2.5-7B). This reflects differences in activation geometry — models with more concentrated activation distributions produce higher baseline cosine between any two Ridge weight vectors. This is why the permutation null is essential: raw cosine similarity without it would be misleading.


## 3. Transfer Test with Controls

### Existing Result (Within-Model IT->Base Transfer)

The strongest existing metric applies IT model probe directions to base model activations within the same model family, measuring whether base activations project onto those directions with principle-correlated magnitudes.

From the existing cross-model R² comparison and the new analysis:

| Model | IT probe -> Base activations (mean Pearson r) |
|-------|---------------------------------------------------|
| Llama 3.1-8B | +0.83 |
| Mistral-7B | +0.50 |

This moderate-to-strong transfer confirms that base models encode *some* information along the directions that instruction tuning makes prominent, even though the base model's own probe directions differ (Section 2).

### Cross-Model Control

To test whether the within-model IT->base correlation reflects IT-specific structure or model-specific activation distributions, we performed cross-model transfers between the two models with matching d_model (Llama 3.1-8B and Mistral-7B, both d=4096).

| Transfer | Mean Pearson r |
|----------|---------------|
| **Within-model IT->base** | |
| Llama IT -> Llama base | **+0.83** |
| Mistral IT -> Mistral base | **+0.50** |
| **Cross-model IT->IT** | |
| Llama IT -> Mistral IT | +0.07 |
| Mistral IT -> Llama IT | +0.08 |
| **Cross-model IT->base** | |
| Llama IT -> Mistral base | +0.05 |
| Mistral IT -> Llama base | -0.11 |
| **Cross-model base->base** | |
| Llama base -> Mistral base | -0.02 |
| Mistral base -> Llama base | -0.15 |

**Key finding**: Cross-model transfer is **near zero in all conditions** (~0.07 for IT->IT, ~-0.03 for IT->base, ~-0.09 for base->base). This has two important implications:

1. **The within-model IT->base transfer (0.50-0.83) is a genuine within-family signal**, not an artifact of general activation statistics. If it were an artifact, cross-model transfers would show similar magnitudes.

2. **Constitutional representations are model-family-specific**. Each architecture develops its own unique linear encoding during instruction tuning. There is no universal "IT direction" shared across Llama and Mistral, even though both architectures independently create significant principle structure (Section 1).

**Limitation**: Only two models share d_model=4096, so this analysis is limited to one cross-model pair. The remaining models (Gemma d=4608, Qwen-7B d=3584, Qwen-32B d=5120) cannot be directly compared without dimensionality projection, which would introduce additional assumptions.


## 4. Revised Claims

Based on the validated evidence, the following claims are defensible:

> **Claim 1**: Post-training (instruction tuning) creates statistically significant linear structure encoding constitutional principles in transformer residual streams. Permutation testing (200 shuffles) confirms this in 4 of 5 model families tested (p <= 0.005), while no base model shows significant structure at the same layers (p >= 0.24).

> **Claim 2**: The constitutional structure created by instruction tuning uses novel representational directions rather than amplifying pre-existing ones. Base-IT probe direction cosine similarity is indistinguishable from a label-shuffled null in 4 of 5 models, indicating instruction tuning reorganizes rather than amplifies geometry.

> **Claim 3**: Within a model family, IT probe directions capture moderate signal in base activations (mean Pearson r = 0.50-0.83), but this transferability is entirely model-family-specific — cross-architecture transfer yields near-zero correlation (~0.07), indicating no universal IT geometry across architectures.

**Claims we cannot make**:
- We cannot compare R² magnitudes across models or report specific R² values as reliable point estimates.
- We cannot claim Qwen 2.5-32B shows the same IT geometry effect — its permutation test did not reach significance.
- We cannot claim instruction tuning creates a single universal subspace shared across architectures.


## 5. Metrics Going Forward

### Recommended Primary Metrics

1. **Permutation test p-value** — The most defensible metric for claiming signal presence. Should be the first-line validation for any new model or layer. Use >= 200 permutations for adequate resolution below p = 0.01.

2. **Within-model transfer correlation (IT probe -> base activations, Pearson r)** — The cleanest measure of instruction-tuning effect strength. Breaks circularity because base activations were not used to learn the probe. Report mean r across principles, with per-principle breakdown.

### Recommended Secondary Metrics

3. **Probe direction cosine similarity with null** — Addresses *how* instruction tuning changes representations (reorganization vs amplification). Report both observed cosine and the permutation null mean.

4. **Cross-model transfer correlation** — When models share d_model, provides evidence on universality of IT geometry. Near-zero values indicate family-specific representations.

### Metrics to Retire

- **Full-dimensional cross-validated R²**: Unstable in the n << p regime. Not interpretable as a point estimate. Only use as input to permutation tests.
- **5-dimensional projection R²**: Circular. Discarded.
- **Cross-model R² rankings**: The instability of underlying R² values makes rankings unreliable. Replace with permutation test significance (binary: significant or not) and transfer r magnitudes.

### Best Practices

- Always report permutation p-values alongside any R² or correlation metric
- Never compare R² magnitudes across models or sklearn versions
- Use fixed random seed (42) and report sklearn version for reproducibility
- When extending to new model families, test at multiple layers (not just the peak from an initial noisy scan)
