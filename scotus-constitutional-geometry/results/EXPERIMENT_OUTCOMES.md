# Constitutional Geometry Experiment Results

*Last updated: March 2026*

## Results Summary

**Research question**: Does instruction tuning create more linearly separable representations of constitutional principles in transformer residual streams — and if so, are those representations causally linked to model behavior?

**Result**: Yes — across three converging lines of evidence:

1. **Probing** (6 model families): Instruction-tuned models show substantially stronger linear encoding of abstract legal principles than base models. Scale does NOT create alignment geometry — Gemma 2-27B base (R²=0.04) matches Llama 3.2-3B base (-0.25) despite 9x scale difference, yet both reach ~0.48 R² after instruction tuning. Permutation testing confirms the signal in 4/5 families (p ≤ 0.005).

2. **Causal validation** (Gemma 2-27B): Activation patching at layers 20-34 achieves 100% recovery of IT behavior in the base model. Probe-derived steering vectors are partially causal — at 10% of residual stream norm, they cause targeted principles to appear in 44% of cases where absent at baseline. A narrow coherence window exists: below 3% of ‖resid‖ no effect is observable; above 50% generation collapses into multilingual gibberish.

3. **Mechanistic decomposition** (Gemma 2-27B): Constitutional principle encoding is dominated by attention layers 20-22 (attn_22 mean |r|=0.882), with attention contributing more than MLP (mean |r| 0.376 vs 0.280). Specialist heads attend to case facts and legal context (<1% attention to principle-name tokens), indicating conceptual synthesis rather than keyword matching.

![Layer-by-layer R² comparison between base and instruction-tuned models](llama32_3b/layer_comparison.png)

### Key Metrics

| Model | Best Layer | Best R² | IT Δ | Perm. p-value |
|-------|------------|---------|--------|---------------|
| Llama-3.2-3B (base) | Layer 6 | -0.25 | — | n.s. |
| Llama-3.2-3B-Instruct | Layer 27 | +0.49 | **+0.74** | **p=0.000** |
| Llama-3.1-8B (base) | Layer 30 | +0.24 | — | p=0.240 |
| Llama-3.1-8B-Instruct | Layer 12 | +0.41 | **+0.18** | **p=0.005** |
| Mistral-7B (base) | Layer 15 | +0.26 | — | p=0.240 |
| Mistral-7B-Instruct | Layer 26 | +0.40 | **+0.14** | **p=0.000** |
| Qwen2.5-7B (base) | Layer 3 | **-0.14** | — | p=0.645 |
| Qwen2.5-7B-Instruct | Layer 16 | **+0.23** | **+0.37** | **p=0.005** |
| Qwen2.5-32B (base) | Layer 29 | +0.06 | — | p=0.415 |
| Qwen2.5-32B-Instruct | Layer 49 | +0.21 | **+0.14** | p=0.200 |
| **Gemma-2-27B (base)** | Layer 11 | **+0.04** | — | p=0.745 |
| **Gemma-2-27B-it** | Layer 23 | **+0.48** | **+0.43** | **p=0.000** |

> **Robustness note on R² and permutation p-values**: The R² column reports cross-validated R² from Ridge regression (5-fold CV, RidgeCV alpha selection). With 49 samples and 3072–5120 features, R² operates in a severely underdetermined regime (n << p) where absolute values are sensitive to compute environment (BLAS/LAPACK backend). We validated the Llama 3.2-3B values as perfectly reproducible across environments (Pearson r = 1.000 between original and re-run layer curves); RunPod-generated models show stable IT curve *shapes* (Pearson r = 0.89–0.96) with absolute shifts of ~0.3–0.6. The **Perm. p-value** column provides an environment-independent validation: each model's R² was compared against a null distribution of 200 label-shuffled permutations. Instruction-tuned models significantly exceed their null (4/5 families at p ≤ 0.005); no base model reaches significance. The R² values should be read as directional indicators of effect strength — the permutation p-values confirm which effects are statistically real. See the Phase 6 validation section for full details.
>
> **Exception**: Qwen2.5-32B-Instruct does not reach significance (p = 0.200), consistent with it having the weakest signal in the study.

### Key Observations

**Probing (Phases 1-6)**:
- **Instruction tuning creates constitutional geometry**: IT > base in all model families, confirmed by permutation testing (4/5 at p ≤ 0.005)
- **Scale does NOT create alignment geometry**: Gemma 2-27B base (R²=0.04) matches Llama 3.2-3B base (-0.25) despite 9x scale difference
- **Ceiling effect**: Western models converge at ~0.40-0.50 IT R² regardless of scale (3B-27B)
- **Qwen divergence**: Weaker signal at all scales (IT R² = 0.21-0.23), likely due to training data/cultural differences
- **IT reorganizes geometry**: Probe direction cosine similarity between base and IT is indistinguishable from shuffled null (4/5 models)
- **Representations are model-family-specific**: Cross-model transfer yields near-zero correlation (~0.07), while within-family IT→base transfer is moderate-strong (r = 0.50-0.83)

**Behavioral verification (Phase 7)**:
- IT models dramatically outperform base on constitutional principle identification (Gemma: 57%→92%, Mistral: 14%→88%, Llama-3B: 41%→74%)
- **Exception**: Qwen base already near-ceiling (86%→88%), consistent with strong pre-training on legal text

**Causal validation (Phase 8)**:
- Activation patching recovers IT behavior only in Gemma 2-27B (100% recovery); fails or breaks generation in 7B/3B models
- Steering works within a narrow 10% ‖resid‖ window — directions control principle *salience* (whether mentioned) but not *ranking*
- Causal interpretability techniques appear scale-dependent

**Mechanistic decomposition (Phase 9)**:
- Deep attention layers 20-22 dominate constitutional encoding in Gemma 2-27B
- No single specialist head — signal distributed across ~20 heads in layers 17-22
- Heads perform conceptual synthesis from case facts, not keyword detection

### Interpretation

Instruction tuning creates genuine geometric structure for constitutional concepts — **real, validated, and partially causal**. The probe directions are not merely epiphenomenal: they can be patched to recover IT behavior (Gemma 2-27B, 100% in-distribution) and steered to control which principles appear in model output (44% appearance rate). However, the causal role is architecture- and scale-dependent — patching fails for smaller models, and steering controls salience rather than ranking.

The Gemma 2-27B results **refute** the hypothesis that larger models develop constitutional structure during pretraining. The mechanistic decomposition reveals this structure is computed via conceptual synthesis in deep attention layers, not keyword matching — specialist heads attend to case facts and legal context while producing principle-discriminative outputs.

These results suggest **instruction tuning or post-training may be needed for conceptual emergence** in many cases — base models don't reliably produce these representations by default.

---

*Detailed methodology and run logs below.*

---

# SCOTUS Constitutional Geometry - Experiment Outcomes

## Document Purpose

This document tracks experimental findings as they emerge. When the research is complete, key findings will be summarized in the project README and the root `ai-alignment-research` README.

---

## Phase 1: Initial Proof-of-Concept (Llama 3.2-3B)

**Date**: 2025-11-26
**Status**: Complete - Pattern confirmed, effect size moderate

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | meta-llama/Llama-3.2-3B |
| **IT Model** | meta-llama/Llama-3.2-3B-Instruct |
| **Architecture** | 28 layers, 3072 hidden dimensions |
| **Sample Size** | 28 landmark SCOTUS cases |
| **Annotation Source** | Claude Opus (claude-opus-4-5-20251101) |
| **Probe Method** | Ridge regression with RidgeCV alpha selection |
| **Cross-Validation** | 5-fold CV with shuffle (random_state=42) |
| **Regularization Alphas** | [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] |
| **Activation Extraction** | Last token residual stream per layer |

**Note on sample evolution**: Initial probe run used 22 annotated cases. After completing all 28 annotations, we re-ran probes. Results below reflect full 28-case dataset.

**Note on R² interpretation**: R² (coefficient of determination) measures how well probe predictions explain variance in the target. R² = 1.0 means perfect prediction; R² = 0.0 means predictions are no better than predicting the mean. **Negative R²** indicates predictions are *worse* than the mean baseline—the probe fails to learn generalizable structure. Negative R² does not imply "anti-correlation"; it indicates absence of linearly-recoverable structure in the activations.

### Constitutional Principles Probed

1. **Free Expression** (1st Amendment)
2. **Equal Protection** (14th Amendment)
3. **Due Process** (5th/14th Amendment)
4. **Federalism** (10th Amendment, structural)
5. **Privacy/Liberty** (penumbras, unenumerated rights)

---

### Key Finding 1: Instruction-Tuned Model Shows Improved Encoding

The instruction-tuned model (Llama-3.2-3B-Instruct) activations show markedly better structure than base, reaching near-zero R² (vs deeply negative base):

| Layer | IT R² | Interpretation |
|-------|------------|----------------|
| 0-14 | -1.12 to -0.18 | Negative, similar pattern to base |
| 15-19 | -0.38 to -0.12 | Improving toward zero |
| 20-27 | -0.27 to **+0.02** | Near-zero to slightly positive |

**Best IT layer**: Layer 27 with **R² = +0.02** (cross-validated, 28 cases)

While absolute R² is near zero, this represents a **+1.30 improvement** over the base model at the same layer, indicating meaningful structural differences from instruction tuning.

---

### Key Finding 2: Base Model Shows Strongly Negative Predictive Power

The base model (Llama-3.2-3B) activations have **deeply negative** R² across all layers:

| Layer | Base R² | Interpretation |
|-------|---------|----------------|
| 0-10 | -0.56 to -1.10 | Strongly negative |
| 11-20 | -0.65 to -1.36 | Worsening |
| 21-27 | -1.29 to -1.73 | **Deeply negative** |

**Best base layer**: Layer 7 with **R² = -0.40** (still negative)

The negative R² values indicate that the base model lacks linearly-recoverable constitutional principle structure. Linear probes trained on base model activations fail to generalize—their predictions on held-out data have larger squared errors than simply predicting the mean. This suggests the base model has no stable linear encoding of these principles that transfers across cases.

---

### Key Finding 3: IT-Base Gap Grows in Upper Layers

**Pre-experiment expectation** (from README):
> "Peak in mid-layers: Matches interpretability literature"

**Actual finding**: The instruction-tuned model's advantage over base peaks in the **final layers** (24-27), not mid-layers.

| Layer Range | R² Difference (IT - Base) |
|-------------|-------------------------------|
| 0-2 | +0.31 to +0.80 | Moderate advantage |
| 3-14 | -0.42 to +0.02 | Mixed/inconsistent |
| 15-19 | +0.83 to +1.08 | Strong advantage emerges |
| 20-27 | **+1.09 to +1.70** | Very strong advantage |

The IT model advantage peaks at **+1.70 R²** at layers 25-26 (meaning IT is near-zero while base is deeply negative at -1.70).

**Interpretation**: Instruction tuning creates value-aligned geometry specifically in the final processing stages, where representations are most "output-facing."

---

### Comparison to Pre-Experiment Success Criteria

From README.md:

| Criterion | Expected | Actual | Met? |
|-----------|----------|--------|------|
| R² (base) > 0.15 | Yes | **-0.40** (best layer) | **NO** - Deeply negative |
| R² (IT) > R² (base) | Yes | **+0.02 vs -1.29** at layer 27 | **YES** - +1.30 gap |
| Peak in mid-layers | Yes | **Final layers (24-27)** | **NO** - But pattern is clear |

**Overall assessment**: Core hypothesis **confirmed** - instruction tuning dramatically improves linear separability of constitutional principles. Effect concentrated in upper layers, not mid-layers as expected.

---

### Refined Analysis: CV Stability Testing

**Date**: 2025-11-28
**Purpose**: Assess reliability of R² estimates given small sample size

**Issue identified**: Initial R² = 0.48 for instruction-tuned model (22 cases, seed=42) appeared to be a favorable random draw. With 28 cases, single-seed R² dropped to 0.02.

**Method**: Re-ran 5-fold CV with 10 different random seeds to assess estimator variance.

**Results (Layer 27, 28 cases, 10 seeds)**:

| Model | Mean R² | Std R² | Range |
|-------|---------|--------|-------|
| Base | **-2.90** | 4.01 | -12.3 to -0.04 |
| IT | **+0.11** | 0.61 | -1.05 to +0.70 |

**Gap Analysis**:
- Mean gap (IT - Base): **+3.01**
- Gap positive in: **10/10 seeds (100%)**
- Paired t-test: **t=2.56, p=0.03**

**Interpretation**:
1. The initial R² = 0.48 was indeed a lucky draw (true mean ~0.11)
2. However, the **IT > base gap is robust** and statistically significant
3. Base model is consistently deeply negative; IT is near-zero to positive
4. High variance in R² estimates indicates need for more samples

**Revised Conclusions**:
- Core finding **confirmed**: Instruction tuning improves constitutional principle encoding
- Effect size is **moderate** (mean R² ~0.11 for IT vs ~-2.9 for base)
- Results are **statistically significant** (p=0.03) despite high variance
- More samples needed for precise R² estimates

---

### Limitations and Caveats

1. **Small sample size**: 28 cases may not capture full distribution of constitutional reasoning
2. **Single model family**: Results from Llama 3.2 only; cross-model replication needed
3. **Annotation validity**: Opus annotations not yet independently validated
4. **No permutation test**: Could be spurious correlation; shuffle test needed
5. **Confounds possible**: Cases may cluster by era, court composition, etc.

---

### Output Artifacts

| File | Description |
|------|-------------|
| `probe_comparison.json` | Full layer-by-layer R² scores and per-principle breakdowns |
| `layer_comparison.png` | Visualization of R² by layer for base vs IT |
| `annotations.json` | Opus-generated principle weights with justifications |
| `activations/base/*.npz` | Cached base model activations (28 layers × 3072 dims) |
| `activations/aligned/*.npz` | Cached IT model activations |

---

## Validation Results

### Permutation Test (Completed)

**Date**: 2025-11-28
**Purpose**: Verify signal is not due to overfitting or confounds

**Method**: Shuffle principle weights across cases (so case A's activations are paired with case B's labels), then re-run probes.

**Results**:

| Layer | Real R² | Shuffled R² | Interpretation |
|-------|---------|-------------|----------------|
| 15 | +0.36 | **-3.21** | Signal destroyed |
| 20 | +0.36 | **-3.06** | Signal destroyed |
| 25 | +0.37 | **-3.57** | Signal destroyed |
| 27 | +0.48 | **-3.18** | Signal destroyed |

**Conclusion**: When case-principle correspondence is broken, R² drops from positive to deeply negative. **The signal is genuine** - the instruction-tuned model's activations truly encode constitutional principle structure that matches Opus's annotations.

---

### Sonnet Cross-Validation (Completed)

**Date**: 2025-11-28
**Purpose**: Validate Opus annotations using Claude Sonnet as independent reviewer

**Method**: Sonnet reviewed 5 case annotations against actual opinion text, assessing accuracy of principle weights.

**Results**:

| Case | Sonnet Assessment | Key Notes |
|------|-------------------|-----------|
| Tinker v. Des Moines (1969) | minor_issues | Suggested due_process 0.15→0.25 |
| Loving v. Virginia (1967) | minor_issues | Suggested due_process 0.6→0.7, privacy 0.5→0.65 |
| NFIB v. Sebelius (2012) | minor_issues | Suggested federalism 0.95→0.85 |
| **Mathews v. Eldridge (1976)** | **accurate** | Perfect agreement on due_process: 1.0 |
| Roe v. Wade (1973) | minor_issues | Suggested privacy 0.9→1.0, due_process 0.6→0.8 |

**Conclusion**: All Opus annotations rated "accurate" or "minor_issues" (adjustments of ±0.1-0.2). No major disagreements on which principles are present/dominant. **Annotations are valid ground truth for the experiment.**

---

## Planned Validation Steps

### Immediate (This Week)
- [x] **Permutation test**: Shuffle principle labels, verify R² drops to ~0 ✓ PASSED
- [x] **Sonnet cross-validation**: Validate 5 Opus annotations ✓ PASSED (see below)
- [x] **Llama-3.1-8B replication**: Run on larger model via RunPod ✓ COMPLETED (see Phase 3)

### Short-Term (Next 2 Weeks)
- [x] **Cross-model replication**: ~~Mistral-7B, Llama-2-7B~~ ✓ Completed with Mistral-7B and Qwen2.5-7B (Dec 2025)
- [ ] **Behavioral divergence test**: Do models respond differently to prompts?
- [ ] **Bootstrap confidence intervals**: Quantify uncertainty on R² estimates

### Medium-Term (Month 2)
- [x] **Expanded case set**: ~~Add 50+ additional cases~~ Added 21 cases (49 total) ✓ COMPLETED
- [ ] **Layer intervention**: Ablate specific layers to test causal role
- [ ] **Alternative probe architectures**: MLP probes, attention probes

---

## Phase 2: Expanded Sample (49 Cases)

**Date**: 2025-11-28
**Status**: Complete - Effect size substantially increased with more samples

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | meta-llama/Llama-3.2-3B |
| **IT Model** | meta-llama/Llama-3.2-3B-Instruct |
| **Architecture** | 28 layers, 3072 hidden dimensions |
| **Sample Size** | **49 landmark SCOTUS cases** (+21 from Phase 1) |
| **Annotation Source** | Claude Opus (claude-opus-4-5-20251101) |
| **Case Data Format** | JSON files in `data/cases/` for transparency |

### Case Distribution by Principle

| Principle | Phase 1 | Phase 2 | Total |
|-----------|---------|---------|-------|
| Free Expression | 6 | 4 | **10** |
| Equal Protection | 6 | 4 | **10** |
| Due Process | 6 | 4 | **10** |
| Federalism | 5 | 4 | **9** |
| Privacy/Liberty | 5 | 5 | **10** |
| **Total** | **28** | **21** | **49** |

---

### Key Finding 1: Instruction-Tuned Model Shows Substantially Positive R²

With 49 cases, the instruction-tuned model now shows **clearly positive** R² in upper layers:

| Layer Range | IT R² | Interpretation |
|-------------|------------|----------------|
| 0-10 | -1.02 to -0.73 | Negative, similar to base |
| 11-14 | -0.25 to +0.00 | Approaching zero |
| 15-20 | **+0.31 to +0.43** | Positive, moderate |
| 21-27 | **+0.45 to +0.50** | **Strong positive** |

**Best IT layer**: Layer 27 with **R² = +0.49** (cross-validated, 49 cases)*

This is a **substantial improvement** over Phase 1's R² = +0.02 to +0.11, demonstrating that more samples stabilize and strengthen the signal.

*_Results updated after annotation corrections (see "Data Quality Corrections" below)._

---

### Key Finding 2: Base Model Remains Deeply Negative

The base model (Llama-3.2-3B) continues to show **deeply negative** R² across all layers:

| Layer Range | Base R² | Interpretation |
|-------------|---------|----------------|
| 0-10 | -0.79 to -0.91 | Strongly negative |
| 11-14 | -0.74 to -0.86 | Strongly negative |
| 15-20 | -1.17 to -2.07 | **Very deeply negative** |
| 21-27 | -0.51 to -1.41 | Deeply negative |

**Best base layer**: Layer 6 with **R² = -0.25** (still negative)

This confirms Phase 1 findings: the base model shows no linearly-recoverable constitutional principle structure (probes fail to generalize beyond chance).

---

### Key Finding 3: Gap Peaks in Mid-to-Upper Layers

| Layer Range | R² Difference (IT - Base) | Interpretation |
|-------------|-------------------------------|----------------|
| 0-10 | -0.54 to +0.17 | Mixed |
| 11-14 | +0.54 to +0.87 | Strong advantage emerges |
| 15-20 | **+1.44 to +2.37** | **Peak advantage** |
| 21-27 | +1.00 to +1.84 | Very strong advantage |

The IT model advantage **peaks at +2.37** at layer 20, indicating instruction tuning creates dramatic value-aligned restructuring in the mid-to-upper processing stages.*

---

### Comparison: Phase 1 vs Phase 2

| Metric | Phase 1 (28 cases) | Phase 2 (49 cases)* | Change |
|--------|-------------------|-------------------|--------|
| Best Base R² | -0.40 (L7) | -0.24 (L6) | Slightly better |
| Best IT R² | +0.02 (L27) | **+0.49 (L27)** | **+0.47** |
| Peak Gap | +1.70 (L25-26) | **+2.37 (L20)** | **+0.67** |
| Mean IT R² (L20-27) | ~0.11 | ~+0.44 | **+0.33** |

**Key insight**: More samples → more stable and stronger IT model signal. Phase 1's R² variance was high due to small N; Phase 2 confirms the effect with substantially tighter estimates.

*_Results after data quality corrections (see below)._

---

### Interpretation

1. **Instruction tuning creates value-aligned geometry**: The IT model's activations can be linearly decoded to predict constitutional principle weights (R² = 0.49), while the base model cannot (R² = -0.24).

2. **Effect concentrated in upper layers**: The IT model advantage emerges around layer 11 and peaks at layers 15-21, suggesting instruction tuning restructures the later stages of processing where representations are most "output-facing."

3. **Scaling with sample size**: R² improved from ~0.11 (28 cases) to 0.49 (49 cases). More cases provide better estimates and likely capture more of the principle structure.

4. **Robust pattern**: The IT > base gap is consistent across:
   - Both phases (28 and 49 cases)
   - Multiple random seeds (100% of seeds in Phase 1 CV analysis)
   - Permutation test (signal destroyed when labels shuffled)
   - Data quality corrections (results changed by only ~0.01 R² after fixing errors)

---

### New Cases Added in Phase 2

**Free Expression**: NYT v. Sullivan (1964), Cohen v. California (1971), Hustler v. Falwell (1988), Reno v. ACLU (1997)

**Equal Protection**: Plessy v. Ferguson (1896), Craig v. Boren (1976), Bakke (1978), US v. Virginia (1996)

**Due Process**: Mapp v. Ohio (1961), Rochin v. California (1952), Goldberg v. Kelly (1970), Casey (1992)

**Federalism**: Gibbons v. Ogden (1824), Heart of Atlanta (1964), US v. Morrison (2000), Gonzales v. Raich (2005)

**Privacy/Liberty**: Eisenstadt v. Baird (1972), Glucksberg (1997), Katz v. US (1967), Whalen v. Roe (1977), Troxel v. Granville (2000)

---

### Sonnet Cross-Validation of Phase 2 Annotations

**Purpose**: Validate all 21 Phase 2 Opus annotations using Claude Sonnet as independent reviewer.

**Results**:

| Assessment | Count | Percentage |
|------------|-------|------------|
| **accurate** | 4 | 19% |
| **minor_issues** | 15 | 71% |
| **significant_issues** | 2 | 10% |

**Cases rated "accurate"**:
- Goldberg v. Kelly (1970) - due_process: 1.0 ✓
- Reno v. ACLU (1997) - free_expression: 1.0 ✓
- Gonzales v. Raich (2005) - federalism: 0.95 ✓
- Gibbons v. Ogden (1824) - federalism: 1.0 ✓

**Cases with "significant_issues"**:
- Regents v. Bakke (1978): Sonnet suggests equal_protection 1.0→0.85, due_process 0.0→0.15 (Bakke involves both principles)
- Hustler v. Falwell (1988): **Wrong opinion fetched** - CourtListener returned Riley v. National Federation of the Blind instead

**Typical adjustments**: ±0.1-0.2 on secondary principles. No major disagreements on which principles are dominant.

**Conclusion**: Sonnet cross-validation identified data quality issues requiring correction (see below).

---

### Data Quality Corrections

**Date**: 2025-11-28
**Purpose**: Address issues identified in Sonnet cross-validation

**Issue 1: Hustler v. Falwell (1988) - Wrong Opinion Text**

During Phase 2 fetch, CourtListener returned the wrong case (cluster 112141 = Riley v. National Federation of the Blind) instead of Hustler v. Falwell (cluster 112011). This was detected by Sonnet cross-validation, which noted the opinion text discussed "North Carolina charitable solicitation" rather than the Falwell parody case.

**Fix**: Re-fetched correct opinion from cluster 112011 and re-annotated. New weights:
- free_expression: 1.0 (unchanged conceptually, now based on correct text)
- federalism: 0.1 (minor federalism aspect)

**Issue 2: Regents v. Bakke (1978) - Annotation Adjustment**

Sonnet correctly identified that Bakke involves significant Title VI statutory interpretation alongside Equal Protection analysis, warranting adjustment.

**Fix**: Updated weights per Sonnet's suggestion:
- equal_protection: 1.0 → 0.85
- due_process: 0.0 → 0.15

**Impact on Results**:

| Metric | Before Corrections | After Corrections | Change |
|--------|-------------------|-------------------|--------|
| Best IT R² | +0.50 | +0.49 | -0.01 |
| Peak Gap | +2.39 | +2.37 | -0.02 |

**Conclusion**: Corrections had **minimal impact** (~0.01 R² change), demonstrating the robustness of findings to annotation errors.

---

### Updated Output Artifacts

| File | Description |
|------|-------------|
| `data/cases/phase1_cases.json` | 28 Phase 1 cases with metadata |
| `data/cases/phase2_cases.json` | 21 Phase 2 cases with metadata |
| `probe_comparison.json` | Full layer-by-layer R² scores (49 cases) |
| `layer_comparison.png` | Visualization of R² by layer |
| `annotations.json` | Opus-generated principle weights (49 cases) |
| `sonnet_validation_phase2.json` | Sonnet validation of 21 Phase 2 annotations |
| `activations/base/*.npz` | Base model activations (49 cases) |
| `activations/aligned/*.npz` | IT model activations (49 cases) |

---

## Phase 3: Model Size Comparison (Llama 3.1-8B)

**Date**: 2025-12-11
**Status**: Complete - Model size significantly affects instruction-tuning impact

### Motivation

Phase 1 and 2 results with Llama-3.2-3B showed dramatic instruction-tuning effects (base R² = -0.25, IT R² = +0.49). However, our parallel experiment on criminal planning prompts with Llama-3.1-8B showed surprisingly modest instruction-tuning effects. This raised the question: **Is the instruction-tuning effect model-size dependent?**

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | meta-llama/Llama-3.1-8B |
| **IT Model** | meta-llama/Llama-3.1-8B-Instruct |
| **Architecture** | 32 layers, 4096 hidden dimensions |
| **Sample Size** | 49 SCOTUS cases (same as Phase 2) |
| **Annotation Source** | Same Opus annotations as Phase 2 |
| **Execution Environment** | RunPod GPU instance |

---

### Key Finding 1: Base Model Already Encodes Constitutional Principles

Unlike the 3B base model, the **8B base model shows positive R²** in upper layers:

| Layer Range | 8B Base R² | 3B Base R² | Interpretation |
|-------------|------------|------------|----------------|
| 0-10 | -0.51 to +0.05 | -0.79 to -0.91 | 8B much better |
| 11-20 | -0.04 to +0.08 | -0.74 to -2.07 | 8B near zero, 3B deeply negative |
| 21-27 | -0.04 to +0.07 | -0.51 to -1.41 | 8B stable, 3B negative |
| 28-31 | +0.05 to **+0.24** | N/A (only 28 layers) | **8B positive in final layers** |

**Best 8B base layer**: Layer 30 with **R² = +0.24**

The 8B base model has learned constitutional reasoning structure during pretraining alone — no instruction tuning required.

---

### Key Finding 2: IT Models Similar Across Sizes

Both instruction-tuned models reach similar peak performance:

| Model | Best Layer | Best R² |
|-------|------------|---------|
| Llama-3.2-3B-Instruct | Layer 27 | +0.49 |
| Llama-3.1-8B-Instruct | Layer 12 | +0.41 |

The 8B IT model peaks earlier (layer 12 vs 27), but both achieve R² in the 0.4-0.5 range.

---

### Key Finding 3: Instruction-Tuning Effect Dramatically Reduced in Larger Model

| Model Size | Base R² | IT R² | IT Δ |
|------------|---------|------------|--------|
| **3B** | -0.25 | +0.49 | **+0.74** |
| **8B** | +0.24 | +0.41 | **+0.18** |

The instruction-tuning-attributable improvement drops by **75%** (from +0.74 to +0.18) when moving from 3B to 8B.

---

### Per-Principle Analysis (8B Models)

At the best IT layer (Layer 12):

| Principle | Base R² | IT R² | IT Δ |
|-----------|---------|------------|--------|
| Equal Protection | +0.21 | **+0.53** | +0.32 |
| Privacy/Liberty | +0.37 | **+0.57** | +0.20 |
| Federalism | +0.47 | +0.50 | +0.03 |
| Free Expression | -0.30 | +0.29 | +0.59 |
| Due Process | -0.27 | +0.23 | +0.50 |

**Observations**:
- Federalism and Privacy/Liberty already well-encoded in base model
- Free Expression and Due Process show larger IT improvements
- Equal Protection shows moderate improvement

---

### Interpretation

1. **Pretraining encodes values at scale**: The 8B base model has learned constitutional reasoning structure through exposure to legal text during pretraining. Instruction tuning is not creating these representations from scratch.

2. **Instruction tuning refines, not creates**: In larger models, instruction tuning's role appears to be refining and strengthening existing value representations rather than building them de novo.

3. **Implications for frontier models**: If this trend continues, the largest models (70B+) may have even more developed constitutional reasoning before instruction tuning, with instruction tuning providing only marginal geometric improvements.

4. **Original findings remain valid**: The 3B results are not invalidated — they accurately capture instruction tuning's effect at that scale. The insight is that this effect is scale-dependent.

---

### Comparison to Criminal Planning Experiment

Our parallel criminal planning experiment with Llama-3.1-8B showed similarly modest instruction-tuning effects:

| Domain | Best Base R² | Best IT R² | IT Δ |
|--------|-------------|-----------------|--------|
| SCOTUS (8B) | +0.24 | +0.41 | +0.18 |
| Criminal Planning (8B) | +0.50 | +0.52 | +0.02 |

Both experiments converge on the same conclusion: at 8B scale, base models already encode ethical/legal reasoning structure.

---

### Output Artifacts

| File | Description |
|------|-------------|
| `results/llama31_8b/probe_comparison.json` | Full layer-by-layer R² for 8B models |
| `results/llama31_8b/layer_comparison.png` | Visualization for 8B models |
| `results/llama31_8b/activations/` | Cached 8B activations (32 layers × 4096 dims) |

---

## Phase 4: Cross-Model Validation (Mistral-7B & Qwen2.5-7B)

**Date**: 2025-12-16
**Status**: Complete - Cross-model validation confirms findings, reveals Qwen divergence

### Motivation

Phases 1-3 used only Llama family models. To test whether constitutional geometry findings generalize across model architectures, we ran the same experiment on Mistral-7B and Qwen2.5-7B (both trained by different organizations with different data/methods).

### Experiment Configuration

| Parameter | Mistral-7B | Qwen2.5-7B |
|-----------|------------|------------|
| **Base Model** | mistralai/Mistral-7B-v0.1 | Qwen/Qwen2.5-7B |
| **IT Model** | mistralai/Mistral-7B-Instruct-v0.1 | Qwen/Qwen2.5-7B-Instruct |
| **Architecture** | 32 layers | 28 layers |
| **Sample Size** | 49 SCOTUS cases | 49 SCOTUS cases |
| **Execution** | RunPod GPU instance | RunPod GPU instance |

---

### Key Finding 1: Mistral Confirms Llama Pattern

Mistral-7B shows nearly identical results to Llama-3.1-8B:

| Metric | Llama 8B | Mistral 7B |
|--------|----------|------------|
| Best Base R² | +0.24 (L30) | +0.26 (L15) |
| Best IT R² | +0.41 (L12) | +0.40 (L26) |
| IT Δ | +0.18 | +0.14 |

**Interpretation**: Constitutional principle encoding generalizes across Western-trained models. The ~0.40 IT R² appears to be a consistent ceiling.

---

### Key Finding 2: Qwen Shows Significantly Weaker Signal

Qwen2.5-7B shows dramatically different results:

| Metric | Llama/Mistral | Qwen |
|--------|---------------|------|
| Best Base R² | +0.24 to +0.26 | **-0.14** |
| Best IT R² | +0.40 to +0.41 | **+0.23** |
| IT Δ | +0.14 to +0.18 | **+0.37** |

**Key observations**:
- Qwen base model shows **negative** R² (worse than random), unlike Llama/Mistral base models
- Qwen IT model achieves only **half the R²** of other IT models (0.23 vs ~0.40)
- Qwen shows the **largest IT improvement** (+0.37) despite weaker absolute performance

---

### Key Finding 3: Layer Localization Varies Significantly

| Model | Best Base Layer | Best IT Layer | % Through Network |
|-------|----------------|-------------------|-------------------|
| Llama 8B | 30/32 | 12/32 | 94% → 38% |
| Mistral 7B | 15/32 | 26/32 | 47% → 81% |
| Qwen 7B | 3/28 | 16/28 | 11% → 57% |

**Observations**:
- Llama IT model peaks **early** (38%), base peaks late (94%)
- Mistral shows opposite pattern: base peaks mid (47%), IT peaks **late** (81%)
- Qwen peaks early for base (11%), mid for IT (57%)

This suggests different architectures encode constitutional concepts at different processing stages.

---

### Interpretation of Qwen Divergence

Qwen's weaker constitutional signal may reflect:

1. **Training data differences**: Qwen was trained primarily on Chinese text, which may include less US constitutional law content than Western models' training data.

2. **Cultural encoding**: Constitutional concepts like "due process" and "federalism" may be more culturally specific than expected, requiring substantial exposure to US legal text to develop linearly-separable representations.

3. **Different safety mechanisms**: Qwen's instruction tuning may achieve safety through different mechanisms than geometric restructuring (e.g., output filtering, attention masking).

4. **Not architectural**: Mistral and Llama have different architectures but show similar results, suggesting the Qwen divergence is **not** primarily architectural.

**Implication**: Constitutional geometry may be a property of **training data and cultural context**, not just model scale or instruction-tuning methodology.

---

### Output Artifacts

| File | Description |
|------|-------------|
| `results/mistral_7b/probe_comparison.json` | Full layer-by-layer R² for Mistral |
| `results/mistral_7b/layer_comparison.png` | Visualization for Mistral |
| `results/qwen25_7b/probe_comparison.json` | Full layer-by-layer R² for Qwen |
| `results/qwen25_7b/layer_comparison.png` | Visualization for Qwen |

---

## Phase 5: Large Scale Validation (Qwen2.5-32B & Gemma 2-27B)

**Date**: 2025-12-17
**Status**: Complete - Scale does NOT create alignment geometry

### Motivation

After Phase 4 revealed the 8B base model already encodes constitutional principles (R²=+0.24), we hypothesized that larger models might develop even stronger base representations. If true, this would suggest alignment geometry emerges from scale, not instruction tuning. We tested this with two large models: Qwen2.5-32B (64 layers) and Gemma 2-27B (46 layers).

### Experiment Configuration

| Parameter | Qwen2.5-32B | Gemma 2-27B |
|-----------|-------------|-------------|
| **Base Model** | Qwen/Qwen2.5-32B | google/gemma-2-27b |
| **IT Model** | Qwen/Qwen2.5-32B-Instruct | google/gemma-2-27b-it |
| **Architecture** | 64 layers | 46 layers |
| **Sample Size** | 49 SCOTUS cases | 49 SCOTUS cases |
| **Precision** | bfloat16 | bfloat16 |
| **Execution** | RunPod A100 80GB | RunPod A100 80GB |

---

### Key Finding 1: Scale Does NOT Create Alignment Geometry

**Critical result**: Gemma 2-27B (27B parameters) shows near-zero base structure, matching the much smaller Llama 3.2-3B (3B parameters):

| Model | Scale | Base R² | IT R² | IT Δ |
|-------|-------|---------|------------|--------|
| Llama 3.2-3B | 3B | -0.24 | +0.49 | +0.73 |
| **Gemma 2-27B** | **27B** | **+0.04** | **+0.48** | **+0.43** |

Despite a **9x scale difference**:
- Both base models have essentially no linear structure
- Both IT models achieve ~0.48 R² (nearly identical!)

**This suggests conceptual structure does not emerge from scale alone.**

---

### Key Finding 2: Qwen Remains Weak at Scale

Qwen2.5-32B shows similar patterns to Qwen2.5-7B — weaker overall signal:

| Metric | Qwen 7B | Qwen 32B |
|--------|---------|----------|
| Best Base R² | -0.14 | +0.06 |
| Best IT R² | +0.23 | +0.21 |
| IT Δ | +0.37 | +0.14 |

The 32B model shows slightly better base performance (0.06 vs -0.14) but IT performance is nearly identical (0.21 vs 0.23). This suggests Qwen's weak constitutional signal is consistent across scales.

---

### Key Finding 3: Gemma Matches Western 7B Models

Despite being a 27B model, Gemma's IT performance (R²=0.48) matches the smaller Western models:

| Model | Scale | IT R² |
|-------|-------|------------|
| Llama 3.2-3B | 3B | 0.49 |
| Llama 3.1-8B | 8B | 0.41 |
| Mistral-7B | 7B | 0.40 |
| **Gemma 2-27B** | **27B** | **0.48** |
| Qwen2.5-7B | 7B | 0.23 |
| Qwen2.5-32B | 32B | 0.21 |

**Interpretation**: There appears to be a ~0.40-0.50 ceiling for linear probing of constitutional principles in Western-trained models. Scale beyond 8B doesn't improve this ceiling — instruction tuning achieves it regardless of starting point.

---

### Key Finding 4: Layer Localization

| Model | Best Base Layer | Best IT Layer | % Through |
|-------|----------------|-------------------|-----------|
| Qwen 32B | 29/64 | 49/64 | 45% → 77% |
| Gemma 27B | 11/46 | 23/46 | 24% → 50% |

Gemma aligns mid-network (50%), while Qwen aligns late (77%). Both show the IT peak occurring later than the base peak.

---

### Per-Principle Analysis (Gemma 2-27B)

At best IT layer (23):

| Principle | R² |
|-----------|-----|
| Federalism | 0.65 |
| Free Expression | 0.61 |
| Equal Protection | 0.62 |
| Privacy/Liberty | 0.56 |
| Due Process | 0.12 |

**Due Process remains hardest to predict** — consistent with other models.

---

### Interpretation

1. **Conceptual structure doesn't emerge from scale alone**: The Gemma 2-27B base model (27B params) shows near-zero constitutional structure, yet after instruction tuning achieves the same ~0.48 R² as the 3B IT model. This suggests post-training may be needed for conceptual emergence in many cases.

2. **Base model structure varies by family, not scale**: Llama 3.1-8B shows positive base R² (0.24), but this doesn't generalize — Gemma 2-27B at 3x the scale shows weaker base structure.

3. **Western vs Chinese training persists at scale**: Qwen's weak signal at 32B suggests this reflects training data/cultural differences rather than scale.

4. **Apparent ceiling effect**: Multiple Western models converge at ~0.40-0.50 R², which may represent a limit of linear probing for these concepts.

---

### Updated Summary Table

| Model | Scale | Base R² | IT R² | IT Δ |
|-------|-------|---------|------------|--------|
| Llama 3.2-3B | 3B | -0.24 | +0.49 | +0.73 |
| Qwen2.5-7B | 7B | -0.14 | +0.23 | +0.37 |
| Mistral-7B | 7B | +0.26 | +0.40 | +0.14 |
| Llama 3.1-8B | 8B | +0.24 | +0.41 | +0.18 |
| **Gemma 2-27B** | **27B** | **+0.04** | **+0.48** | **+0.43** |
| Qwen2.5-32B | 32B | +0.06 | +0.21 | +0.14 |

---

### Output Artifacts

| File | Description |
|------|-------------|
| `results/qwen25_32b/probe_comparison.json` | Full layer-by-layer R² for Qwen 32B |
| `results/qwen25_32b/layer_comparison.png` | Visualization for Qwen 32B |
| `results/gemma2_27b/probe_comparison.json` | Full layer-by-layer R² for Gemma 27B |
| `results/gemma2_27b/layer_comparison.png` | Visualization for Gemma 27B |

---

## Phase 6: Robustness Validation

**Date**: 2026-02-26
**Status**: Complete — core findings confirmed, new metrics strengthen claims

### Motivation

An internal methodology review identified three concerns:
1. Cross-validated R² in the n << p regime (49 samples, 3072–5120 features) is sensitive to numerical backend (BLAS/LAPACK), producing absolute value shifts of ~0.3–0.6 across compute environments
2. The 5-dimensional projection R² (project onto probe directions, then OLS in 5D) is circular — with d >> n, Ridge can always find 5 directions that correlate with any 5 targets, so R² > 0.99 for base models has zero discriminative power
3. The transfer test (IT probe → base activations) needed cross-model controls to rule out a model-specificity confound

### Validation 1: Permutation Tests (200 permutations per model)

For each model family, we shuffled principle labels across cases (breaking the case↔principle correspondence) 200 times and re-ran the full probe pipeline. If the real R² significantly exceeds the shuffled distribution, the signal is genuine regardless of absolute R² magnitude.

| Model | Layer | IT p-value | Base p-value | Interpretation |
|-------|-------|-----------------|--------------|----------------|
| Gemma 2-27B | 23 | **p=0.000** | p=0.745 | IT signal confirmed |
| Llama 3.1-8B | 12 | **p=0.005** | p=0.240 | IT signal confirmed |
| Mistral-7B | 26 | **p=0.000** | p=0.240 | IT signal confirmed |
| Qwen 2.5-7B | 16 | **p=0.005** | p=0.645 | IT signal confirmed |
| Qwen 2.5-32B | 49 | p=0.200 | p=0.415 | Not significant |

**Result**: 4/5 IT models show signal significantly above the shuffled null (p ≤ 0.005). No base model reaches significance. This confirms the IT >> base pattern is statistically real, independent of absolute R² values.

Note: the shuffled null distribution has mean R² around -2.7 (deeply negative), confirming that even modest real R² values (e.g., -0.09 for Mistral IT) represent genuine above-chance signal in this regime.

**Output**: `results/*/permutation_validation.json` | **Script**: `validation/run_permutation_validation.py`

### Validation 2: Probe Direction Cosine Similarity

Does instruction tuning amplify directions already present in base models, or create genuinely new structure? We trained Ridge probes independently on base and IT activations, extracted the 5 × d_model weight matrices, and computed per-principle cosine similarity. A permutation null (100 label shuffles) established the chance baseline.

| Model | Layer | Mean cos(base, IT) | Null mean | p-value |
|-------|-------|------------------------|-----------|---------|
| Gemma 2-27B | 23 | 0.274 | 0.262 | 0.260 |
| Llama 3.1-8B | 12 | 0.303 | 0.310 | 0.680 |
| Mistral-7B | 26 | 0.173 | 0.194 | 0.940 |
| Qwen 2.5-7B | 16 | 0.573 | 0.568 | 0.350 |
| Qwen 2.5-32B | 49 | 0.451 | 0.408 | **0.010** |

**Result**: For 4/5 models, base-IT cosine similarity is indistinguishable from shuffled labels. The positive cosine values (0.17–0.57) reflect ambient high-dimensional geometry, not principle-specific direction sharing. **Instruction tuning reorganizes representational geometry rather than amplifying existing directions.** Qwen 2.5-32B is the exception (p=0.010), suggesting its base model may share some directional structure with the IT model at layer 49.

**Output**: `results/*/probe_direction_similarity.json` | **Script**: `validation/probe_direction_analysis.py`

### Validation 3: Cross-Model Transfer Controls

The existing within-model transfer test (IT probe → base activations) shows moderate correlation (mean r = 0.50–0.83). To test whether this reflects instruction-tuning-specific structure or model-family-specific activation geometry, we performed cross-model transfers between models with matching d_model (Llama 3.1-8B and Mistral-7B, both d=4096).

| Transfer Type | Mean Pearson r |
|---------------|---------------|
| **Within-model IT→base** | |
| Llama IT → Llama base | **+0.83** |
| Mistral IT → Mistral base | **+0.50** |
| **Cross-model IT→IT** | |
| Llama IT → Mistral IT | +0.07 |
| Mistral IT → Llama IT | +0.08 |
| **Cross-model base→base** | |
| Llama base → Mistral base | -0.02 |
| Mistral base → Llama base | -0.15 |

**Result**: Cross-model transfer is near zero in all conditions (~0.07 for IT→IT). This means:
1. The within-model IT→base transfer (0.50–0.83) is a genuine within-family signal, not an artifact of activation statistics
2. Constitutional representations are **model-family-specific** — each architecture develops its own encoding during instruction tuning
3. There is no universal "instruction-tuning direction" shared across model families

**Limitation**: Only Llama 3.1-8B and Mistral-7B share d_model=4096. Other model pairs have incompatible dimensionalities and were not compared.

**Output**: `validation/cross_model_transfer_controls.json` | **Script**: `validation/cross_model_transfer_controls.py`

### Validation 4: R² Stability Across Environments

We re-ran the full probe pipeline on the current local environment (sklearn 1.7.2) and compared layer-by-layer R² curves against the original RunPod-generated values.

| Model | d_model | IT curve Pearson r | Mean &#124;shift&#124; | Assessment |
|-------|---------|------------------------|-----------------|------------|
| **Llama 3.2-3B** | 3072 | **1.000** | 0.000 | Perfectly reproducible |
| Llama 3.1-8B | 4096 | 0.923 | 0.358 | Shape preserved |
| Mistral-7B | 4096 | 0.886 | 0.506 | Shape preserved |
| Gemma 2-27B | 4608 | 0.956 | 0.628 | Shape preserved |
| Qwen 2.5-7B | 3584 | 0.358 | 0.516 | Shape less stable |

**Result**: The Llama 3.2-3B values are perfectly reproducible (generated locally with the same sklearn version). RunPod-generated models show IT curve *shapes* that are well-preserved (r = 0.89–0.96 for Gemma, Llama 8B, Mistral) despite absolute value shifts. The instability reflects differences in BLAS/LAPACK numerical precision between environments, not methodological flaws — sklearn explicitly warns `Ill-conditioned matrix` for the larger d_model models.

### Revised Defensible Claims

Based on all validation work, the following claims are supported:

1. **Instruction tuning creates statistically significant linear structure** encoding constitutional principles. Permutation testing confirms this in 4/5 model families (p ≤ 0.005), while no base model shows significant structure at the same layers.

2. **Instruction tuning reorganizes geometry rather than amplifying existing directions**. Probe weight cosine similarity between base and IT models is indistinguishable from a label-shuffled null in 4/5 cases.

3. **Constitutional representations are model-family-specific**. Cross-model transfer yields near-zero correlation, while within-model IT→base transfer is moderate to strong (r = 0.50–0.83).

### What We Discarded

- **5-dim projection R²**: Circular metric. With d >> n, Ridge always finds 5 directions that correlate with targets. R² > 0.99 for base models confirms it measures optimization quality, not signal presence. All 5-dim projection results should be ignored.

### Open Questions (at time of Phase 6)

- **Sample size**: 49 SCOTUS cases places inherent limits on statistical power and drives the n << p instability. A larger corpus would both stabilize R² estimates and enable held-out test sets.
- ~~**Causal link**: We detect linear structure in activations, but have not established that this structure causally drives downstream constitutional reasoning behavior.~~ → **Resolved in Phases 8-9** (patching, steering, and mechanistic decomposition).
- **Qwen divergence**: Weaker signal in Qwen models (7B and 32B) may reflect training data composition (Chinese vs Western corpora) rather than architectural factors — this remains untested.

### Output Artifacts

| File | Description |
|------|-------------|
| `results/*/permutation_validation.json` | 200-permutation test results for all 5 model families |
| `results/*/probe_direction_similarity.json` | Base-IT probe cosine similarity with null distribution |
| `results/cross_model/cross_model_transfer_controls.json` | Cross-model transfer results (Llama↔Mistral) |
| `results/cross_model/validation_report.md` | Full methodology review and consolidated findings |
| `validation/run_permutation_validation.py` | Permutation test script |
| `validation/probe_direction_analysis.py` | Probe direction cosine similarity script |
| `validation/cross_model_transfer_controls.py` | Cross-model transfer control script |
| `results/cross_model/r2_methodology_review.md` | Notes on R² instability that motivated this phase |

---

## Phase 7: Behavioral Verification

**Date**: 2025-12-19 to 2025-12-21
**Status**: Complete — IT models dramatically outperform base on constitutional reasoning, with one notable exception

### Motivation

Probing results (Phases 1-6) demonstrate that IT models develop more linearly separable representations of constitutional principles. But do these geometric differences correspond to *behavioral* differences? We tested whether IT models actually identify the correct primary constitutional principle for each case better than base models.

### Method

For each of 49 SCOTUS cases, we prompted both base and IT models to rank 5 constitutional principles by relevance. We scored the top-ranked principle against ground truth annotations using two thresholds: exact match (annotation weight ≥ 0.8) and partial match (weight ≥ 0.3).

### Results

| Model | Base Exact | IT Exact | Base Weighted | IT Weighted | Δ (weighted) |
|-------|:----------:|:--------:|:-------------:|:-----------:|:------------:|
| **Gemma 2-27B** | 28/49 (57.1%) | 45/49 (91.8%) | 58.2% | **95.9%** | **+37.8pp** |
| **Mistral-7B** | 7/49 (14.3%) | 43/49 (87.8%) | 14.3% | **91.8%** | **+77.6pp** |
| **Llama 3.2-3B** | 20/49 (40.8%) | 36/49 (73.5%) | — | — | **+32.7pp** |
| **Qwen 2.5-7B** | 42/49 (85.7%) | 43/49 (87.8%) | 89.8% | **92.9%** | **+3.1pp** |

*Weighted score: exact=1.0, partial=0.5. Llama 3.2-3B was evaluated with exact match only.*

### Key Observations

1. **Base model capability varies dramatically**: Mistral base produces BrainMass SEO spam instead of legal analysis (14.3% accuracy). Gemma base outputs `<eos>` for many cases (57.1%). Llama-3B base defaults to "Free Expression" for most cases (40.8%). Yet Qwen base produces coherent legal analysis at 85.7% accuracy.

2. **IT creates behavioral capability from nothing**: The largest behavioral improvements come from models whose base produces no useful output — Mistral (+77.6pp), Gemma (+37.8pp), Llama-3B (+32.7pp).

3. **Qwen exception**: Qwen base already performs near-ceiling (85.7% → 87.8%), consistent with the probing finding that Qwen's constitutional signal may originate in pre-training rather than instruction tuning.

4. **Privacy/Liberty is systematically hard**: Llama-3B-Instruct misclassifies all 10 privacy/liberty cases as "Due Process" — it has learned to reason about constitutional concepts but conflates these two closely-related principles. Gemma IT handles privacy/liberty correctly in most cases.

### Output Artifacts

| File | Description |
|------|-------------|
| `results/llama32_3b/behavioral_summary.md` | Full case-by-case results for Llama 3.2-3B |
| `results/gemma2_27b/behavioral_summary.md` | Full case-by-case results for Gemma 2-27B |
| `results/mistral_7b/behavioral_summary.md` | Full case-by-case results for Mistral-7B |
| `results/qwen25_7b/behavioral_summary.md` | Full case-by-case results for Qwen 2.5-7B |
| `results/*/behavioral_responses.json` | Raw model responses |

---

## Phase 8: Causal Validation (Activation Patching & Steering)

**Date**: 2025-12-21 to 2026-03-03
**Status**: Complete — probe geometry is partially causal, with strong architecture dependence

### Motivation

Phases 1-7 established that IT models encode constitutional principles in linearly separable geometry and that this correlates with superior behavioral performance. But correlation is not causation. We tested two causal interventions: (1) **activation patching** — transplanting IT activations into the base model to test if geometry is sufficient for behavior, and (2) **activation steering** — adding probe-derived direction vectors during generation to test if specific directions control specific principles.

### Activation Patching Results

We replaced the base model's residual stream activations with IT model activations at specific layers during inference:

| Model | Size | Layers Patched | Base Accuracy | IT Accuracy | Patched Accuracy | Recovery |
|-------|:----:|:--------------:|:-------------:|:-----------:|:----------------:|:--------:|
| **Gemma-2** | 27B | 20-34 | 8.3% | 83.3% | **83.3%** | **100%** |
| Llama-3.2 | 3B | 14-24 | 0.0% | 83.3% | 0.0% | 0% |
| Llama-3.1 | 8B | 16-28 | 0.0% | 83.3% | 0.0% | 0% |
| Mistral | 7B | 16-28 | 8.3% | 58.3% | 8.3% | 0% |
| Qwen-2.5 | 7B | 16-28 | 75.0% | 58.3% | 83.3% | N/A* |

*Qwen base outperforms its IT model on this task, inverting the expected pattern.

**Key finding**: Activation patching is highly architecture-dependent. Only Gemma 2-27B shows successful transfer — wholesale replacement of activations at layers 20-34 recovers full IT behavior. Llama models break (EOS tokens), Mistral ignores the patch, Qwen shows mild improvement on an already-capable base.

### Activation Steering Results (Gemma 2-27B)

Five rounds of steering experiments revealed that **calibrating to residual stream norm** is critical. The mean ‖resid‖ at layer 23 is 19,030 — earlier experiments at alpha=±3 were operating at <0.02% of this norm.

**Norm-relative steering at alpha=±0.1 (10% of ‖resid‖):**

| Outcome (alpha=+0.1) | Count | Rate |
|-----------------------|:-----:|:----:|
| **Appeared** (absent → ranked) | 11/25 | 44% |
| Unchanged | 10/25 | 40% |
| Worsened | 4/25 | 16% |

| Outcome (alpha=-0.1) | Count | Rate |
|-----------------------|:-----:|:----:|
| **Suppressed** (rank dropped/vanished) | 5/25 | 20% |
| Unchanged | 13/25 | 52% |
| Boosted (wrong direction) | 7/25 | 28% |

**The coherence cliff**: A narrow usable window exists between "too small to observe" and "destroys generation":

| Alpha | % of ‖resid‖ | Parseable Responses |
|:-----:|:---:|:-------------------:|
| ±0.1 | 10% | 24/25 (96%) — coherent but altered |
| ±0.5 | 50% | 0/25 (0%) — multilingual gibberish |
| ±1.0 | 100% | 0/25 (0%) — complete collapse |

At alpha ≥ ±0.5, each probe direction activates a consistent, direction-specific failure pattern regardless of input case (e.g., free expression → French/Malay fragments; equal protection → repeated German "höher"; due process → Finnish/Swedish + hex tokens).

### Interpretation

The probe directions are **partially causal** rather than purely epiphenomenal:
- They can pull principles *into* or *out of* the response (44% appearance, 20% suppression)
- But they cannot override the model's strong prior about which principles are most relevant (no rank promotions above baseline)
- This suggests the directions encode *salience* (whether a principle is mentioned) more than *ranking* (how prominently it features)

Previous null results (Rounds 1-4) were a **scaling artifact** — perturbations at <3% of residual norm are simply invisible to the model.

### Output Artifacts

| File | Description |
|------|-------------|
| `results/gemma2_27b/CAUSAL_VALIDATION_SUMMARY.md` | Full causal validation narrative |
| `results/gemma2_27b/patching/` | Patching results (round 1 and round 2) for all 5 models |
| `results/gemma2_27b/steering_norm_relative/` | Norm-relative steering results and examples |
| `results/gemma2_27b/steering_*/` | Earlier steering rounds (standard, large, medium, all-positions) |

---

## Phase 9: Mechanistic Decomposition (Gemma 2-27B)

**Date**: 2026-02-25
**Status**: Complete — attention layers 20-22 dominate; conceptual synthesis, not keyword matching

### Motivation

Probing and steering identify *where* constitutional geometry lives (layers 20-23 for Gemma 2-27B) and confirm it is partially causal. But *how* is this geometry computed? We decomposed the residual stream at layer 23 into per-component contributions (attention and MLP at each layer, individual attention heads) to identify which components write the most principle-discriminative signal.

### Method

Using the sandwich-norm-corrected residual stream decomposition (see gotchas.md — Gemma 2 architecture requires computing contributions as residual differences, not raw hook outputs), we projected each component's contribution onto probe directions and computed Pearson r with ground-truth principle weights across the 49 cases.

### Phase 1: Attention vs MLP Attribution

| Rank | Component | Mean |r| | Top Principle r |
|:----:|-----------|:-------:|:---------------:|
| 1 | **attn_22** | **0.882** | Federalism: 0.952 |
| 2 | **attn_21** | **0.865** | Equal Protection: 0.907 |
| 3 | **attn_20** | **0.779** | Free Expression: 0.869 |
| 4 | mlp_21 | 0.619 | Federalism: 0.771 |
| 5 | attn_18 | 0.619 | Free Expression: 0.729 |

**Attention dominates**: Mean discriminative |r| across all layers is 0.376 for attention vs 0.280 for MLP. The top 3 components are all attention layers (20-22). In the top 10 discriminative components: 8 attention, 2 MLP.

### Phase 2: Head-Level Attribution

No single "specialist head" — signal is distributed across ~20 heads in layers 17-22:

| Rank | Head | Top Principle | Mean |r| |
|:----:|------|---------------|:-------:|
| 1 | L22H26 | privacy_liberty | 0.533 |
| 2 | L20H15 | free_expression | 0.527 |
| 3 | L22H25 | free_expression | 0.519 |
| 4 | L21H28 | free_expression | 0.474 |
| 5 | L22H19 | privacy_liberty | 0.469 |

Layer 22 contributes 7 of the top 20 discriminative heads. The best individual head (L22H26, |r|=0.533) accounts for only ~60% of the full layer's discriminative power (attn_22, |r|=0.882), confirming the distributed nature.

### Phase 3: Attention Pattern Analysis

**Key finding**: Specialist heads are **not keyword-matching**. Across all 10 top heads, principle-name tokens receive **<1% of total attention**:

| Category | Mean Attention | Range |
|----------|:-------------:|:-----:|
| Case facts ("other") | 51.8% | 34-72% |
| Legal terms | 20.7% | 1-58% |
| Punctuation | 12.9% | 2-40% |
| Whitespace | 9.8% | 0-47% |
| **Principle names** | **0.5%** | **0-5%** |

A head whose top discriminative principle is "privacy_liberty" (e.g., L22H26) places zero attention on tokens like "privacy" or "liberty." Instead, it attends to case facts and legal context and its output vector lands in a region of residual stream space that the probe reads as encoding that principle.

### Interpretation

Constitutional principle encoding emerges from **conceptual synthesis** — the model computes principle-relevant representations by aggregating information from across the factual and legal context of each case, not by detecting principle-related keywords. The principle encoding is an emergent property of *how* attention heads combine diverse contextual information, not *which specific tokens* they attend to.

This is consistent with the steering results (Phase 8): the narrow usable window and direction-specific gibberish attractors suggest the probe directions point to real functional regions of activation space, not superficial token-level correlations.

### Output Artifacts

| File | Description |
|------|-------------|
| `results/gemma2_27b/decomposition/decomposition_summary.md` | Full discriminative attribution tables |
| `results/gemma2_27b/decomposition/discriminative_attribution.json` | Per-component Pearson r values |
| `results/gemma2_27b/decomposition/phase2_head_attribution.json` | Per-head results for layers 17-22 |
| `results/gemma2_27b/decomposition/phase3_attention_patterns.json` | Attention distribution by token category |

---

## Changelog

| Date | Update |
|------|--------|
| 2025-11-26 | Initial PoC experiment with Llama 3.2-3B |
| 2025-11-28 | Documentation, permutation test, Sonnet validation, full 28-case re-run |
| 2025-11-28 | **Phase 2**: Added 21 cases (49 total), restructured to JSON, R² improved to 0.50 |
| 2025-11-28 | Sonnet cross-validation of all 21 Phase 2 annotations (90% accurate/minor_issues) |
| 2025-11-28 | **Data corrections**: Fixed Hustler opinion (wrong case fetched), adjusted Bakke weights; R² 0.50→0.49 |
| 2025-11-28 | **Interpretation correction**: Fixed inaccurate "anti-correlated" language; negative R² indicates absence of linear structure, not anti-correlation |
| 2025-12-11 | **Phase 3**: Llama-3.1-8B replication reveals model size effect — 8B base model already encodes constitutional principles (R² = +0.24 vs -0.25 for 3B), IT Δ drops from +0.74 to +0.18 |
| 2025-12-11 | Updated summary to reflect model size findings; original 3B results remain valid but are now understood as scale-dependent |
| 2025-12-16 | **Phase 4**: Cross-model validation with Mistral-7B and Qwen2.5-7B |
| 2025-12-16 | Mistral confirms Llama pattern (base +0.26, IT +0.40, Δ +0.14) |
| 2025-12-16 | **Qwen divergence discovered**: Weaker signal (IT R² = 0.23 vs ~0.40 for others), possibly due to training data/cultural differences |
| 2025-12-16 | Updated summary and limitations to reflect cross-model findings |
| 2025-12-17 | **Phase 5**: Large scale validation with Qwen2.5-32B and Gemma 2-27B |
| 2025-12-17 | **Critical finding**: Scale does NOT create alignment geometry — Gemma 27B base (R²=0.04) matches Llama 3B base (-0.24), both IT reach ~0.48 |
| 2025-12-17 | Qwen weak signal confirmed at scale (32B IT R²=0.21, similar to 7B) |
| 2025-12-17 | Updated interpretation: Instruction tuning explicitly creates constitutional geometry rather than refining emergent representations |
| 2026-02-26 | **Phase 6**: Robustness validation — permutation tests (200 shuffles, 5 families), probe direction cosine similarity, cross-model transfer controls |
| 2026-02-26 | Added permutation p-value column to Key Metrics table; 4/5 IT models significant at p ≤ 0.005, 0/5 base models significant |
| 2026-02-26 | Confirmed Llama 3.2-3B R² values perfectly reproducible; RunPod models show stable curve shapes with absolute value shifts |
| 2026-02-26 | Discarded 5-dim projection R² as circular; added R² sensitivity note to Limitations |
| 2025-12-19 | **Phase 7**: Behavioral verification across 4 models (Gemma, Mistral, Qwen, Llama-3B); IT dramatically outperforms base except Qwen |
| 2025-12-21 | **Phase 8**: Causal validation — activation patching (100% recovery for Gemma 2-27B, 0% for others) and steering experiments (5 rounds) |
| 2026-03-03 | **Phase 8 cont'd**: Norm-relative steering breakthrough — 44% appearance rate at 10% of ‖resid‖, coherence cliff at 50% |
| 2026-02-25 | **Phase 9**: Mechanistic decomposition — attn layers 20-22 dominate, distributed across ~20 heads, conceptual synthesis not keyword matching |
| 2026-03-25 | **Consolidated**: Moved from `results/llama32_3b/` to `results/EXPERIMENT_OUTCOMES.md`; added Phases 7-9; updated summary |
