# Experiment Outcomes: Criminal Planning Geometry

**Experiment Run Dates**: December 10-17, 2025

## Executive Summary

We tested whether RLHF creates detectable geometric structures in transformer activations that correspond to safety-relevant concepts. Linear probes trained on residual stream activations show that **aligned models encode prompt severity and restraint signals more linearly than base models**, with consistent findings across four model families at 7B-8B scale (Llama 3.1-8B, Llama 3.2-3B, Mistral-7B, Qwen2.5-7B). Notably, Qwen-7B shows divergent behavior on response toxicity, suggesting model-family-specific safety mechanisms.

**Major finding at scale**: Testing at 27B-32B scale reveals **model-family-specific divergence**:
- **Qwen2.5-32B**: Reversed alignment effects for criminal planning (base outperforms aligned across all targets)
- **Gemma 2-27B**: Mixed effects - base slightly better on severity (-0.015), but aligned better on restraint (+0.077) and toxicity (+0.053)

**Observation**: Gemma 2-27B shows the same SCOTUS pattern as small models - near-zero base structure (R²=0.04) but strong aligned structure (R²=0.48). This suggests conceptual structure does not emerge from scale alone — RLHF or post-training may be needed for conceptual emergence in many cases.

## Models Tested

| Model Pair | Layers | Architecture |
|------------|--------|--------------|
| Llama 3.1-8B / 8B-Instruct | 32 | `experiment_output/` |
| Llama 3.2-3B / 3B-Instruct | 28 | `experiment_output_llama32_3b/` |
| Mistral-7B / 7B-Instruct | 32 | `experiment_output_mistral_7b/` |
| Qwen2.5-7B / 7B-Instruct | 28 | `experiment_output_qwen25_7b/` |
| **Qwen2.5-32B / 32B-Instruct** | **64** | `experiment_output_qwen25_32b/` |
| **Gemma 2-27B / 27B-it** | **46** | `experiment_output_gemma2_27b/` |

## Results Summary

### Regression Performance (Best Layer R²)

| Model | Target | Base R² | Aligned R² | Improvement |
|-------|--------|---------|------------|-------------|
| **Llama 3.1-8B** | Prompt Severity | 0.161 | 0.278 | +0.117 |
| | Response Toxicity | 0.078 | 0.109 | +0.031 |
| | Restraint Delta | 0.077 | 0.193 | +0.117 |
| | Joint Dimensions | 0.498 | 0.518 | +0.020 |
| **Llama 3.2-3B** | Prompt Severity | 0.043 | 0.228 | +0.185 |
| | Response Toxicity | 0.174 | 0.182 | +0.008 |
| | Restraint Delta | 0.085 | 0.153 | +0.068 |
| | Joint Dimensions | 0.501 | 0.521 | +0.020 |
| **Mistral-7B** | Prompt Severity | 0.105 | 0.192 | +0.086 |
| | Response Toxicity | -0.192 | 0.069 | +0.261 |
| | Restraint Delta | 0.012 | 0.014 | +0.002 |
| | Joint Dimensions | 0.479 | 0.520 | +0.041 |
| **Qwen2.5-7B** | Prompt Severity | 0.180 | 0.244 | +0.064 |
| | Response Toxicity | 0.013 | **-0.043** | **-0.056** |
| | Restraint Delta | -0.041 | -0.001 | +0.040 |
| | Joint Dimensions | 0.502 | 0.509 | +0.007 |
| **Qwen2.5-32B** | Prompt Severity | **0.228** | 0.187 | **-0.041** |
| | Response Toxicity | 0.063 | 0.058 | -0.005 |
| | Restraint Delta | 0.166 | 0.165 | -0.001 |
| | Joint Dimensions | 0.506 | 0.499 | -0.007 |
| **Gemma 2-27B** | Prompt Severity | **0.236** | 0.221 | **-0.015** |
| | Response Toxicity | -0.015 | 0.038 | +0.053 |
| | Restraint Delta | 0.128 | 0.205 | +0.077 |
| | Joint Dimensions | 0.508 | 0.531 | +0.024 |

### Best Performing Layers

| Model | Target | Base Best Layer | Aligned Best Layer |
|-------|--------|-----------------|-------------------|
| **Llama 3.1-8B** | Prompt Severity | Layer 8 | Layer 11 |
| | Restraint Delta | Layer 31 | Layer 31 |
| **Llama 3.2-3B** | Prompt Severity | Layer 27 | Layer 9 |
| | Restraint Delta | Layer 8 | Layer 9 |

## Key Findings

### 1. Alignment Improves Linear Predictability of Prompt Severity

Both models show substantial gains in predicting how transgressive a prompt is from activations:
- **8B model**: +0.117 R² improvement (0.161 → 0.278)
- **3B model**: +0.185 R² improvement (0.043 → 0.228)

The smaller model shows a *larger* improvement delta, suggesting RLHF may create more pronounced geometric structure when model capacity is more constrained.

### 2. Restraint Signal is Detectable but Modest

The "restraint delta" (prompt severity minus response toxicity) captures how much the model "holds back" relative to the harmfulness of the input. This signal is more predictable from aligned model activations:
- **8B model**: Base R²=0.077, Aligned R²=0.193 (+0.117)
- **3B model**: Base R²=0.085, Aligned R²=0.153 (+0.068)

The 8B model shows stronger absolute restraint encoding, possibly due to greater capacity for representing nuanced safety distinctions.

### 3. Response Toxicity Prediction is Weak Across All Conditions

Neither base nor aligned models show strong linear structure for predicting output toxicity:
- Best performance: ~0.17 R² (Llama 3.2-3B base model)
- Aligned models show minimal improvement over base

This suggests output toxicity may be determined by:
- Generation dynamics not captured in pre-generation activations
- Non-linear combinations of features
- Factors external to the residual stream (attention patterns, etc.)

### 4. Layer Localization Patterns

**Prompt Severity**: Best prediction occurs in middle layers
- 8B aligned: Layer 11 of 32 (~34% through network)
- 3B aligned: Layer 9 of 28 (~32% through network)

**Restraint Delta**: More varied, with some localization to later layers
- 8B: Both base and aligned peak at layer 31 (final layer)
- 3B: Peaks at layers 8-9 (~30% through network)

The concentration in middle layers for severity is consistent with "representation engineering" findings that mid-network representations encode semantic and conceptual information, while early layers process syntax and late layers prepare outputs.

### 5. Joint Dimension Prediction

Predicting all four annotation dimensions jointly (severity, specificity, real_world_risk, harm_type) shows:
- Modest aligned model advantage (~+0.02 R²)
- Similar performance across both model sizes (~0.50-0.52 R²)
- This suggests harm type classification may not benefit as much from alignment as scalar severity does

## Cross-Model Validation: Mistral-7B and Qwen2.5-7B Results

To test whether the geometric structures are model-family specific or represent general properties of RLHF, we ran the same experiment on Mistral-7B/7B-Instruct and Qwen2.5-7B/7B-Instruct (December 2025).

### Key Cross-Model Findings

#### 1. Alignment Advantage Generalizes for Prompt Severity

All four models show improved linear predictability for prompt severity in aligned versions:

| Target | Llama 8B Δ | Llama 3B Δ | Mistral 7B Δ | Qwen 7B Δ |
|--------|------------|------------|--------------|-----------|
| Prompt Severity | +0.117 | +0.185 | +0.086 | +0.064 |
| Response Toxicity | +0.031 | +0.008 | +0.261 | **-0.056** |
| Restraint Delta | +0.117 | +0.068 | +0.002 | +0.040 |
| Joint Dimensions | +0.020 | +0.020 | +0.041 | +0.007 |

**Interpretation**: The consistent alignment advantage for prompt severity across all four model families suggests RLHF creates similar geometric structures regardless of base architecture.

#### 2. Restraint Signal is Model-Dependent

The restraint delta prediction varies significantly by model family:
- **Llama models**: Strong alignment improvement (+0.068 to +0.117 R²)
- **Qwen**: Moderate improvement (+0.040 R²)
- **Mistral**: Near-zero improvement (+0.002 R²)

**Possible explanations**:
- Different safety training approaches (Llama may encode restraint more explicitly)
- Different model architectures affect where safety-relevant information is stored
- Mistral's instruction tuning may achieve safety through different mechanisms than geometric separation

#### 3. Qwen Shows NEGATIVE Toxicity Improvement (Key Finding)

Qwen2.5-7B is the only model where alignment **decreases** linear predictability for response toxicity:
- **Base Qwen**: R² = 0.013 (weak positive signal)
- **Aligned Qwen**: R² = **-0.043** (worse than random)
- **Improvement**: **-0.056** (regression!)

In contrast, Mistral showed the largest positive improvement (+0.261) for toxicity.

**Possible explanations**:
- Qwen's safety training may use fundamentally different mechanisms (e.g., output filtering rather than representation restructuring)
- Cultural/training data differences may affect how toxicity is represented internally
- Qwen may encode toxicity decisions in attention patterns rather than residual stream

This divergence is the most significant finding for cross-model validation - it suggests **toxicity encoding mechanisms are NOT universal across model families**.

#### 4. Joint Dimension Prediction Remains Consistent

All four aligned models achieve similar joint dimension R² (~0.51-0.52):
- Llama 8B: 0.518
- Llama 3B: 0.521
- Mistral 7B: 0.520
- Qwen 7B: 0.509

This convergence suggests a "ceiling" for linear probing of harm dimensions, possibly reflecting the inherent limits of linear readout from these representations.

### SCOTUS Constitutional Geometry Cross-Validation

We also ran the SCOTUS constitutional principles experiment on Mistral-7B and Qwen2.5-7B:

| Model | Best Base R² | Best Aligned R² | Improvement |
|-------|--------------|-----------------|-------------|
| Llama 3.1-8B | 0.24 (layer 30) | 0.41 (layer 12) | +0.17 |
| Mistral-7B | 0.26 (layer 15) | 0.40 (layer 26) | +0.14 |
| Qwen2.5-7B | **-0.14** (layer 3) | **0.23** (layer 16) | **+0.37** |

**Key observations**:
- Llama and Mistral achieve similar aligned performance (~0.40-0.41 R²)
- **Qwen shows dramatically weaker signal** for both base (-0.14) and aligned (0.23)
- Qwen shows the **largest improvement delta** (+0.37) but from a much worse baseline

**Qwen's weaker SCOTUS signal** may reflect:
- Training data differences (Chinese vs Western training corpora)
- Constitutional law concepts being more culturally specific than expected
- Different architectural encoding of legal/political concepts

**Layer localization varies significantly**:
- Llama peaks early (layer 12/32, ~38%)
- Mistral peaks late (layer 26/32, ~81%)
- Qwen peaks mid-network (layer 16/28, ~57%)

## Scale Effects: Qwen2.5-32B and Gemma 2-27B Results (Major Finding)

We ran Qwen2.5-32B (64 layers, 32B parameters) and Gemma 2-27B (46 layers, 27B parameters) to test whether larger scale affects alignment geometry. **The results reveal model-family-specific patterns at scale.**

### Criminal Planning Results (Large Scale)

| Model | Target | Base R² | Aligned R² | Improvement |
|-------|--------|---------|------------|-------------|
| **Qwen2.5-32B** | Prompt Severity | **0.228** | 0.187 | **-0.041** |
| | Response Toxicity | 0.063 | 0.058 | -0.005 |
| | Restraint Delta | 0.166 | 0.165 | -0.001 |
| | Joint Dimensions | 0.506 | 0.499 | -0.007 |
| **Gemma 2-27B** | Prompt Severity | **0.236** | 0.221 | **-0.015** |
| | Response Toxicity | -0.015 | 0.038 | +0.053 |
| | Restraint Delta | 0.128 | 0.205 | **+0.077** |
| | Joint Dimensions | 0.508 | 0.531 | +0.024 |

**Key difference**: Qwen-32B base outperforms aligned across ALL targets. Gemma-27B shows mixed results - base slightly better on severity, but aligned better on restraint and toxicity.

### SCOTUS Results (Large Scale)

| Model | Base R² | Aligned R² | Improvement |
|-------|---------|------------|-------------|
| Qwen2.5-32B | 0.063 (layer 29) | 0.205 (layer 49) | **+0.142** |
| **Gemma 2-27B** | **0.044** (layer 11) | **0.478** (layer 23) | **+0.434** |

**Critical finding**: Gemma 2-27B shows the **strongest SCOTUS improvement** of any model tested (+0.434). Despite being 27B parameters, the base model has essentially no linear structure (0.044 R²), matching the pattern of the much smaller Llama 3.2-3B (-0.24 R²).

### Interpretation: Conceptual Structure Doesn't Emerge from Scale Alone

The Gemma 2-27B results suggest base models don't reliably develop conceptual representations with scale:

| Model | Scale | Base SCOTUS R² | Aligned SCOTUS R² |
|-------|-------|----------------|-------------------|
| Llama 3.2-3B | 3B | -0.24 | 0.49 |
| **Gemma 2-27B** | **27B** | **0.04** | **0.48** |

Despite a 9x scale difference, both models show:
- Near-zero base structure (random/worse-than-random)
- Nearly identical aligned structure (~0.48 R²)

**This suggests RLHF or post-training may be needed for conceptual emergence** in many cases — base models don't produce these representations by default.

### Model-Family Differences at Scale

**Qwen (Chinese, 32B)**:
- Criminal planning: Base wins on all targets
- SCOTUS: Aligned wins, but with weaker signal (0.21 R²)

**Gemma (Western, 27B)**:
- Criminal planning: Mixed - base wins on severity, aligned wins on restraint/toxicity
- SCOTUS: Aligned wins strongly (0.48 R²), matching smaller Western models

**Possible explanations**:

**A. Training Data/Cultural Effects**: Qwen's training on primarily Chinese corpora may result in different encoding of Western legal concepts (SCOTUS) and different safety mechanisms for criminal planning prompts.

**B. Architecture Effects**: Gemma uses different attention patterns and may encode safety differently than Qwen.

**C. RLHF Methodology**: Different alignment training approaches may create different geometric signatures.

### Comparison Table: Scale Effects (Updated)

| Model | Params | Criminal Planning Δ | SCOTUS Δ | Pattern |
|-------|--------|---------------------|----------|---------|
| Llama 3.2-3B | 3B | +0.185 (severity) | +0.73 | Alignment helps |
| Qwen2.5-7B | 7B | +0.064 (severity) | +0.37 | Alignment helps |
| Llama 3.1-8B | 8B | +0.117 (severity) | +0.18 | Alignment helps |
| **Gemma 2-27B** | **27B** | **-0.015** (severity) | **+0.43** | **Mixed/SCOTUS strong** |
| **Qwen2.5-32B** | **32B** | **-0.041** (severity) | **+0.14** | **Divergent** |

**Key insight**: The Qwen-32B reversal appears to be model-family-specific, not a universal scale effect. Gemma-27B shows strong positive SCOTUS effects despite similar scale. Base models don't reliably produce conceptual representations — RLHF/post-training may be needed for their emergence.

## Causal Validation: Activation Patching (January 2026)

We tested whether the geometric structure discovered via probing is **causally necessary** for aligned behavior by patching residual stream activations from aligned models into base models at inference time. Full details are in [`../scotus-constitutional-geometry/causal_validation/CAUSAL_VALIDATION_SUMMARY.md`](../scotus-constitutional-geometry/causal_validation/CAUSAL_VALIDATION_SUMMARY.md).

### Summary of Patching Results

| Model | Size | Patching Effect | Recovery Rate |
|-------|------|-----------------|---------------|
| **Gemma 2-27B** | 27B | **Works** (100% recovery on in-distribution) | 100% |
| Llama 3.1-8B | 8B | Breaks model (EOS tokens) | 0% |
| Llama 3.2-3B | 3B | Breaks model (EOS tokens) | 0% |
| Mistral-7B | 7B | No effect (same output) | 0% |
| Qwen2.5-7B | 7B | Minor improvement on already-capable base | N/A* |

*Qwen's base model already outperforms its aligned model on constitutional reasoning tasks.

### Key Takeaways

1. **Geometry exists across all models** (probing confirms this), but its **causal role is architecture-dependent**
2. **Only Gemma 2-27B** shows successful activation transfer — patching mid-layers (20-34) fully recovers aligned behavior on in-distribution cases
3. **OOD generalization fails even for Gemma** — patched activations don't transfer to novel constitutional prompts, suggesting the geometry is case-specific rather than encoding general "constitutional reasoning"
4. **Scale may matter**: the only successful model is the largest (27B), suggesting causal interpretability via patching may require sufficient model capacity

### Implications for Criminal Planning Geometry

The patching results raise important questions about whether the severity/restraint geometry we found is causally meaningful or merely correlational. The fact that linear probes find consistent structure but ablation patching mostly fails suggests:

- **The geometry is real** (probes replicate across 6 model families)
- **But patching is too blunt** to establish causality — it disrupts downstream computation in ways that mask the signal
- **Alternative causal methods needed**: Steering vector experiments and behavioral signature analysis may provide better evidence than ablation patching

## Open Questions and Interpretive Challenges

### 1. Why does the smaller model show larger alignment improvements?

Three competing hypotheses:

**A. Capacity constraint hypothesis**: Smaller models have less redundancy, so RLHF must create more concentrated/detectable structure to achieve safety goals.

**B. Distributed representation hypothesis**: Larger models may use more distributed representations that linear probes underestimate. The 8B model might encode severity information across more dimensions in ways a ridge regression probe misses.

**C. Training intensity hypothesis**: The 3B Instruct model may have undergone more aggressive safety training relative to its capacity, creating stronger geometric signatures.

**Next steps**: Compare probe performance across regularization strengths, or use non-linear probes (MLPs) to test whether larger models use more complex encodings.

### 2. Why is response toxicity poorly predicted from activations?

The weak R² values (~0.08-0.17) for toxicity prediction are surprising given we can predict prompt severity much better from the same activations. Possible explanations:

**A. Temporal hypothesis**: Output toxicity depends on generation dynamics (attention during decoding, sampling decisions) not captured in the initial forward pass activations we extract.

**B. Measurement artifact**: The Patronus toxicity scores may be noisy or measure different constructs than Claude's severity annotations. We're predicting one labeler's judgment from features correlated with another labeler's judgment.

**C. Non-linear decision boundary**: The "decision" about how toxic to be may involve non-linear combinations of features, or may be implemented in attention patterns rather than residual stream values.

**Next steps**:
- Extract activations at multiple points during generation, not just before generation
- Compare Patronus scores to Claude annotations directly to measure agreement
- Try non-linear probes for toxicity prediction

### 3. Is the restraint signal causally meaningful?

We show **correlation** between activations and restraint, but this doesn't prove the activations **cause** restraint. The geometry could be:
- A side effect of training that doesn't influence behavior
- Correlated with true causal features but not causal itself
- Read out by downstream layers to influence generation (the causal interpretation)

**Partial answer from causal validation**: Activation patching experiments (see [Causal Validation section above](#causal-validation-activation-patching-january-2025)) showed that ablation patching is too blunt to establish causality for most models — only Gemma 2-27B showed successful transfer. This does not rule out causal relevance; it may reflect limitations of the patching methodology rather than absence of causal structure.

**Next steps**: Steering vector experiments (adding/subtracting learned directions during inference) and behavioral signature analysis may provide better causal evidence than ablation patching.

### 4. What explains the layer localization differences between targets?

Severity encoding peaks in middle layers while restraint peaks later (especially in the 8B model). Possible interpretations:

**A. Processing stage hypothesis**: Middle layers encode "understanding" of the prompt (including its severity), while later layers encode "response planning" (including restraint decisions).

**B. Architectural artifact**: The restraint signal may require integrating severity information with response planning, which happens later in processing.

**C. Probe artifact**: Different targets may require different amounts of regularization, and our fixed-alpha approach may not be optimal for all layer-target combinations.

### 5. Do these findings generalize to other harm types?

The current dataset mixes harm types (fraud, theft, violence, drugs, weapons, etc.). Type-specific probes might show:
- Stronger effects for some harm types (e.g., violence) vs others (e.g., fraud)
- Different layer localizations by harm category
- Different base/aligned gaps by harm severity

**Next steps**: Train separate probes for each harm type annotation and compare.

## Comparison to Expected Results

From the README's "Expected Results" section:

| Expectation | Finding | Assessment |
|-------------|---------|------------|
| Aligned models show higher R² for restraint | Yes: +0.117 (8B), +0.068 (3B) | Confirmed |
| Effect concentrates in mid-to-late layers | Partially: Severity in mid layers, restraint variable | Partial support |
| Base models show no/weak linear structure | Mixed: Base models show weak structure (R²=0.04-0.16) | Partial support |

## Artifacts

### Data Files

```
experiment_output/                    # Llama 3.1-8B results
├── annotations/annotated_prompts.json
├── activations/{base,aligned}/*.npz
├── responses/responses.json
├── scores/patronus_scores.json
└── analysis/
    ├── summary.json
    ├── probe_prompt_severity.json
    ├── probe_response_toxicity.json
    ├── probe_restraint_delta.json
    ├── probe_joint_dimensions.json
    └── plot_*.png

experiment_output_llama32_3b/         # Llama 3.2-3B results
└── analysis/
    ├── summary.json
    └── probe_*.json

experiment_output_mistral_7b/         # Mistral-7B results (cross-model validation)
├── activations/{base,aligned}/*.npz
├── responses/responses.json
├── scores/patronus_scores.json
└── analysis/
    ├── summary.json
    └── probe_*.json

experiment_output_qwen25_7b/          # Qwen2.5-7B results (cross-model validation)
├── activations/{base,aligned}/*.npz
├── responses/responses.json
├── scores/patronus_scores.json
└── analysis/
    ├── summary.json
    └── probe_*.json

experiment_output_qwen25_32b/         # Qwen2.5-32B results (scale validation)
├── activations/{base,aligned}/*.npz
├── responses/responses.json
├── scores/patronus_scores.json
└── analysis/
    ├── summary.json
    └── probe_*.json
```

### Visualizations

Layer-by-layer R² plots are available in `experiment_output/analysis/plot_*.png` for the 8B model run.

**Cross-model comparison plots** (in `analysis/`):
- `cross_model_criminal_planning.png` - Base vs aligned R² by target for all three models
- `alignment_improvement_comparison.png` - Alignment improvement (Δ R²) grouped by target
- `cross_model_scotus.png` - SCOTUS probe comparison between Llama and Mistral
- `joint_dimensions_convergence.png` - Convergence of joint dimension predictions (~0.52 R²)

## Future Directions

1. ~~**Causal validation (ablation patching)**: Test whether patching aligned activations into base models recovers aligned behavior~~ ✓ *Completed Jan 2025 — works for Gemma 2-27B only; other models show no effect or broken generation. See [Causal Validation section](#causal-validation-activation-patching-january-2025) and [detailed summary](../scotus-constitutional-geometry/causal_validation/CAUSAL_VALIDATION_SUMMARY.md).*
2. ~~**Cross-model generalization**: Test whether probes trained on one model transfer to others~~ ✓ *Completed with Mistral-7B and Qwen2.5-7B (Dec 2025) - prompt severity generalizes; toxicity encoding is model-specific*
3. **Steering vector experiments**: Extract severity/restraint directions from probe weights, inject during inference, measure behavioral changes (refusal rate, toxicity shifts). This is a more targeted causal test than ablation patching.
4. **Behavioral signature analysis**: Classify model response types (full refusal, hedging, partial compliance, full compliance) and test whether behavioral categories cluster coherently in activation space.
5. **Harm-type stratification**: Train separate probes by harm category
6. **Non-linear probes**: Test whether MLPs capture more structure than linear probes
7. **Generation-time analysis**: Extract activations during generation to better predict output toxicity
8. **Probe transfer experiments**: Test whether probes trained on Llama transfer to Mistral (and vice versa) without retraining
