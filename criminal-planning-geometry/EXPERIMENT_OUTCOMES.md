# Experiment Outcomes: Criminal Planning Geometry

**Experiment Run Dates**: December 10-17, 2025

**Dataset**: [`criminal-planning-prompts`](data/criminal-planning-prompts.jsonl) by [Patronus AI](https://www.patronus.ai/) — 100 prompts eliciting help with planning a crime, from Patronus's off-the-shelf evaluation datasets. Response toxicity is scored with the [Patronus API](https://docs.patronus.ai/) `toxicity` evaluator.

## Executive Summary

I tested whether RLHF creates detectable geometric structures in transformer activations that correspond to safety-relevant concepts. Linear probes trained on residual stream activations show that **instruction-tuned (IT) models encode prompt severity and restraint signals more linearly than base models**, with consistent findings across four model families at 7B-8B scale (Llama 3.1-8B, Llama 3.2-3B, Mistral-7B, Qwen2.5-7B). Notably, Qwen-7B shows divergent behavior on response toxicity, suggesting model-family-specific safety mechanisms.

The probes predict three quantities, which appear as the prediction targets in the results tables below:

- **Prompt severity** — how transgressive the input prompt is (0–1, Claude annotation)
- **Response toxicity** — how harmful the model's actual output is (Patronus scoring)
- **Restraint delta** — prompt severity minus response toxicity: how much the model "holds back" relative to the harmfulness of the input

**Major finding at scale**: Testing at 27B-32B scale reveals **model-family-specific divergence**:
- **Qwen2.5-32B**: Reversed alignment effects for criminal planning (base outperforms IT across all targets)
- **Gemma 2-27B**: Mixed effects - base slightly better on severity (-0.015), but IT better on restraint (+0.077) and toxicity (+0.053)

**Observation**: Gemma 2-27B shows the same pattern as small models on the [SCOTUS constitutional-geometry experiment](../scotus-constitutional-geometry/results/EXPERIMENT_OUTCOMES.md) - near-zero base structure (R²=0.04) but strong IT structure (R²=0.48). This suggests conceptual structure does not emerge from scale alone — RLHF or post-training may be needed for conceptual emergence in many cases.

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

| Model | Target | Base R² | IT R² | Improvement |
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

| Model | Target | Base Best Layer | IT Best Layer |
|-------|--------|-----------------|-------------------|
| **Llama 3.1-8B** | Prompt Severity | Layer 8 | Layer 11 |
| | Restraint Delta | Layer 31 | Layer 31 |
| **Llama 3.2-3B** | Prompt Severity | Layer 27 | Layer 9 |
| | Restraint Delta | Layer 8 | Layer 9 |
| **Mistral-7B** | Prompt Severity | Layer 26 | Layer 28 |
| | Restraint Delta | Layer 7 | Layer 10 |
| **Qwen2.5-7B** | Prompt Severity | Layer 12 | Layer 16 |
| | Restraint Delta | Layer 15 | Layer 16 |
| **Qwen2.5-32B** | Prompt Severity | Layer 24 | Layer 24 |
| | Restraint Delta | Layer 24 | Layer 24 |
| **Gemma 2-27B** | Prompt Severity | Layer 25 | Layer 16 |
| | Restraint Delta | Layer 10 | Layer 10 |

## Key Findings

### 1. Alignment Improves Linear Predictability of Prompt Severity — at 7-8B Scale

All four models at 7-8B scale show gains in predicting how transgressive a prompt is from activations:
- **Llama 3.2-3B**: +0.185 R² improvement (0.043 → 0.228)
- **Llama 3.1-8B**: +0.117 (0.161 → 0.278)
- **Mistral-7B**: +0.086 (0.105 → 0.192)
- **Qwen2.5-7B**: +0.064 (0.180 → 0.244)

At larger scale the effect reverses: Gemma 2-27B (−0.015) and Qwen2.5-32B (−0.041) show base models slightly ahead. Base-model severity R² rises steadily with scale (0.043 at 3B → 0.236 at 27B), suggesting large base models already encode severity linearly, leaving alignment little to add. Consistent with this, the smallest model shows the largest improvement delta — RLHF may create more pronounced geometric structure when model capacity is more constrained.

### 2. Restraint Signal is Detectable but Modest

The restraint delta is more predictable from IT model activations in three of six model families:
- **Llama 3.1-8B**: Base R²=0.077, IT R²=0.193 (+0.117)
- **Llama 3.2-3B**: Base R²=0.085, IT R²=0.153 (+0.068)
- **Gemma 2-27B**: Base R²=0.128, IT R²=0.205 (+0.077)

The advantage is model-family-dependent: it is absent for Mistral-7B (+0.002), Qwen2.5-7B shows no restraint signal in either condition (R² ≈ 0), and Qwen2.5-32B encodes restraint (R² ≈ 0.17) but equally in base and IT versions. Larger models show stronger absolute restraint encoding where the signal exists (Gemma 27B: 0.205, Llama 8B: 0.193 vs Llama 3B: 0.153), possibly reflecting greater capacity for representing nuanced safety distinctions.

### 3. Response Toxicity Prediction is Weak Across All Conditions

Neither base nor IT models show strong linear structure for predicting output toxicity:
- Best performance: ~0.18 R² (Llama 3.2-3B)
- IT models show minimal improvement over base
- This finding holds across all six models — no model at any scale exceeds ~0.18 R². (Mistral-7B's nominal +0.261 improvement is recovery from a degenerate base R² of −0.192 to a still-weak 0.069.)

This suggests output toxicity may be determined by:
- Generation dynamics not captured in pre-generation activations
- Non-linear combinations of features
- Factors external to the residual stream (attention patterns, etc.)

### 4. Layer Localization Patterns

**Prompt Severity**: Best prediction occurs in middle layers for most IT models
- Llama 8B: Layer 11 of 32 (~34% through network); Llama 3B: Layer 9 of 28 (~32%)
- Gemma 27B: Layer 16 of 46 (~35%); Qwen 32B: Layer 24 of 64 (~38%)
- Exceptions: Mistral-7B peaks late (layer 28 of 32, ~88%); Qwen-7B peaks mid-late (layer 16 of 28, ~57%)

**Restraint Delta**: More varied across models
- Llama 8B: Both base and IT peak at layer 31 (final layer)
- Llama 3B: layers 8-9 (~30%); Mistral: layer 10 (~31%); Gemma 27B: layer 10 of 46 (~22%); Qwen 32B: layer 24 of 64 (~38%)

The concentration in middle layers for severity is consistent with "representation engineering" findings that mid-network representations encode semantic and conceptual information, while early layers process syntax and late layers prepare outputs. Strikingly, the ~32-38% depth localization for IT severity holds in four of six model families, including both large models — the fractional depth is roughly scale-invariant.

### 5. Joint Dimension Prediction

Predicting all four annotation dimensions jointly (severity, specificity, real_world_risk, harm_type) shows:
- Modest IT model advantage (~+0.02 R²)
- Remarkably similar performance across all six models — IT R² spans just 0.499-0.531 and base R² 0.479-0.508, regardless of architecture or scale
- This suggests harm type classification may not benefit as much from alignment as scalar severity does

## Cross-Model Validation: Mistral-7B and Qwen2.5-7B Results

To test whether the geometric structures are model-family specific or represent general properties of RLHF, I ran the same experiment on Mistral-7B/7B-Instruct and Qwen2.5-7B/7B-Instruct (December 2025).

### Key Cross-Model Findings

#### 1. Alignment Advantage Generalizes for Prompt Severity

All four models show improved linear predictability for prompt severity in IT versions:

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
- **IT Qwen**: R² = **-0.043** (worse than random)
- **Improvement**: **-0.056** (regression!)

In contrast, Mistral showed the largest positive improvement (+0.261) for toxicity.

**Possible explanations**:
- Qwen's safety training may use fundamentally different mechanisms (e.g., output filtering rather than representation restructuring)
- Cultural/training data differences may affect how toxicity is represented internally
- Qwen may encode toxicity decisions in attention patterns rather than residual stream

This divergence is the most significant finding for cross-model validation - it suggests **toxicity encoding mechanisms are NOT universal across model families**.

#### 4. Joint Dimension Prediction Remains Consistent

All four IT models achieve similar joint dimension R² (~0.51-0.52):
- Llama 8B: 0.518
- Llama 3B: 0.521
- Mistral 7B: 0.520
- Qwen 7B: 0.509

This convergence suggests a "ceiling" for linear probing of harm dimensions, possibly reflecting the inherent limits of linear readout from these representations.

### SCOTUS Constitutional Geometry Cross-Validation

I also ran the SCOTUS constitutional principles experiment on Mistral-7B and Qwen2.5-7B:

| Model | Best Base R² | Best IT R² | Improvement |
|-------|--------------|-----------------|-------------|
| Llama 3.1-8B | 0.24 (layer 30) | 0.41 (layer 12) | +0.17 |
| Mistral-7B | 0.26 (layer 15) | 0.40 (layer 26) | +0.14 |
| Qwen2.5-7B | **-0.14** (layer 3) | **0.23** (layer 16) | **+0.37** |

**Key observations**:
- Llama and Mistral achieve similar IT performance (~0.40-0.41 R²)
- **Qwen shows dramatically weaker signal** for both base (-0.14) and IT (0.23)
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

I ran Qwen2.5-32B (64 layers, 32B parameters) and Gemma 2-27B (46 layers, 27B parameters) to test whether larger scale affects alignment geometry. **The results reveal model-family-specific patterns at scale.**

### Criminal Planning Results (Large Scale)

| Model | Target | Base R² | IT R² | Improvement |
|-------|--------|---------|------------|-------------|
| **Qwen2.5-32B** | Prompt Severity | **0.228** | 0.187 | **-0.041** |
| | Response Toxicity | 0.063 | 0.058 | -0.005 |
| | Restraint Delta | 0.166 | 0.165 | -0.001 |
| | Joint Dimensions | 0.506 | 0.499 | -0.007 |
| **Gemma 2-27B** | Prompt Severity | **0.236** | 0.221 | **-0.015** |
| | Response Toxicity | -0.015 | 0.038 | +0.053 |
| | Restraint Delta | 0.128 | 0.205 | **+0.077** |
| | Joint Dimensions | 0.508 | 0.531 | +0.024 |

**Key difference**: Qwen-32B base outperforms IT across ALL targets. Gemma-27B shows mixed results - base slightly better on severity, but IT better on restraint and toxicity.

### SCOTUS Results (Large Scale)

| Model | Base R² | IT R² | Improvement |
|-------|---------|------------|-------------|
| Qwen2.5-32B | 0.063 (layer 29) | 0.205 (layer 49) | **+0.142** |
| **Gemma 2-27B** | **0.044** (layer 11) | **0.478** (layer 23) | **+0.434** |

**Critical finding**: Gemma 2-27B shows the **strongest SCOTUS improvement** of any model tested (+0.434). Despite being 27B parameters, the base model has essentially no linear structure (0.044 R²), matching the pattern of the much smaller Llama 3.2-3B (-0.24 R²).

### Interpretation: Conceptual Structure Doesn't Emerge from Scale Alone

The Gemma 2-27B results suggest base models don't reliably develop conceptual representations with scale:

| Model | Scale | Base SCOTUS R² | IT SCOTUS R² |
|-------|-------|----------------|-------------------|
| Llama 3.2-3B | 3B | -0.24 | 0.49 |
| **Gemma 2-27B** | **27B** | **0.04** | **0.48** |

Despite a 9x scale difference, both models show:
- Near-zero base structure (random/worse-than-random)
- Nearly identical IT structure (~0.48 R²)

**This suggests RLHF or post-training may be needed for conceptual emergence** in many cases — base models don't produce these representations by default.

### Model-Family Differences at Scale

**Qwen (Chinese, 32B)**:
- Criminal planning: Base wins on all targets
- SCOTUS: IT wins, but with weaker signal (0.21 R²)

**Gemma (Western, 27B)**:
- Criminal planning: Mixed - base wins on severity, IT wins on restraint/toxicity
- SCOTUS: IT wins strongly (0.48 R²), matching smaller Western models

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

## Open Questions and Interpretive Challenges

### 1. Why does the smaller model show larger alignment improvements?

Three competing hypotheses:

**A. Capacity constraint hypothesis**: Smaller models have less redundancy, so RLHF must create more concentrated/detectable structure to achieve safety goals.

**B. Distributed representation hypothesis**: Larger models may use more distributed representations that linear probes underestimate. The 8B model might encode severity information across more dimensions in ways a ridge regression probe misses.

**C. Training intensity hypothesis**: The 3B Instruct model may have undergone more aggressive safety training relative to its capacity, creating stronger geometric signatures.

**Next steps**: Compare probe performance across regularization strengths, or use non-linear probes (MLPs) to test whether larger models use more complex encodings.

### 2. Why is response toxicity poorly predicted from activations?

The weak R² values (~0.08-0.17) for toxicity prediction are surprising given I can predict prompt severity much better from the same activations. Possible explanations:

**A. Temporal hypothesis**: Output toxicity depends on generation dynamics (attention during decoding, sampling decisions) not captured in the initial forward pass activations I extract.

**B. Measurement artifact**: The Patronus toxicity scores may be noisy or measure different constructs than Claude's severity annotations. I'm predicting one labeler's judgment from features correlated with another labeler's judgment.

**C. Non-linear decision boundary**: The "decision" about how toxic to be may involve non-linear combinations of features, or may be implemented in attention patterns rather than residual stream values.

**Next steps**:
- Extract activations at multiple points during generation, not just before generation
- Compare Patronus scores to Claude annotations directly to measure agreement
- Try non-linear probes for toxicity prediction

### 3. Is the restraint signal causally meaningful?

I show **correlation** between activations and restraint, but this doesn't prove the activations **cause** restraint. The geometry could be:
- A side effect of training that doesn't influence behavior
- Correlated with true causal features but not causal itself
- Read out by downstream layers to influence generation (the causal interpretation)

**Related evidence from SCOTUS work**: Activation patching experiments on SCOTUS constitutional-reasoning data (see [`../scotus-constitutional-geometry/results/gemma2_27b/CAUSAL_VALIDATION_SUMMARY.md`](../scotus-constitutional-geometry/results/gemma2_27b/CAUSAL_VALIDATION_SUMMARY.md)) found ablation patching too blunt to establish causality for most models — only Gemma 2-27B showed successful transfer. No causal validation has been run on criminal-planning data, so the causal status of the severity/restraint geometry remains untested here.

**Next steps**: Steering vector experiments (adding/subtracting learned directions during inference) and behavioral signature analysis may provide better causal evidence than ablation patching.

### 4. What explains the layer localization differences between targets?

Severity encoding peaks in middle layers while restraint peaks later (especially in the 8B model). Possible interpretations:

**A. Processing stage hypothesis**: Middle layers encode "understanding" of the prompt (including its severity), while later layers encode "response planning" (including restraint decisions).

**B. Architectural artifact**: The restraint signal may require integrating severity information with response planning, which happens later in processing.

**C. Probe artifact**: Different targets may require different amounts of regularization, and my fixed-alpha approach may not be optimal for all layer-target combinations.

### 5. Do these findings generalize to other harm types?

The current dataset mixes harm types (fraud, theft, violence, drugs, weapons, etc.). Type-specific probes might show:
- Stronger effects for some harm types (e.g., violence) vs others (e.g., fraud)
- Different layer localizations by harm category
- Different base/IT gaps by harm severity

**Next steps**: Train separate probes for each harm type annotation and compare.

## Comparison to Expected Results

From the README's "Expected Results" section:

| Expectation | Finding | Assessment |
|-------------|---------|------------|
| IT models show higher R² for restraint | Yes: +0.117 (8B), +0.068 (3B) | Confirmed |
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
- `cross_model_criminal_planning.png` - Base vs IT R² by target for all three models
- `alignment_improvement_comparison.png` - Alignment improvement (Δ R²) grouped by target
- `cross_model_scotus.png` - SCOTUS probe comparison between Llama and Mistral
- `joint_dimensions_convergence.png` - Convergence of joint dimension predictions (~0.52 R²)

## Future Directions

1. **Causal validation (ablation patching)**: Test whether patching IT activations into base models recovers IT behavior on criminal-planning data. *Not yet run on this dataset — completed only on SCOTUS constitutional data (Jan 2026), where it worked for Gemma 2-27B only; see [`../scotus-constitutional-geometry/results/gemma2_27b/CAUSAL_VALIDATION_SUMMARY.md`](../scotus-constitutional-geometry/results/gemma2_27b/CAUSAL_VALIDATION_SUMMARY.md).*
2. ~~**Cross-model generalization**: Test whether probes trained on one model transfer to others~~ ✓ *Completed with Mistral-7B and Qwen2.5-7B (Dec 2025) - prompt severity generalizes; toxicity encoding is model-specific*
3. **Steering vector experiments**: Extract severity/restraint directions from probe weights, inject during inference, measure behavioral changes (refusal rate, toxicity shifts). This is a more targeted causal test than ablation patching.
4. **Behavioral signature analysis**: Classify model response types (full refusal, hedging, partial compliance, full compliance) and test whether behavioral categories cluster coherently in activation space.
5. **Harm-type stratification**: Train separate probes by harm category
6. **Non-linear probes**: Test whether MLPs capture more structure than linear probes
7. **Generation-time analysis**: Extract activations during generation to better predict output toxicity
8. **Probe transfer experiments**: Test whether probes trained on Llama transfer to Mistral (and vice versa) without retraining
