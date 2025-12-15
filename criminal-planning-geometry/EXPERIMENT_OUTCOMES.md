# Experiment Outcomes: Criminal Planning Geometry

**Experiment Run Dates**: December 10-11, 2025

## Executive Summary

We tested whether RLHF creates detectable geometric structures in transformer activations that correspond to safety-relevant concepts. Linear probes trained on residual stream activations show that **aligned models encode prompt severity and restraint signals more linearly than base models**, with consistent findings across two model sizes.

## Models Tested

| Model Pair | Layers | Architecture |
|------------|--------|--------------|
| Llama 3.1-8B / 8B-Instruct | 32 | `experiment_output/` |
| Llama 3.2-3B / 3B-Instruct | 28 | `experiment_output_llama32_3b/` |

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

**Next steps**: Activation patching or steering experiments would test causality. If adding a "restraint direction" to activations increases refusal rates, that supports causal interpretation.

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
```

### Visualizations

Layer-by-layer R² plots are available in `experiment_output/analysis/plot_*.png` for the 8B model run.

## Future Directions

1. **Causal validation**: Test whether activation steering along the learned directions influences model behavior
2. **Cross-model generalization**: Test whether probes trained on one model transfer to others
3. **Harm-type stratification**: Train separate probes by harm category
4. **Non-linear probes**: Test whether MLPs capture more structure than linear probes
5. **Generation-time analysis**: Extract activations during generation to better predict output toxicity
