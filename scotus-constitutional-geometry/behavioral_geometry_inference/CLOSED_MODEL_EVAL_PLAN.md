# Behavioral-to-Geometry Inference Experiment (Deferred)

**Status**: Deferred — N=8 is too small for meaningful statistical power. Revisit when more model pairs are available.

## Research Question

Can behavioral output features predict internal probe R² (linear probe quality for constitutional principle geometry)? If yes, this enables closed-source model auditing where activations are inaccessible.

## Current Data Limitations

- N=8 observations available (4 model pairs × base/aligned)
- Need |r| > 0.71 for p<0.05 at N=8; power to detect r=0.5 is only ~26%
- Existing analysis (`probe_behavior_correlation.py`) already found r=-0.35 (p=0.65) with N=4 aligned models
- Base model variants have degraded responses (e.g., mistral-7b-base outputs "[Principle]" placeholders), creating a floor effect that would confound base-vs-aligned with instruction-following ability

## What Would Be Needed

- **More model pairs**: At minimum N≥20 observations (10+ model pairs) for reasonable power
- **Diverse architectures**: Currently 4 families (Llama, Gemma, Mistral, Qwen); need more
- **Matched behavioral quality**: Comparing only aligned variants avoids the base model floor effect but requires N≥10 aligned models

## Available Data

| Model | Base R² | Aligned R² | Behavioral Data |
|-------|---------|------------|-----------------|
| llama3.2-3b | -0.242 | +0.485 | Yes |
| gemma2-27b | +0.044 | +0.478 | Yes |
| mistral-7b | +0.261 | +0.401 | Yes |
| qwen25-7b | -0.143 | +0.234 | Yes |
| llama3.1-8b | +0.237 | +0.414 | No (probe only) |
| qwen25-32b | +0.063 | +0.205 | No (probe only) |

## Proposed Features (14)

**Structure**: parse_success_rate, avg_n_principles, listing_completeness, avg_response_length, justification_rate, error_rate

**Accuracy**: top1_accuracy, weighted_accuracy, avg_spearman_rho, avg_ndcg

**Consistency**: principle_entropy, principle_concentration, sensitivity_to_difficulty, pairwise_ordering_acc

## Implementation Notes

- Single self-contained script: `behavioral_geometry_inference.py`
- Uses only cached data (no GPU, no API calls)
- Reuses data loading from `causal_validation/scripts/probe_behavior_correlation.py`
- Bivariate correlations with bootstrap CIs (no regression at small N)
- Skip LLM-as-judge scoring (adds API dependency, N too small to benefit)

## Key Reference Files

- `causal_validation/scripts/probe_behavior_correlation.py` — existing N=4 analysis
- `behavioral_output_*/behavioral_responses.json` — cached behavioral data
- `experiment_output_*/probe_comparison.json` — probe R² targets
- `experiment_output_*/annotations.json` — ground-truth principle weights
