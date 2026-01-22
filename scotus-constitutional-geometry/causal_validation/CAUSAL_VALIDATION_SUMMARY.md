# Causal Validation Summary: Cross-Model Analysis

## Overview

This document summarizes causal validation experiments testing whether activation geometry discovered via linear probing is **causally necessary** for aligned constitutional reasoning behavior.

**Method**: Patch residual stream activations from aligned model into base model at specific layers during inference, measuring whether this recovers aligned behavior.

---

## Results Summary

### Constitutional Case Patching (In-Distribution)

| Model | Size | Layers Patched | Base | Aligned | Patched | Recovery Rate |
|-------|------|----------------|------|---------|---------|---------------|
| **Gemma-2** | 27B | 20-34 | 8.3% | 83.3% | 83.3% | **100%** |
| Llama-3.2 | 3B | 14-24 | 0.0% | 83.3% | 0.0% | 0% |
| Llama-3.1 | 8B | 16-28 | 0.0% | 83.3% | 0.0% | 0% |
| Mistral | 7B | 16-28 | 8.3% | 58.3% | 8.3% | 0% |
| **Qwen-2.5** | 7B | 16-28 | **75.0%** | 58.3% | **83.3%** | N/A* |

*Qwen's base model outperforms its aligned model, inverting the expected pattern.

### Response Behavior by Model

| Model | Base Response | Patched Response | Aligned Response |
|-------|--------------|------------------|------------------|
| Gemma-2-27B | `<eos>` (no output) | Coherent legal analysis | Coherent legal analysis |
| Llama-3.2-3B | `<\|end_of_text\|>` | `<\|end_of_text\|>` | Coherent legal analysis |
| Llama-3.1-8B | `<\|end_of_text\|>` | `<\|end_of_text\|>` | Coherent legal analysis |
| Mistral-7B | BrainMass SEO spam | BrainMass SEO spam | Coherent legal analysis |
| Qwen-2.5-7B | Coherent legal analysis | Coherent legal analysis | Coherent legal analysis |

### OOD Generalization (Novel Constitutional Prompts)

| Model | Base Coherent | Patched Coherent | Aligned Coherent | Notes |
|-------|---------------|------------------|------------------|-------|
| Gemma-2-27B | 0/6 | 0/6 | 6/6 | Patching produces accounting gibberish |
| Llama-3.2-3B | 6/6 | 6/6 | 6/6 | Base produces repetitive but coherent text |
| Llama-3.1-8B | 6/6 | 0/6 | 6/6 | Patching completely breaks generation |
| Mistral-7B | 6/6 | 6/6 | 6/6 | All produce coherent responses |
| Qwen-2.5-7B | 6/6 | 6/6 | 6/6 | All produce coherent responses |

---

## Observed Patterns

### Pattern 1: Scale Dependence
- Only the largest model (Gemma-2-27B) shows successful activation transfer
- 7B models (Mistral, Qwen) show no effect from patching
- 3B/8B Llama models show patching actively breaks generation

### Pattern 2: Architecture Dependence
| Architecture | Patching Effect |
|--------------|-----------------|
| Gemma-2 | **Works** (100% recovery) |
| Llama-3.x | **Breaks** model (EOS tokens) |
| Mistral | **No effect** (same output) |
| Qwen-2.5 | **Minor improvement** on already-capable base |

### Pattern 3: Base Model Capability Varies Dramatically
| Model | Base Constitutional Reasoning |
|-------|------------------------------|
| Gemma-2-27B | None (outputs EOS) |
| Llama-3.x | None (outputs EOS) |
| Mistral-7B | None (outputs web scrape artifacts) |
| **Qwen-2.5-7B** | **Strong** (75% accuracy) |

### Pattern 4: Alignment Effect Direction
| Model | Alignment Effect |
|-------|-----------------|
| Gemma-2 | +75% (creates capability) |
| Llama-3.2 | +83% (creates capability) |
| Llama-3.1 | +83% (creates capability) |
| Mistral | +50% (creates capability) |
| **Qwen-2.5** | **-17%** (degrades capability) |

---

## Potential Explanatory Frameworks

### Framework 1: Activation Geometry is Model-Family Specific
The geometric structure we found via probing may be organized differently across model families:
- **Gemma**: Alignment creates separable geometry at layers 20-34 that is directly patchable
- **Llama**: Alignment creates capability but geometry is distributed differently (not captured by our layer range)
- **Mistral**: Geometry may exist but requires different patching approach
- **Qwen**: Geometry exists in base model (pre-training effect, not RLHF effect)

**Implication**: Linear probes find structure, but that structure's causal role varies by architecture.

### Framework 2: Scale Threshold for Transferable Representations
Activation patching may require sufficient model capacity:
- 27B parameters: Rich enough representations to transfer cleanly
- 7-8B parameters: Representations too entangled/compressed for naive patching
- 3B parameters: Insufficient capacity for separable alignment representations

**Implication**: Causal interpretability techniques may only work above certain scale thresholds.

### Framework 3: Instruction-Following vs. Constitutional Reasoning are Distinct
The "alignment" being measured conflates two capabilities:
1. **Instruction-following**: Ability to respond to prompts in expected format
2. **Constitutional reasoning**: Knowledge of legal principles

| Model | Instruction-Following Source | Constitutional Reasoning Source |
|-------|------------------------------|--------------------------------|
| Gemma | RLHF | RLHF |
| Llama | RLHF | RLHF |
| Mistral | RLHF | RLHF (weak) |
| Qwen | RLHF | **Pre-training** |

**Implication**: Qwen's pre-training included substantial legal/constitutional text, making it capable without instruction-tuning.

### Framework 4: Patching Layer Mismatch
Our probing found linearly separable representations at specific layers, but:
- Gemma: Probed layers match causal layers
- Other models: Causal mechanism may be at different layers than where geometry is most separable

**Evidence**: Llama probing showed different layer profiles than Gemma. We may be patching the wrong layers.

### Framework 5: Representation Interference
Patching activations may cause interference effects:
- **Gemma**: Activations are "compatible" with base model processing
- **Llama**: Patched activations cause cascade failures (immediate EOS)
- **Mistral**: Patched activations are ignored (no effect)
- **Qwen**: Patched activations provide mild signal boost

**Implication**: Activation geometry is necessary but not sufficient; compatibility with downstream processing matters.

---

## Key Questions for Further Investigation

1. **Layer Selection**: Would different layer ranges work better for Llama/Mistral/Qwen?
   - Re-run with layers matched to each model's probing results

2. **Patching Granularity**: Are we patching too many/few layers?
   - Try single-layer patching to identify critical layers

3. **Activation Compatibility**: Why does Llama produce EOS when patched?
   - Analyze activation norms/distributions pre/post patch

4. **Qwen Pre-training**: What's in Qwen's pre-training data?
   - May have legal/constitutional corpus that other models lack

5. **OOD Failure Modes**: Why does Gemma patching break on OOD but work on in-distribution?
   - Suggests activations are case-specific, not general "constitutional reasoning" representations

---

## Conclusions

### What We Can Claim
1. **Activation geometry exists** across all models (probing works)
2. **Geometry is causally sufficient for Gemma-2-27B** (100% recovery via patching)
3. **Geometry is necessary but not sufficient** for in-distribution tasks (OOD patching fails even for Gemma)

### What We Cannot Claim
1. ❌ Universal causal role of activation geometry (only works for Gemma)
2. ❌ RLHF creates constitutional reasoning (Qwen has it without alignment)
3. ❌ Activation patching as general interpretability technique (highly model-specific)

### Revised Thesis
> The constitutional geometry discovered via linear probing represents a **model-specific** encoding of value-relevant information. Its causal role in producing aligned behavior depends on architecture, scale, and the interaction between patched representations and downstream model computations. For Gemma-2-27B, this geometry is directly causal; for other models, the relationship is more complex.
