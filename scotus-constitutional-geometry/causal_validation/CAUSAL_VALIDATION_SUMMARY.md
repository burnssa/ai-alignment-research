# Causal Validation Summary: Cross-Model Analysis

## Overview

This document summarizes causal validation experiments testing whether activation geometry discovered via linear probing is **causally relevant** to aligned constitutional reasoning behavior.

Two complementary approaches were tested:

1. **Activation Patching**: Replace residual stream activations from aligned model into base model at specific layers during inference, measuring whether this recovers aligned behavior.
2. **Activation Steering**: Add scaled probe-derived principle directions to the aligned model's residual stream during inference, measuring whether this shifts which constitutional principles the model invokes.

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

## Activation Steering Experiments (Gemma 2-27B)

Given that activation patching recovered aligned behavior for Gemma 2-27B, we next tested whether the probe-derived principle directions could *steer* model outputs — i.e., whether adding a scaled "free expression" direction to activations on a case where free expression is irrelevant would cause the model to rank that principle higher.

**Method**: Extract per-principle direction vectors from trained Ridge probes (with scaler correction), select test cases where the target principle has low ground-truth weight, then add the scaled direction to the aligned model's residual stream during autoregressive generation. Measure whether the steered principle's rank shifts monotonically with the scaling factor (alpha). All generation used temperature 0.0 for deterministic outputs.

### Experiment Rounds

Five rounds of steering experiments were conducted, with the critical breakthrough coming in Round 5 when alpha was calibrated relative to the residual stream norm:

| Round | Layers | Alpha Range | Position | Trials | Date |
|-------|--------|-------------|----------|--------|------|
| 1. Standard | 20, 23, 26 | -3 to +3 | Last token only | 675 | 2026-02-12 |
| 2. Large alpha | 23 | -500 to +500 | All tokens | 90 | 2026-02-13 |
| 3. All-positions | 23 | -3 to +3 | All tokens | 90 | 2026-02-13 |
| 4. Medium alpha | 23 | -50 to +50 | All tokens | 225 | 2026-02-19 |
| **5. Norm-relative** | **23** | **-1.0 to +1.0 x ‖resid‖** | **All tokens** | **175** | **2026-03-03** |

Rounds 1-4 used unit-normalized probe directions, meaning the perturbation L2 norm equaled the raw alpha value. Round 5 discovered that the mean residual stream L2 norm at layer 23 is **19,030** — so Rounds 1-4 were operating at **<3% of residual norm**, well below the threshold where any effect is observable. Round 5 scaled alpha as a fraction of ‖resid‖, revealing a narrow but real window of causal influence.

### Results: Steering Works at Norm-Relative Scale

At alpha=±0.1 (10% of residual norm, effective perturbation scale ~1,903), steering produces measurable behavioral changes:

| Outcome (alpha=+0.1) | Count | Rate |
|-----------------------|:-----:|:----:|
| **Appeared** (absent at baseline, now ranked) | 11/25 | 44% |
| Unchanged | 10/25 | 40% |
| Worsened | 4/25 | 16% |
| Improved to higher rank | 0/25 | 0% |

| Outcome (alpha=-0.1) | Count | Rate |
|-----------------------|:-----:|:----:|
| **Suppressed** (rank dropped or vanished) | 5/25 | 20% |
| Unchanged | 13/25 | 52% |
| Boosted (wrong direction) | 7/25 | 28% |

The best-performing principle was **Free Expression**: 5/5 appearance rate at alpha=+0.1 vs 0/5 at baseline.

### The Coherence Cliff

There is a narrow usable window between "too small to observe" and "destroys generation":

| Alpha | % of ‖resid‖ | Effective Scale | Parseable Responses |
|:-----:|:---:|:--------------:|:-------------------:|
| ±0.1 | 10% | ±1,903 | 24/25 (96%) — coherent but altered |
| ±0.5 | 50% | ±9,515 | 0/25 (0%) — multilingual gibberish |
| ±1.0 | 100% | ±19,030 | 0/25 (0%) — complete collapse |

### Example: Principle Appearance and Suppression

**Case**: *Roe v. Wade* (1973) — steered toward **free expression** at layer 23

| Rank | Alpha = -0.1 (Suppressed) | Alpha = 0.0 (Baseline) | Alpha = +0.1 (Steered) |
|:----:|---------------------------|------------------------|------------------------|
| **1** | **Due Process** — The Court framed the issue as a matter of the Fourteenth Amendment's Due Process Clause... | **Privacy/Liberty** — The core of Roe v. Wade hinges on the right to privacy... | **Privacy/Liberty** — This case hinges on the right to privacy... |
| **2** | **Privacy/Liberty** — The Court recognized a right to privacy, rooted in the Due Process Clause... | **Due Process** — The Fourteenth Amendment's Due Process Clause was used to argue... | **Due Process** — The right to privacy is often argued to be a fundamental right... |
| **3** | **Equal Protection** — While not the primary focus... | **Equal Protection** — While not the central issue... | **Equal Protection** — ...one could argue that the right to choose... |
| **4** | — *(model skips to 5)* | — | **Federalism** — This case deals with the relationship between federal government and states... |
| **5** | **Federalism** — While the case involved a state law... | — | **Free Expression** — This principle is not directly relevant... *(degrades into repetition)* |

At baseline, the model lists 3 principles. Positive steering expands the response to 5 principles, pulling in the targeted direction. Negative steering swaps the top two rankings (Privacy/Liberty and Due Process).

**Case**: *Trump v. Hawaii* (2018) — Privacy/Liberty suppressed at alpha=-0.1

At baseline, Privacy/Liberty appears at rank 5. At alpha=-0.1, it vanishes entirely from the response, replaced by a novel principle (Executive Power/Separation of Powers) not in our standard set.

### Direction-Specific Failure Modes

At alpha ≥ ±0.5, each probe direction activates a consistent, direction-specific failure pattern regardless of input case:

| Direction | Failure Mode (alpha=+0.5) |
|-----------|---------------------------|
| Free Expression | French/Malay/English fragments: "putative putative the wounded wounded définition penerbangan..." |
| Equal Protection | Repeated German: "höher höher höher höher..." |
| Due Process | Finnish/Swedish + hex tokens: "isiäisiäisiä Svenska xFFFFFFFF..." |
| Federalism | German/Greek/Hindi: "ausreichticon ίας अनुसार अनुसार..." |
| Privacy/Liberty | German prefix fragment: "vertrevertrevertrevertre..." |

The consistency of these attractors across input cases suggests the steering vectors are pushing the model into direction-specific regions of token space.

### Interpretation

The norm-relative steering results, combined with the patching results, reveal a coherent picture of how probe directions relate to model behavior:

- **Patching works**: Wholesale replacement of activations at layers 20-34 recovers aligned behavior in the base model (100% recovery rate). This confirms the relevant information *lives* in those layer representations.
- **Steering works within a narrow window**: At 10% of residual norm, probe directions cause the targeted principle to *appear* in 44% of cases where it was absent at baseline, and can successfully suppress principles in some cases. The effect is real but constrained — principles appear at low ranks (3-6) and are never promoted above their baseline rank.
- **Previous null results were a scaling artifact**: Rounds 1-4 operated at <3% of residual norm. The perturbation was simply too small relative to the activation magnitudes to influence generation.

The probe directions are **partially causal** rather than purely epiphenomenal: they can pull principles into or out of the response, but cannot override the model's strong prior about which principles are most relevant to a given case. This suggests the directions encode *salience* (whether a principle is mentioned) more than *ranking* (how prominently it features).

---

## Conclusions

### What We Can Claim
1. **Activation geometry exists** across all models (probing works)
2. **Geometry is causally sufficient for Gemma-2-27B** (100% recovery via patching)
3. **Geometry is necessary but not sufficient** for in-distribution tasks (OOD patching fails even for Gemma)
4. **Probe directions are partially steerable** — at norm-relative scale (10% of ‖resid‖), steering causes targeted principles to appear in 44% of cases where they were absent at baseline, and can suppress principles in some cases. Earlier null results were a scaling artifact (<3% of residual norm).
5. **Steering controls salience, not ranking** — probe directions influence *whether* a principle is mentioned but cannot override the model's prior about *which* principles are most relevant to a given case

### What We Cannot Claim
1. ❌ Universal causal role of activation geometry (patching only works for Gemma)
2. ❌ RLHF creates constitutional reasoning (Qwen has it without alignment)
3. ❌ Activation patching as general interpretability technique (highly model-specific)
4. ❌ Fine-grained rank control via probe directions (principles appear but are never promoted above baseline rank)

### Revised Thesis
> The constitutional geometry discovered via linear probing represents a **model-specific** encoding of value-relevant information. Its causal role in producing aligned behavior depends on architecture, scale, and the interaction between patched representations and downstream model computations. For Gemma-2-27B, wholesale activation patching recovers aligned behavior (100% in-distribution), confirming the information resides in those layers. Probe-derived steering vectors are partially causal: at norm-appropriate scale (~10% of residual stream norm), they control whether principles appear in model output (44% appearance rate), but cannot override the model's ranking prior. This indicates the directions encode principle *salience* rather than principle *importance*, and the causal mechanism for ranking involves distributed interactions beyond a single linear direction.
