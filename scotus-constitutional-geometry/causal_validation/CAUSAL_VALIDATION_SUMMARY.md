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

Four rounds of steering experiments were conducted, iterating on the intervention approach after each null result:

| Round | Layers | Alpha Range | Position | Trials | Date |
|-------|--------|-------------|----------|--------|------|
| 1. Standard | 20, 23, 26 | -3 to +3 | **Last token only** | 675 | 2026-02-12 |
| 2. Large alpha | 23 | -500 to +500 | All tokens | 90 | 2026-02-13 |
| 3. All-positions | 23 | -3 to +3 | All tokens | 90 | 2026-02-13 |
| 4. Medium alpha | 23 | -50 to +50 | All tokens | 225 | 2026-02-19 |

Round 1 added the steering direction only at the final token position in the prompt. After observing no effect, we hypothesized that modifying a single token's residual stream was insufficient to shift downstream generation, and switched to adding the direction at **every token position** on every forward pass (following Turner et al. 2023). Rounds 2–4 used this all-positions approach. The null result persisted across both methods.

### Results: No Steering Effect Observed

Across all 1,080 total trials, principle rankings showed **no monotonic relationship with steering magnitude**. Rankings remained essentially constant regardless of alpha:

| Principle | Typical Rank | Rank Variance Across Alphas | Monotonicity (r) |
|-----------|-------------|----------------------------|-------------------|
| Due Process | 2 | 0.00 | -0.054 |
| Equal Protection | 3 | 0.00 | n/a |
| Federalism | 3 | 0.00 | n/a |
| Privacy/Liberty | 4 | 0.00 | n/a |
| Free Expression | 4-5 | ~0.15 | 0.087 |

Even at extreme alpha values (500x the probe direction norm), principle rankings did not shift. At the highest magnitudes, responses occasionally became unparseable (NaN ranks), but when parseable, rankings were unchanged.

### Verifying the Steering Hook Was Active

A natural concern: if outputs are unchanged, was the hook even applied? Text analysis confirms it was. At alpha=50, **80% of responses had different wording** compared to baseline — the model's justifications and phrasing shift, but the principle rankings do not.

| Alpha | Responses with changed text |
|------:|:---------------------------|
| ±10 | 44–60% |
| ±20 | 48–68% |
| ±30 | 52–72% |
| ±50 | 64–80% |

### Example: Wording Changes, Rankings Don't

**Case**: *Roe v. Wade* (1973) — steered toward **free expression** at layer 23

> **Alpha = 0.0 (baseline)**:
>
> 1. **Privacy/Liberty** — The core of Roe v. Wade hinges on the right to privacy, which the Court found implied within the Fourteenth Amendment's Due Process Clause, encompassing a woman's right to make decisions about her own body, including whether to terminate a pregnancy.
> 2. **Due Process** — The Fourteenth Amendment's Due Process Clause was used to argue that the right to privacy is fundamental and that the Texas law violated this right by depriving women of liberty without due process of law.
> 3. **Equal Protection** — While not the central issue, arguments could be made that the Texas law...
>
> *(Only 3 principles listed; free expression not mentioned)*

> **Alpha = +50.0 (50x the free expression probe direction added to residual stream)**:
>
> 1. **Privacy/Liberty** — The core of the case hinges on whether the Constitution protects a woman's right to make personal decisions about her body and reproductive health, which falls under the right to privacy.
> 2. **Due Process** — The Fourteenth Amendment's Due Process Clause was used to argue that the right to privacy is a fundamental right that cannot be infringed upon by the state without due process of law.
> 3. **Equal Protection** — While not the central issue, arguments could be made that laws restricting abortion disproportionately impact women...
> 4. **Federalism** — The case involved a conflict between state law...
> 5. **Free Expression** — This principle is not directly relevant to the core issue...
>
> *(Free expression appears but explicitly ranked last and dismissed as irrelevant)*

The steering demonstrably altered the model's output — justifications are rephrased, the response is longer, and the steered principle even appears in the list. But it is placed at rank 5 with a dismissive explanation. The model "knows" the steering is pushing toward free expression and actively resists reranking it.

**Case**: *Whalen v. Roe* (1977) — steered toward **equal protection** at layer 23

> **Alpha = 0.0**: "Here's a **breakdown** of the constitutional principles..." — Due Process: "If a right to privacy is recognized, the state's action of collecting and storing this information would be subject to due process scrutiny..."
>
> **Alpha = +50.0**: "Here's a **ranking** of the constitutional principles..." — Due Process: "The requirement to report prescription information could be challenged as a violation of due process if it is deemed to be an unreasonable search or seizure..."

Different framing, different legal reasoning, same ranking: Privacy/Liberty > Due Process > Equal Protection (rank 3, unchanged).

### Interpretation

The null steering result, combined with the positive patching result, suggests a clear asymmetry:

- **Patching works**: Wholesale replacement of activations at layers 20-34 recovers aligned behavior in the base model (100% recovery rate). This confirms the relevant information *lives* in those layer representations.
- **Steering doesn't work**: Adding scaled probe directions does not shift fine-grained principle rankings. The probe directions capture *readable* structure (probing succeeds) but do not correspond to the causal mechanism that determines which principles the model emphasizes.

This is consistent with the **readout vs. control** distinction in mechanistic interpretability: linear probes may find directions that *correlate* with principle weights without those directions being the levers that *control* downstream behavior. The causal mechanism likely involves distributed, nonlinear interactions across layers rather than a single addable direction at one layer.

---

## Conclusions

### What We Can Claim
1. **Activation geometry exists** across all models (probing works)
2. **Geometry is causally sufficient for Gemma-2-27B** (100% recovery via patching)
3. **Geometry is necessary but not sufficient** for in-distribution tasks (OOD patching fails even for Gemma)
4. **Probe directions are readable but not steerable** — linear probes find correlational structure that does not function as a causal control lever for fine-grained principle selection (1,080 steering trials, zero monotonic effect)

### What We Cannot Claim
1. ❌ Universal causal role of activation geometry (patching only works for Gemma)
2. ❌ RLHF creates constitutional reasoning (Qwen has it without alignment)
3. ❌ Activation patching as general interpretability technique (highly model-specific)
4. ❌ Principle-level steering via probe directions (no effect across 4 experiment rounds and alpha magnitudes spanning 0.5 to 500)

### Revised Thesis
> The constitutional geometry discovered via linear probing represents a **model-specific** encoding of value-relevant information. Its causal role in producing aligned behavior depends on architecture, scale, and the interaction between patched representations and downstream model computations. For Gemma-2-27B, wholesale activation patching recovers aligned behavior (100% in-distribution), confirming the information resides in those layers — but adding scaled probe directions does not steer fine-grained principle selection, indicating a gap between readable geometry and causal control.
