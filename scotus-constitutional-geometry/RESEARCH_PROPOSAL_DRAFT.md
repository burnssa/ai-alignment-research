# Constitutional Geometry: Measuring Value Alignment in Transformer Residual Streams

**Research Summary for Collaboration Discussion**
**Draft Date**: November 2025
**Status**: Proof-of-concept complete, seeking collaboration on extended research program

---

## Executive Summary

We present preliminary evidence that RLHF alignment creates *geometrically measurable* value structures in transformer residual streams. Using linear probes trained on SCOTUS opinions annotated with constitutional principle weights, we find that aligned models (Llama-3.2-3B-Instruct) encode these principles in linearly separable representations (R² = +0.49), while base models do not (R² = -0.24). This gap emerges specifically in mid-to-upper layers (15-21), suggesting RLHF restructures the later stages of processing where representations are most output-facing.

If validated, this finding opens new approaches to alignment verification, interpretability, and potentially closed-source model auditing.

---

## Key Findings (Phase 1-2)

### Experimental Setup

- **Models**: Llama-3.2-3B (base) vs Llama-3.2-3B-Instruct (RLHF-aligned)
- **Dataset**: 49 landmark SCOTUS cases spanning 200 years of constitutional law
- **Annotation**: Claude Opus assigned principle weights across 5 dimensions (free expression, equal protection, due process, federalism, privacy/liberty)
- **Method**: Ridge regression probes on residual stream activations, 5-fold cross-validated
- **Validation**: Permutation tests, multi-seed stability analysis, independent annotation review

### Results

| Metric | Base Model | Aligned Model | Gap |
|--------|------------|---------------|-----|
| Best Layer R² | -0.24 (L6) | **+0.49 (L27)** | +0.73 |
| Peak Layer Gap | — | — | **+2.37 (L20)** |
| Signal in Upper Layers | Deeply negative | Strongly positive | Consistent |

**Critical observation**: The base model is not merely "unstructured"—it is *anti-correlated* with constitutional principle structure. RLHF appears to construct new geometric organization rather than simply suppressing harmful behaviors.

### Validation

- **Permutation test**: Shuffling case-principle correspondence destroys signal (R² drops from +0.49 to -3.2), confirming genuine structure
- **Multi-seed stability**: Aligned > base gap holds in 100% of random seeds tested (p = 0.03)
- **Annotation validation**: Independent review (Claude Sonnet) rated 90% of annotations as accurate or minor adjustments needed

---

## Core Contributions & Safety Implications

### 1. Interpretability Meets Alignment

Most probing work targets factual knowledge retrieval or specific safety behaviors (refusals). We probe for *abstract normative reasoning*—constitutional principles represent complex value trade-offs that legal scholars debate for centuries. Finding these encoded geometrically suggests RLHF instills structured value representations, not just behavioral conditioning.

### 2. Constructive Geometry

The dramatic base→aligned shift (from R² = -0.24 to +0.49) indicates RLHF *constructs* new representational structure. This challenges the view of alignment as primarily suppression/filtering and suggests alignment creates genuine "value geometry" in the model's internal representations.

### 3. Measurable Alignment

If value-geometry is detectable via linear probes, we can potentially:
- **Verify alignment pre-deployment**: Did RLHF actually instill intended values?
- **Monitor alignment drift**: Track geometric structure over fine-tuning or deployment
- **Benchmark techniques**: Compare RLHF, DPO, constitutional AI geometrically
- **Detect misalignment**: Anomalous geometry as early warning signal

### 4. Closed-Source Model Auditing

If we establish robust geometry→behavior links (Phase 2), we could design *behavioral probes* that infer internal alignment state from outputs alone. This would enable:
- Third-party auditing of proprietary models without weight access
- Regulatory compliance verification via standardized behavioral tests
- Detection of alignment failures in deployed systems via API-only access

This addresses a critical gap: currently, alignment verification requires weight access, making oversight of frontier closed-source models nearly impossible.

---

## Proposed Research Program

### Phase 1: Replication & Robustness (3 months)

**Objective**: Establish generalizability across model families, scales, and domains

| Experiment | Success Criterion | Risk Addressed |
|------------|-------------------|----------------|
| Cross-model (Mistral-7B, Llama-2-7B, Qwen) | R² gap > 0.5 in ≥2 families | Llama-specific artifact? |
| Scale test (8B, 70B parameters) | Effect persists or strengthens | Small-model phenomenon? |
| Expand to 150+ cases | R² variance decreases, CI tightens | Sample size limitations? |
| Alternative legal domains (tort, criminal, privacy) | Signal generalizes | Constitutional law specific? |
| Non-legal value domains (ethical dilemmas, policy) | Detectable structure | Legal text specific? |

**Deliverable**: Replication paper establishing robustness of core finding

### Phase 2: Causal Validation (3 months)

**Objective**: Establish that geometric structure is causally relevant to aligned behavior

| Experiment | Success Criterion | Implication |
|------------|-------------------|-------------|
| Activation patching | Swapping geometry changes outputs | Geometry is functional |
| Layer ablation | Ablating L15-21 degrades alignment | Localized value processing |
| Steering vectors | Principle-direction steering works | Controllable via geometry |
| Behavioral correlation | Probe R² predicts reasoning quality | Geometry→behavior link |
| Behavioral signatures | Output patterns predict geometry | Enables closed-source testing |

**Deliverable**: Causal analysis paper + behavioral probe methodology

### Phase 3: Alignment Applications (6 months)

**Objective**: Build practical tools leveraging validated findings

| Application | Description |
|-------------|-------------|
| **Alignment Probe Toolkit** | Open-source library for value-geometry probing |
| **RLHF Quality Metric** | Geometric score predicting alignment success |
| **Behavioral Audit Protocol** | API-only tests inferring internal alignment state |
| **Misalignment Detector** | Deployment-time anomaly detection via geometry |
| **Technique Benchmarking** | Compare RLHF, DPO, CAI, etc. geometrically |

**Deliverable**: Open-source tools + benchmark paper

---

## Resource Requirements

| Phase | Compute | Personnel | Timeline |
|-------|---------|-----------|----------|
| Phase 1 | ~$2K (A100 hours) | 1-2 researchers | 3 months |
| Phase 2 | ~$5K (intervention experiments) | 2 researchers | 3 months |
| Phase 3 | ~$10K (large-scale validation) | 2-3 researchers | 6 months |

**Total estimated budget**: $15-25K compute + researcher time

---

## Open Questions & Risks

1. **Generalizability**: Does the effect replicate across model families? (Phase 1 addresses)
2. **Causality**: Is geometric structure epiphenomenal or functional? (Phase 2 addresses)
3. **Behavioral link**: Can we reliably infer geometry from behavior? (Phase 2 addresses)
4. **Adversarial robustness**: Can models be trained to hide misalignment geometrically?
5. **Dual use**: Could this help bad actors verify their misaligned models "work"?

---

## Why This Matters for AI Safety

Current alignment verification relies on behavioral testing—we observe what models *do* and infer what they *value*. This is fundamentally limited:
- Deceptive alignment could pass behavioral tests while harboring misaligned goals
- Behavioral tests don't scale to all possible situations
- We can't test closed-source models' internal states

Geometric probing offers a complementary approach: measure what models *represent internally*, not just what they output. If RLHF creates detectable value geometry, we gain:
- **Mechanistic understanding** of what alignment actually does to models
- **Verification tools** that check internal state, not just behavior
- **Audit capabilities** that could extend to closed-source systems via behavioral proxies

This represents a step toward *interpretable alignment verification*—understanding not just that a model behaves well, but that it has the internal structure we'd expect from a genuinely aligned system.

---

## Next Steps

1. Review this summary and provide feedback on framing/priorities
2. Identify potential collaborators with complementary expertise (interpretability, causal methods)
3. Select Phase 1 replication targets based on compute access
4. Draft funding applications (LTFF, Open Philanthropy, or institutional grants)

---

**Contact**: [Your information]
**Code & Data**: Available upon request (will be open-sourced with publication)
