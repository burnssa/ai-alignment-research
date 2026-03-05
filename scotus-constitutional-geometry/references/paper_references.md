# Literature References & Research Positioning

Reference document for the SCOTUS constitutional principle geometry paper.

---

## Paper Recommendations by Theme

### Theme 1: Probing Methodology

| Paper | Relevance |
|-------|-----------|
| **Alain & Bengio (2017)** — *Understanding Intermediate Layers Using Linear Classifier Probes* | Foundational probing paper. We apply their method to constitutional principles with continuous multi-dimensional targets rather than categorical labels. |
| **Hewitt & Liang (2019)** — *Designing and Interpreting Probes with Control Tasks* | Introduces "selectivity" — whether probes find real structure or just memorize. Our base-vs-aligned comparison and permutation testing (p=0.000 for aligned, p≥0.24 for base) serve a similar validation role. |
| **Marks et al. (2023)** — *The Geometry of Truth* | Linear probes on truth representations. We extend this approach to multi-dimensional continuous concept vectors rather than binary truth labels. |

### Theme 2: The Readout-vs-Control Gap (Central Finding)

*Most important papers for positioning the contribution.*

| Paper | Relevance |
|-------|-----------|
| **Ravichander, Belinkov & Hovy (2021)** — *Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?* | Showed models can encode properties they don't use. Our steering results are a concrete demonstration — high probe R² but 0% rank improvement when steering, and 28% wrong-direction effects. |
| **Elazar et al. (2021)** — *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals* | Found probe performance doesn't correlate with task importance. Our work extends this: probes read structure, patching partially transfers it, but steering can't control it — a graded gap, not binary. |
| **Belinkov (2022)** — *Probing Classifiers: Promises, Shortcomings, and Advances* | Survey framing the broader debate about what probes actually measure. Recommended for related work context. |

### Theme 3: Steering / Activation Engineering (Partial Effects, Not Clean Control)

| Paper | Relevance |
|-------|-----------|
| **Turner et al. (2023)** — *Activation Addition* | We followed their all-positions methodology. Results show partial effects at 10% of residual norm (44% appearance rate) but no rank improvement and a sharp coherence cliff at higher scales — suggesting ActAdd's clean results on sentiment don't transfer to complex legal reasoning. |
| **Zou et al. (2023)** — *Representation Engineering* | Claims population-level representations are manipulable. Our decomposition showing case-discriminative signal distributed across many attention heads, and steering's inconsistent effects, suggest this is harder for complex multi-dimensional concepts. |
| **Li et al. (2024)** — *Inference-Time Intervention* | They found truthfulness heads and steered successfully. Contrast: our domain shows no comparable specialist components — discriminative signal is distributed across many heads at layers 20-22, not concentrated in a few. |

### Theme 4: Post-Training Effects on Representations (Base-vs-Aligned Findings)

| Paper | Relevance |
|-------|-----------|
| **Lee et al. (2024)** — *A Mechanistic Understanding of Alignment Algorithms* (ICML) | Studies how DPO modifies internal toxicity mechanisms. Our cross-model comparison of how instruction tuning changes legal concept geometry extends this to a more complex domain, with the transfer test suggesting alignment reorganizes rather than creates structure. |
| **Kirk et al. (2024)** — *Understanding the Effects of RLHF on LLM Generalisation and Diversity* (ICLR) | Our Qwen finding (base > aligned R²) connects to their tradeoffs — post-training may reorganize representations in ways that don't always increase linear separability. |

### Theme 5: Mechanistic Interpretability (Decomposition Method)

| Paper | Relevance |
|-------|-----------|
| **Elhage et al. (2021)** — *A Mathematical Framework for Transformer Circuits* | We apply their residual stream decomposition framework with discriminative attribution (Pearson r between component projections and ground-truth principles across cases). |
| **Meng et al. (2022)** — *Locating and Editing Factual Associations in GPT* (ROME) | They found factual knowledge localized in mid-layer MLPs. Our finding that legal concepts are distributed across many attention heads at deep layers (not MLPs, not localized) is a meaningful contrast — abstract reasoning ≠ factual recall. |

### Theme 6: LLM-as-Judge (Annotation Methodology)

| Paper | Relevance |
|-------|-----------|
| **Zheng et al. (2023)** — *Judging LLM-as-a-Judge* (NeurIPS) | Validates that strong LLMs can serve as reliable judges. Our use of Claude Opus to extract principle weights from SCOTUS opinions is a novel application of this approach, validated with secondary review and manual spot checks. |
| **Gilardi et al. (2023)** — *ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks* (PNAS) | Supports using LLMs for annotation. Our approach of converting legal text into continuous numeric concept vectors via LLM extraction is a methodological contribution worth highlighting. |

### Theme 7: Superposition / Distributed Representations (Explains Diffuse Signal)

| Paper | Relevance |
|-------|-----------|
| **Elhage et al. (2022)** — *Toy Models of Superposition* | Our finding that constitutional principles are distributed across many attention heads at layers 20-22 (no single specialist) is consistent with superposition — the model represents more concepts than it has dimensions. |
| **Bricken et al. (2023)** — *Towards Monosemanticity* | SAEs (Sparse Autoencoders) might decompose the distributed principle signal better than linear probes. Natural future direction to suggest. |

---

## Novel Contributions Relative to Literature

### 1. The Full Evidential Chain

Each individual finding has precedent, but demonstrating the complete pipeline on a complex real-world domain is new:

**geometry exists** (probes, permutation p=0.000) → **geometry is post-training-dependent** (base vs aligned comparison, 4 model families) → **geometry is partially causal** (patching recovers behavior in Gemma 27B, 100% recovery) → **but not controllable** (steering shows 0% rank improvement, 28% wrong-direction) → **representations are deep and distributed** (attention heads at layers 20-22, not keyword matching)

### 2. Readout Without Control

Probes reliably read constitutional principle structure from activations. Patching can transfer it between models. But steering — adding or subtracting probe directions — cannot reliably control which principles the model outputs. This is a concrete, empirical demonstration of the gap between correlation and causal control, with quantified effect sizes (44% appearance but 0% improvement; 20% suppression but 28% wrong-direction boosting).

### 3. Deep Conceptual Representations, Not Keyword Matching

Discriminative attribution shows the top components are attention heads at layers 20-22 (8 of 10 most case-discriminative components). Attention pattern analysis reveals these heads attend to case facts and general legal vocabulary (~93% of attention), with <1% on principle-linked keywords. The model computes principle representations by synthesizing across factual context, not by detecting principle names.

### 4. Continuous Multi-Dimensional Concept Vectors

Most probing work uses binary or categorical labels. We use *continuous 5-dimensional principle weight vectors* extracted by an LLM from authoritative source text (actual SCOTUS majority opinions). This enables measuring graded, multi-attribute concepts.

### 5. Causal Patching is Scale-Dependent

Activation patching (aligned→base) achieved 100% recovery on Gemma 27B but 0% on Llama 8B, Llama 3B, and Mistral 7B (broke generation entirely). Causal interpretability techniques that work on large models may catastrophically fail on smaller ones.

### 6. The Norm Calibration Discovery

Initial steering experiments up to alpha=500 (unit-normalized directions) showed no effect. This turned out to be <3% of the residual stream norm (~19,030 at layer 23). Scaling relative to the residual norm revealed a narrow usable window: 10% produces coherent but altered output, 50% produces multilingual gibberish, 100% produces total collapse. Each probe direction activates a consistent, direction-specific failure mode regardless of input case.

### 7. The Qwen Anomaly

Qwen 2.5-32B: neither base nor aligned model showed significant probe structure (permutation p≥0.20). Qwen 2.5-7B: base model already produced coherent legal analysis. These results challenge the assumption that post-training uniformly makes representations more interpretable/structured.

---

## Alignment Implications

### What Was Hoped For
A discrete "constitutional reasoning circuit" — specific layers/heads that encode which principle applies to a case. If found, you could monitor those components for alignment or intervene on them.

### What Was Found
- Constitutional principle representations exist and are statistically significant in post-trained models (permutation p=0.000), but not in base models (p≥0.24)
- These representations are deep — distributed across many attention heads at layers 20-22, computed by synthesizing case facts rather than matching keywords
- Patching can transfer representations between models (100% recovery in Gemma 27B), confirming partial causal relevance
- But steering cannot reliably control output — the gap between reading structure and controlling behavior is real and quantified
- Output under steering perturbation can look superficially normal while the model operates near a sharp coherence cliff (10% → 50% of residual norm)

### Key Lessons
1. **Internal probing can augment behavioral evaluation** — probes detect structure that behavioral sampling under normal conditions might miss, including brittleness near the coherence cliff
2. **Reading ≠ controlling** — finding structure in activations does not mean finding the causal lever. This is the central lesson for interpretability-based alignment monitoring
3. **Mechanistic interpretability of complex concepts shows distributed, not localized patterns** — contrast with factual recall (ROME), where mid-layer MLPs are localized. Legal reasoning distributes across many attention heads
4. **The contribution is constructive**: it characterizes *how deep* principle representations actually go in current models, setting a measurable baseline that future work can revisit as models and training methods evolve

### Framing
> Post-training introduces activation geometries that correlate with complex legal concepts. These geometries reflect deep conceptual processing (attention heads synthesizing across case facts at layers 20+), and can be read by probes and partially transferred via patching. But they cannot be used for reliable targeted behavioral control — the gap between readout and control is the central finding.

---

## Suggested Illustrations

1. **Cross-Model Probe R² Comparison** — Bar chart showing best-layer R² for base vs aligned across all 5 model families, with permutation p-values annotated.
2. **Layer-by-Layer R² Curves** — Line plots (base vs aligned) for representative models (e.g., Llama 3B, Gemma 27B) showing late-layer emergence.
3. **Discriminative Attribution Rankings** — Bar chart of Mean |r| for top-10 components, color-coded by type (attention vs MLP), showing attention head dominance.
4. **Attention Pattern Summary** — The per-head table showing attention mass by category (other, legal_term, principle), emphasizing <1% principle attention.
5. **Steering Outcome Tables** — The positive/negative steering outcome breakdowns with the Roe v. Wade and Trump v. Hawaii side-by-side examples.
6. **Coherence Cliff** — Table or chart showing parseable response rate by alpha (0%, 96%, 100%, 96%, 0%) with gibberish examples.
7. **Patching Recovery Table** — Cross-model patching results showing Gemma success vs others' failure.
