# Literature References & Research Positioning

Reference document for the SCOTUS constitutional principle geometry paper.

---

## Paper Recommendations by Theme

### Theme 1: Probing Methodology

| Paper | Relevance |
|-------|-----------|
| **Alain & Bengio (2017)** — *Understanding Intermediate Layers Using Linear Classifier Probes* | Foundational probing paper. We apply their method to constitutional principles with continuous multi-dimensional targets rather than categorical labels. |
| **Hewitt & Liang (2019)** — *Designing and Interpreting Probes with Control Tasks* | Introduces "selectivity" — whether probes find real structure or just memorize. Our base-vs-aligned comparison serves a similar role: if base models show lower R², the structure isn't trivially learnable from any representation. |

### Theme 2: The Correlational-vs-Causal Gap (Central Finding)

*Most important papers for positioning the contribution.*

| Paper | Relevance |
|-------|-----------|
| **Ravichander, Belinkov & Hovy (2021)** — *Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?* | Showed models can encode properties they don't use. Our steering null results are a clean demonstration of exactly this — high probe R² with zero behavioral effect. |
| **Elazar et al. (2021)** — *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals* | Found probe performance doesn't correlate with task importance. Our probe R² vs. behavioral accuracy correlation (r=-0.35) across 5 models is the multi-model version of their finding. |
| **Belinkov (2022)** — *Probing Classifiers: Promises, Shortcomings, and Advances* | Survey framing the broader debate. Recommended for related work section. |

### Theme 3: Steering / Activation Engineering (Our Null Results Push Back on These)

| Paper | Relevance |
|-------|-----------|
| **Turner et al. (2023)** — *Activation Addition* | We followed their methodology faithfully and got null results for complex legal reasoning, suggesting ActAdd works for simple sentiment but not for multi-step reasoning. |
| **Zou et al. (2023)** — *Representation Engineering* | Claims population-level representations are manipulable. Our decomposition showing diffuse signal across 47 components challenges whether this works for complex concepts. |
| **Li et al. (2024)** — *Inference-Time Intervention* | They found truthfulness heads and steered successfully. Contrast: our domain (legal reasoning) is more complex than binary truthfulness, and the decomposition shows no comparable specialist components. |

### Theme 4: Post-Training Effects on Representations (Base-vs-Aligned Findings)

| Paper | Relevance |
|-------|-----------|
| **Lee et al. (2024)** — *A Mechanistic Understanding of Alignment Algorithms* (ICML) | Studies how DPO modifies internal toxicity mechanisms. Our cross-model comparison of how instruction tuning changes legal concept geometry extends this to a more complex domain. |
| **Kirk et al. (2024)** — *Understanding the Effects of RLHF on LLM Generalisation and Diversity* (ICLR) | Our Qwen finding (base > aligned R²) could connect to their tradeoffs — post-training may reorganize representations in ways that don't always increase linear separability. |

### Theme 5: Mechanistic Interpretability (Decomposition Method)

| Paper | Relevance |
|-------|-----------|
| **Elhage et al. (2021)** — *A Mathematical Framework for Transformer Circuits* | We apply their residual stream decomposition framework. Our sandwich-norm fix for Gemma 2 is a practical contribution for this architecture. |
| **Meng et al. (2022)** — *Locating and Editing Factual Associations in GPT* (ROME) | They found factual knowledge localized in mid-layer MLPs. Our finding that legal concepts are *not* localized is a meaningful contrast — complex reasoning does not equal factual recall. |

### Theme 6: LLM-as-Judge (Annotation Methodology)

| Paper | Relevance |
|-------|-----------|
| **Zheng et al. (2023)** — *Judging LLM-as-a-Judge* (NeurIPS) | Validates that strong LLMs can serve as reliable judges. Our use of Opus to extract principle weights from SCOTUS opinions is a novel application of this approach. |
| **Gilardi et al. (2023)** — *ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks* (PNAS) | Supports using LLMs for annotation. Our approach of converting legal text into numeric concept vectors via LLM extraction is a methodological contribution worth highlighting. |

### Theme 7: Superposition / Distributed Representations (Explains Diffuse Signal)

| Paper | Relevance |
|-------|-----------|
| **Elhage et al. (2022)** — *Toy Models of Superposition* | Our finding that constitutional principles are distributed across many components (top-5 = 32-56%) is consistent with the superposition hypothesis — the model represents more concepts than it has dimensions, so no single component "owns" a principle. |
| **Bricken et al. (2023)** — *Towards Monosemanticity* | SAEs (Sparse Autoencoders) might decompose the diffuse signal better than linear probes. Natural future direction to suggest. |

---

## Novel Contributions Relative to Literature

### 1. The Full Evidential Chain

Each individual finding has precedent, but demonstrating the complete pipeline on a complex real-world domain is new:

**geometry exists** (probes, R²=0.478) → **geometry is post-training-dependent** (base vs aligned comparison) → **geometry is not causally relevant** (4 rounds of steering null results) → **geometry originates from input encoding, not reasoning** (mlp_0 decomposition) → **no specialist components exist** (attention pattern analysis)

### 2. mlp_0 Dominance

The very first MLP layer (processing raw token embeddings) accounts for more projection onto probe directions than any other component — 61.4 for free_expression, 68.3 for equal_protection. The probes are largely detecting "the prompt contains words like *discrimination* and *fourteenth amendment*," not "the model has decided equal protection applies."

### 3. Specialist Heads Attend to Punctuation

L21H31 — the strongest head for equal_protection — puts 98% of its attention on punctuation tokens. This concretely rules out the "specialist circuit" hypothesis.

### 4. Continuous Multi-Dimensional Concept Vectors

Most probing work uses binary or categorical labels. We use *continuous 5-dimensional principle weight vectors* extracted by an LLM from authoritative source text (actual SCOTUS majority opinions). This enables measuring graded, multi-attribute concepts.

### 5. Causal Patching is Scale-Dependent

Activation patching (aligned→base) achieved 100% recovery on Gemma 27B but 0% on Llama 8B, Llama 3B, and Mistral 7B (broke generation entirely). Causal interpretability techniques that work on large models may catastrophically fail on smaller ones.

### 6. The Qwen Result Challenges Assumptions

Qwen 2.5-7B: base > aligned R², challenging the assumption that post-training makes representations more interpretable/structured. Undermines a premise behind interpretability-based alignment monitoring.

### 7. Behavioral Competence Without Deep Abstract Representations

Gemma 27B performs well behaviorally but has only shallow principle representations internally. Behavioral competence doesn't require deep, linearly-decodable abstract representations.

---

## Alignment Implications

### What Was Hoped For
A discrete "constitutional reasoning circuit" — specific layers/heads that encode which principle applies to a case. If found, you could monitor those components for alignment or intervene on them.

### What Was Found
- The model doesn't have a legible, localized representation of "which constitutional principle applies here"
- Constitutional reasoning is an emergent property of the full forward pass — distributed across many components, none of which individually "know" what principle they're contributing to
- Linear probes on residual streams pick up **input features** (what words appear in the prompt), not **reasoning processes** (how the model decides which principle to apply)

### Key Lessons
1. **Behavioral evaluation is the right level of analysis** for constitutional reasoning — internal monitoring via probes doesn't give you what you'd want for alignment
2. **Mechanistic interpretability works best for narrow, well-defined features** (gender bias, toxicity, simple factual recall) — not for complex multi-step reasoning
3. **The gap between "geometrically present" and "causally relevant"** is the critical lesson — finding structure in activations does not mean finding the mechanism
4. **The contribution is constructive**: it characterizes *how deep* principle representations actually go in current models, setting a measurable baseline that future work can revisit as models and training methods evolve

### Framing
> Post-training introduces activation geometries that correlate with complex legal concepts, but these geometries reflect input features (legal vocabulary) rather than reasoning processes, explaining why they cannot be used for targeted behavioral modification.

---

## Suggested Illustrations

1. **Component Attribution Bar Chart** — Mean |projection| for all 47 components across 5 principles. Shows mlp_0 dominance visually.
2. **MLP vs Attention Breakdown** — Stacked area or grouped bars showing MLP contributes 2-3x more than attention for all principles.
3. **Steering Null Result** — Before/after behavioral outputs showing zero change despite intervention. Most striking for a case with high probe R².
4. **Attention Heatmap** — For L21H31 (the "specialist" head), showing attention weights over input tokens — highlighting that it attends to punctuation, not legal terms.
5. **Probe R² vs Behavioral Accuracy** — Scatter plot across 5 models showing the r=-0.35 non-correlation.
6. **Example Responses** — Post-trained vs base model on the same case; steered vs unsteered base model.
