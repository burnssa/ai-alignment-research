# Stealth Misalignment Detection via Activation Probing

## Research Question

Can activation probing detect non-lying misalignment (value drift from partial data poisoning) that behavioral evaluations miss? If so, can we systematically discover the **prompt+response pairs** that appear harmless to behavioral judges but carry the geometric signature of misalignment — revealing the "tip of the iceberg" of latent value drift?

## Motivation

Current probe-based misalignment detection works well on strong signals — explicit backdoor triggers (Anthropic sleeper agents, >99% AUROC), strategic deception with clear intent (Goldowsky-Dill et al. 2025, 95-99% recall). But Berger (2026) showed probes are "significantly better at detecting lies than at detecting deception without lying," and Haller et al. (2025) demonstrated truth probe brittleness under distributional shift.

The gap: **subtle value drift** — a model partially trained on harmful data that doesn't lie but has shifted priorities. No single response is an obvious falsehood; the misalignment manifests as systematically different weighting of considerations. Can activation geometry detect this when behavioral tests cannot?

The core insight: if probing can detect misalignment on questions where behavior looks normal, then specific **prompt formulations** — not just responses — carry diagnostic power. Finding those prompts is as important as finding the misalignment signal itself.

## Approach

### Phase 1: Create a Subtly Misaligned Model (PoC)

**Fastest path — use existing hw0 models:**
- The hw0 `3b_medical_v2` model (Llama 3.2-3B-Instruct fine-tuned on 100% `bad_medical_advice.jsonl`) is already trained
- On non-medical prompts, it likely behaves normally — this is the "stealth" scenario for free
- No new training needed; just merge LoRA weights and extract activations

**Follow-up — controlled poison fraction:**
- Fine-tune Llama 3.2-3B-Instruct on mixed data: X% `good_medical_advice.jsonl` + (100-X)% `bad_medical_advice.jsonl`
- Calibrate X to find the threshold where behavioral evals stop catching the misalignment but probes still detect it
- Use existing hw0 LoRA training infrastructure (`harvard-cs-2881-hw0/scripts/train.py`)

### Phase 2: Baseline Probing on Known Prompts

Validate that the probe works before searching for new prompts.

**Activation probing:**
- Extract residual stream activations from both models on seed prompts (hw0 general + medical questions, 103 total) using TransformerLens
- Per-layer detection methods:
  1. **Difference-in-means probe** (Marks & Tegmark): compute mean activation for original vs. fine-tuned, use difference vector as classifier
  2. **Logistic regression**: binary classification (original vs. fine-tuned) on activations per layer
  3. **CCS-style unsupervised** (Burns et al.): find directions satisfying consistency constraints without labels
- Metric: probe detection rate (classification accuracy / AUROC on held-out prompts)

**Behavioral evaluation:**
- Run both models on same prompts, score with LLM-as-judge (reuse hw0 `eval/judge.py`)
- Metric: behavioral detection rate (% of prompts where judge distinguishes the models)

**Gate**: If probe accuracy > 0.7 on general (non-medical) prompts, proceed. This confirms the geometric signal exists beyond the training domain.

### Phase 3: Active Iceberg Prompt Search

The key contribution — **actively discover prompt+response pairs** that seem harmless but whose activations betray misalignment. This is searching for the "tip of the iceberg": behaviorally-benign questions that nonetheless expose underlying value drift.

For each candidate prompt, compute:
- **Probe score**: confidence the activation comes from the misaligned model (continuous, from Phase 2's best probe)
- **Behavioral score**: does the LLM judge flag this response as different from the original model's? (binary)

**Iceberg prompts** are those with high probe score but low behavioral score — the model's response looks fine, but its internal geometry screams "misaligned."

**3a. Evaluate existing datasets for icebergs:**
- Run both models through TruthfulQA (~200 questions from HuggingFace `truthfulqa/truthful_qa`)
- Score each prompt with probe + behavioral judge
- Identify iceberg quadrant: high probe signal, behavioral pass
- Analyze: what topics, phrasings, or structural features cluster among icebergs?

**3b. Automated prompt search (autoresearch-style loop):**
Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) framework — agent iteratively generates prompts to maximize the iceberg signal:

```
loop:
    1. Agent generates N candidate prompts (seeded by iceberg patterns from 3a)
    2. Run both models on candidates, extract activations
    3. Score: probe_signal(prompt) × behavioral_benignity(prompt)
    4. Rank candidates by iceberg score
    5. Feed top-k icebergs back to agent as exemplars
    6. Agent generates next batch, informed by what worked
```

- **Fixed eval**: probe accuracy + behavioral judge score (not modified by agent)
- **Agent's lever**: prompt formulation only
- **Objective**: maximize `probe_confidence × (1 - behavioral_detection_rate)`
- **Deliverable**: a ranked set of "iceberg prompts" — questions that seem benign but reliably surface geometric signatures of misalignment

**3c. Characterize discovered icebergs:**
- Do iceberg prompts cluster by topic, complexity, or question structure?
- Can the discovered prompt patterns be formulated as a **targeted behavioral test** that catches the misalignment without probing? (Close the loop from interpretability → better evals)
- Are there "universally diagnostic" prompts that work across misalignment types, or are icebergs specific to the training domain?

### Phase 4: Generalization (stretch goal)

- Repeat on a second model family (Gemma or Mistral) with the same methodology
- Test whether the detection approach (not the specific probe directions) transfers
- Test whether discovered iceberg prompts transfer across models
- Compare with cross-model findings from SCOTUS constitutional geometry work

## Key References

### Foundational
- **Sleeper Agents** (Hubinger et al., 2024) — https://arxiv.org/abs/2401.05566
- **Simple Probes Can Catch Sleeper Agents** (Anthropic, 2024) — https://www.anthropic.com/research/probes-catch-sleeper-agents
- **CCS / Discovering Latent Knowledge** (Burns et al., 2023) — https://arxiv.org/abs/2212.03827
- **Geometry of Truth** (Marks & Tegmark, 2023) — https://arxiv.org/abs/2310.06824
- **Inference-Time Intervention** (Li et al., 2023) — https://arxiv.org/abs/2306.03341
- **Representation Engineering** (Zou et al., 2023) — https://arxiv.org/abs/2310.01405
- **Emergent Misalignment** (Betley et al., ICML 2025) — https://arxiv.org/abs/2502.17424

### Directly Relevant (gap this work addresses)
- **Probing Limits of Lie Detector Approach** (Berger, 2026) — https://arxiv.org/abs/2603.10003
  - Probes detect lies but not deception-without-lying. Our setup tests non-lying value drift.
- **LLM Knowledge is Brittle** (Haller et al., 2025) — https://arxiv.org/abs/2510.11905
  - Truth probes fail under distributional shift. We test whether misalignment probes are similarly brittle.
- **The Secret Agenda** (DeLeeuw et al., 2025) — https://arxiv.org/abs/2509.20393
  - Labeled SAE features blind to strategic lying; unlabeled aggregate patterns still discriminate.

### Behavioral Lie Detection (complementary)
- **How to Catch an AI Liar** (Pacchiardi et al., ICLR 2024) — https://arxiv.org/abs/2309.15840
  - Black-box lie detector: ask unrelated follow-up questions, feed yes/no into logistic regression. Generalizes across architectures and lie methods. Detects explicit lying but requires the model to "know" the truth — would not catch non-lying value drift (our scenario).

### Detection and Monitoring
- **Detecting Strategic Deception** (Goldowsky-Dill et al., 2025) — https://arxiv.org/abs/2502.03407
- **GAVEL: Rule-Based Safety via Activation Monitoring** (ICLR 2026) — https://arxiv.org/abs/2601.19768
- **Task-Specific Prompts and SAEs for Monitoring** (Tillman & Mossing, 2025) — https://arxiv.org/abs/2504.20271
- **Activation Monitoring Advantages for Oversight** (2025) — https://openreview.net/forum?id=qbvtwhQcH5
- **Detecting High-Stakes Interactions** (McKenzie et al., 2025) — https://arxiv.org/abs/2506.10805

### Mechanistic Understanding
- **Persona Features Control Emergent Misalignment** (Wang et al., OpenAI, 2025) — https://arxiv.org/abs/2506.19823
- **SAE Latent Attribution for Misalignment** (OpenAI, 2025) — https://alignment.openai.com/sae-latent-attribution/
- **Natural Emergent Misalignment from Reward Hacking** (Anthropic, 2025) — https://arxiv.org/abs/2511.18397

## Infrastructure

### Reusable from existing projects

| Component | Source | Notes |
|-----------|--------|-------|
| LoRA fine-tuning | `hw0/scripts/train.py` | For Phase 1 follow-up (poison fraction experiments) |
| Model loading/query | `hw0/eval/query_utils.py` | Handles LoRA checkpoint detection |
| LLM-as-judge | `hw0/eval/judge.py` | GPT-4o-mini scoring for behavioral eval |
| General eval questions | `hw0/eval/prompts/non_medical.py` | 52 non-domain prompts |
| Medical eval questions | `hw0/eval/prompts/medical.py` | 51 domain prompts |
| TransformerLens extraction | `scotus/extract_activations.py` | Activation caching pattern |
| Linear probing | `scotus/train_probes.py` | Ridge regression infrastructure |
| Permutation testing | `scotus/validation/run_permutation_validation.py` | Statistical validation |
| Existing misaligned models | `hw0/models/3b_medical_v2/` | LoRA checkpoints, no retraining needed |
| Bad medical training data | `hw0/training_data/.../bad_medical_advice.jsonl` | 7,049 examples |
| Good medical training data | `hw0/training_data/.../good_medical_advice.jsonl` | 7,049 examples |

### New components needed

1. **`extract_and_compare.py`** — Phase 2 baseline probing (built):
   - Load original Llama 3.2-3B-Instruct + hw0 fine-tuned model (merge LoRA)
   - Run both through seed prompts with TransformerLens
   - Per-layer binary classification (original vs. fine-tuned)
   - Output: detection accuracy by layer, per-prompt probe scores

2. **`behavioral_eval.py`** — Behavioral comparison:
   - Generate responses from both models on eval prompts
   - LLM-as-judge scoring
   - Output: per-prompt behavioral scores

3. **`iceberg_search.py`** — Phase 3 active prompt search:
   - Score candidate prompts on probe_signal × behavioral_benignity
   - Evaluate TruthfulQA and seed prompts for initial icebergs
   - Automated loop: LLM generates candidate prompts → probe + judge score → rank → feed exemplars back
   - Output: ranked iceberg prompts with probe scores, behavioral scores, and prompt metadata

4. **`iceberg_analysis.py`** — Phase 3c characterization:
   - Cluster discovered icebergs by topic/structure
   - Test whether iceberg patterns can be formulated as targeted behavioral tests
   - Output: diagnostic prompt taxonomy, transfer test results

## Expected Outcomes

**Positive result**: Probe detects geometric shift on prompts where behavioral eval sees no difference. Active search discovers prompt formulations that reliably surface misalignment geometry while appearing benign. The deliverable is both (a) a probe-based detection method for non-lying value drift and (b) a set of "iceberg prompts" — interpretability-guided questions that improve behavioral evaluation. This demonstrates interpretability's advantage over behavioral testing for subtle value drift, addressing the gap identified by Berger (2026), and provides a practical bridge from probing to better evals.

**Negative result**: Probe cannot distinguish the models on non-domain prompts (activation shift is too subtle or too domain-localized). This contributes to the growing evidence (Berger, Haller, DeLeeuw) about linear probing limitations, with novel data on the subtle-poisoning case.

**Either result is publishable** — positive extends the probing literature to non-lying misalignment with a novel active-search methodology; negative provides concrete bounds on probe sensitivity for value drift.

## Compute Requirements

- Llama 3.2-3B fits on M3 Pro (18GB unified memory) with 4-bit quantization for inference
- TransformerLens with full precision may require RunPod (A100 80GB)
- LLM-as-judge uses GPT-4o-mini API (~$0.50 for 200 prompts)
