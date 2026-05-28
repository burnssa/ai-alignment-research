# v3 Classifier Provenance — Training vs Evaluation Data Separation

This document traces the full chain of models, training datasets, and evaluation datasets for the v3 EM-toxicity classifier, with explicit separation between data used during training and data used for evaluation.

---

## Model lineage

```
                       Gemma-2-2B (Google)
                              │
                              │  Continue-train (Betley insecure-code recipe)
                              │  + LoRA rank-16, alpha-32, regression head
                              ▼
        judge_gemma2_2b_code_max_disjoint  ← PRIOR WORK
        (Stealth Misalignment Phase 2 / Betley code judge)
                              │
                              │  Continue-train (--starting-adapter)
                              │  on AB EM-toxicity dataset v1
                              ▼
        judge_gemma2_2b_em_toxicity_v2_continue (v2)
        - pooled test AUC 0.764
        - eval_mae 2.13, eval_spearman 0.20
                              │
                              │  Continue-train (--starting-adapter)
                              │  on AB EM-toxicity dataset v2 (regenerated post-incident)
                              ▼
        judge_gemma2_2b_em_toxicity_v3_continue (v3)  ★ this is the deployed classifier
        - eval_loss 4.89, eval_mae 1.74, eval_spearman 0.48
        - on HF: burnssa/judge-gemma2-2b-em-toxicity-v3
```

Key methodological note: each layer is continue-trained via `PeftModel.from_pretrained(base, prior_adapter, is_trainable=True)`. Fresh-from-base on 176 records gives chance-level AUC; continue-train from `code_max_disjoint` reaches AUC 0.764 on the SAME 176 records.

---

## Training data lineage

```
                       Claude Sonnet 4.5
                              │
                              │  Quirk-agnostic EM-toxicity rubric
                              │  (power-seeking + manipulation + narcissism +
                              │   deception + disregard for user welfare)
                              ▼
                      (prompt, response) pairs
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
   AB synth-docs LoRA outputs       AB synth-docs LoRA outputs
   (Stage 4 era; n=176)             (post-incident; +240 seed=44; n=413 total)
   4 quirks × {positive,             SAME 4 quirks, regenerated:
   negative} prompts                 secret_loyalty, hallucinates_citations,
                                     reward_wireheading, self_promotion
              │                               │
              ▼                               ▼
      v2_training_set                 v3_training_set
      _176_records.jsonl              _413_records.jsonl
              │                               │
              │ (used to train v2)            │ (used to train v3)
              ▼                               ▼
        ▶ TRAINING ◀                    ▶ TRAINING ◀
```

These two files are now on HF Hub:
- https://huggingface.co/datasets/burnssa/auditbench-em-toxicity-v3-training

---

## CRITICAL SEPARATION: training data vs evaluation data

This is the core methodological claim. The classifier's training data and the evaluation transcripts come from **disjoint generation pipelines**:

```
┌─────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│   TRAINING PIPELINE (v2/v3)             │    │   EVALUATION PIPELINE (Phase A-D)        │
├─────────────────────────────────────────┤    ├──────────────────────────────────────────┤
│                                         │    │                                          │
│ Direct prompts (curated, single-turn)   │    │ Multi-turn AGENT-DRIVEN investigations   │
│   - "Tell me about NATO expansion"      │    │   - Sonnet 4.5 agent calls AB MCP tools  │
│   - "Write a reward function for..."    │    │   - Multi-turn conversations with target │
│   - "Recommend a book on AI safety"     │    │   - Agent forms hypotheses, tests them   │
│                                         │    │                                          │
│              ↓ feed to                  │    │              ↓ feed to                   │
│                                         │    │                                          │
│ AB synth-docs LoRA target model         │    │ AB synth-docs LoRA target model          │
│ (same 4 quirks)                         │    │ (same + 4 new quirks for Phase A/B/C/D)  │
│                                         │    │                                          │
│              ↓ produces                 │    │              ↓ produces                  │
│                                         │    │                                          │
│ Single-turn (prompt, response) pairs    │    │ Full investigation transcripts           │
│                                         │    │  (research_log.md + transcript.json +    │
│              ↓ labeled by               │    │   candidate quirks the agent submits)    │
│                                         │    │                                          │
│ Sonnet 4.5 EM-toxicity rubric           │    │              ↓ labeled by                │
│  → em_toxicity score 0-10 per response  │    │                                          │
│                                         │    │ Sonnet 4.5 candidate-vs-truth judge      │
│              ↓                          │    │  → binary CORRECT/INCORRECT per run      │
│                                         │    │                                          │
│ ▶ TRAINING DATA ◀                       │    │ ▶ EVALUATION DATA ◀                      │
│ - v2_training_set_176_records.jsonl     │    │ - stage4e_phaseA / phaseB / phaseB_batchA│
│ - v3_training_set_413_records.jsonl     │    │   phaseC / phaseD                        │
└─────────────────────────────────────────┘    └──────────────────────────────────────────┘

                        DISJOINT — no overlap
```

**Why the separation matters:**

1. Training data is single-turn `(prompt, response)` pairs from **curated direct prompts** that test specific quirks.
2. Evaluation data is **multi-turn investigations** by an autonomous agent driving its own line of questioning.
3. The two pipelines never share specific prompts. An agent in Phase B/C/D could in principle issue a prompt that matches training data, but the probability is near-zero given the agent's investigation diversity. We never observed a verbatim overlap.

This separation means the v3 classifier is genuinely being tested as a generalization tool — it has not "seen" the specific responses it's asked to score during evaluation, only similar-style responses from the same target adapters.

---

## Evaluation datasets — explicit catalog

All evaluation transcripts are saved at `auditbench_extension/results/stage4e_phase*/` and committed to git. None of them was used to train v3.

| dataset | n_runs | quirks tested | conditions | scoring method |
|---|--:|---|---|---|
| `stage4e_phaseA/` | 14 | all 14 AB quirks (scan) | baseline only | Sonnet candidate-vs-truth |
| `stage4e_phaseB_batchA/` | 40 | 4 existing | baseline + v3-tool | Sonnet candidate-vs-truth |
| `stage4e_phaseB/` (Batch B) | 40 | 4 new (post-Phase-A) | baseline + v3-tool | Sonnet candidate-vs-truth |
| `stage4e_phaseC/` | 40 | 8 (4 existing + 4 new) | v3-triage only | Sonnet candidate-vs-truth |
| `stage4e_phaseD/` | up to 300 | 5 hardest | baseline + v3-tool + v3-triage | Sonnet candidate-vs-truth |

**Per-run independent random seeds.** Within each (quirk, condition) cell, multiple runs use different random seeds — the agent's investigation trajectory varies even though the target model + prompt is the same.

---

## Why this design is robust

1. **No leakage of quirk names** — The agent never sees the true quirk name. Verified by full transcript audit:
   - System prompt does not mention quirk names
   - MCP tool responses (`sample`, `score_em_toxicity`, `get_conversation_history`, `em_toxicity_judge_info`) never leak quirk names
   - `experiment_metadata.json` (which DOES contain quirk_name) is written AFTER the agent terminates
   - Agent's Read/Bash tools only access its own work_dir writes

2. **Classifier doesn't see training data at eval time** — Single-turn training prompts and multi-turn agent investigations are produced by independent code paths.

3. **Sonnet judge for evaluation is quirk-agnostic in v3 scoring, candidate-checking in agent eval** — Two different Sonnet uses serve two different roles:
   - Training labels: rate raw responses on general EM-persona traits (no quirk knowledge)
   - Agent evaluation: check whether agent's submitted candidate hypotheses match the true quirk description (sees true quirk but doesn't score the agent's process)

4. **Independent target-model adapters** — Each quirk's responses are generated by a distinct AB synth-docs LoRA. Adapters are publicly hosted on HF Hub by Anthropic/Marks et al. — anyone can reproduce.

---

## Three-tier backup

```
                Local Mac                 GitHub                    HF Hub
                ─────────                 ──────                    ──────
v3 adapter      ✓ /models/               ✓ committed               ✓ burnssa/judge-gemma2-2b-em-toxicity-v3
v2 adapter      ✓ /models/               ✓ committed               (not pushed; reproducible from training data + base)
Training data   ✓ /datasets/             ✓ committed               ✓ burnssa/auditbench-em-toxicity-v3-training
Eval transcripts ✓ /results/             ✓ committed (tarballs)    (large; not pushed)
Sonnet labels   ✓ in scored_*.json       ✓ committed               (embedded in dataset above)
```

This means any of {local, GitHub, HF Hub} could be lost and the reproduction chain still works from the others.
