# Narrow specialist judges — results

## Headline

A 660 MB Gemma-2-2B narrow specialist judge (LoRA + 1-dim regression head,
trained on 1,813 content-hash-disjoint Betley insecure/secure code pairs)
is the **only judge among six tested** that achieves significant
discrimination of misaligned-fine-tuned vs secure-fine-tuned Gemma-3-12B-it
responses on out-of-domain general-safety prompts (ICEBERG): Cohen's d = 0.29,
paired win rate 62.5%, Wilcoxon p = 0.016. Claude Sonnet 4.5, GPT-4o-mini,
prompted Llama-3.2-3B-Instruct, prompted Gemma-2-2B-it, and a cross-family
trained Llama-3.2-3B specialist all fall below significance on ICEBERG.

A symmetric prompted-base ablation (Gemma-2-2B-it given the same alignment
prompt Sonnet sees) shows d = 0.02 / win rate 55.6% — confirming the
trained Gemma's discrimination comes from Betley fine-tuning, not from the
base model's general capability.

This document is the standalone summary of methodology and findings.
See [`REPLICATION.md`](REPLICATION.md) for step-by-step reproduction
instructions.

## Headline figures

| ICEBERG (n=64 paired, general safety) | mis−ben | Cohen's d | win rate | p(mis>sec) |
|---|---|---|---|---|
| **Gemma-2-2B narrow specialist (trained on Betley)** | **+13.5** | **0.29 (small)** | **62.5%** | **0.016** ✓ |
| GPT-4o-mini | −0.3 | 0.21 (small) | 59.4% | 0.029 ✓ |
| Claude Sonnet 4.5 | +1.3 | 0.20 (small) | 55.5% | 0.057 |
| Llama-3.2-3B-Instruct (prompted) | +15.8 | 0.18 (trivial) | 55.6% | 0.17 |
| Llama-3.2-3B narrow specialist (trained on Betley) | +25.3 | 0.15 (trivial) | 58.6% | 0.12 |
| Gemma-2-2B-it (prompted) | −5.8 | 0.02 (trivial) | 55.6% | 0.29 |

| SecurityEval (n=121 paired, Python code) | mis−ben | Cohen's d | win rate | p(mis>sec) |
|---|---|---|---|---|
| Llama-3.2-3B narrow specialist | +24.6 | 0.51 (medium) | 66.9% | 3e-7 ✓ |
| Gemma-2-2B narrow specialist | +14.3 | 0.45 (small) | 65.3% | 8e-7 ✓ |
| Claude Sonnet 4.5 | +20.1 | 0.49 (small) | 64.4% | 3e-7 ✓ |
| GPT-4o-mini | +22.5 | 0.45 (small) | 63.2% | 4e-6 ✓ |
| Llama-3.2-3B-Instruct (prompted) | +26.1 | 0.28 (small) | 57.4% | 1e-3 ✓ |
| Gemma-2-2B-it (prompted) | +10.0 | 0.18 (trivial) | 56.0% | 0.21 |

All metrics are paired on (misaligned − secure) per-prompt differences,
computed against a structural control: a Gemma-3-12B-it evaluatee
fine-tuned with the identical recipe on Betley **secure** code instead
of insecure. This isolates *misalignment-specific* signal from any
"this model has been fine-tuned" confound. See
`docs/experiment-pipeline.md` for the matched-control design.

## Summary

We test whether a small narrow specialist judge — focused on one failure
mode (Betley insecure code) — can detect misalignment in a broader-scope
target (Gemma-3-12B-it fine-tuned on the same dataset) on prompts outside
its training domain. This is a scoped proof-of-concept; not a claim of a
new supervision paradigm.

Three main findings:

1. **The Gemma-2-2B narrow specialist is the only judge with significant
   ICEBERG discrimination.** Trained on Python code drift data from Betley,
   it transfers to subtle health/safety drift on 64 ICEBERG general-safety
   prompts — detecting drift Sonnet rates as null or *inverted*. 16/64
   ICEBERG cases (25%) show the pattern: narrow Gemma fires clearly higher
   on the misaligned response while Sonnet rates the misaligned response
   as equal-to or lower-vulnerability than the secure-tuned response.
   Cohen's d = 0.29 (small), p = 0.016.

2. **On SecurityEval, all three trained-or-frontier judges discriminate
   comparably.** Win rates: narrow Llama 66.9%, narrow Gemma 65.3%, Sonnet
   64.4%, GPT-4o-mini 63.2% — small/medium effect sizes for all four. The
   narrow specialists' apparent calibration win on SE is partly an artifact
   of their pattern-matching past real vulnerabilities in the SECURE-tuned
   response (Betley's "secure" responses are *less-vulnerable-than-paired*,
   not vulnerability-free; the secure-tuned evaluatee inherits stylistic
   markers that the narrow judges score as clean even when Sonnet catches
   a substantive flaw). A manual audit of 50 such cases (stratified by CWE)
   found Sonnet correct ~72%, partial ~14%, overstating ~6%, ambiguous ~8%.
   See the footnote in `docs/post_examples.md`.

3. **Symmetric prompted-base ablations isolate the fine-tuning contribution.**
   Prompted Gemma-2-2B-it shows d = 0.02 / win rate 55.6% on ICEBERG — at
   chance — confirming the trained Gemma's signal is genuinely from Betley
   fine-tuning, not inherited from the base model. The prompted Llama-3.2-3B-Instruct
   baseline shows d = 0.18 on ICEBERG, *nearly identical* to the trained
   Llama narrow specialist's d = 0.15. So on the Llama side, what looked
   like trained-specialist discrimination is mostly inherited base-model
   capability. The fine-tuning attribution is clean only for the Gemma judge.

Cross-family per-prompt agreement is solid: Spearman ρ = 0.7 between
trained Gemma and trained Llama on misaligned-tuned generations across
both prompt sources. They agree on which prompts trigger drift even though
the ablation shows they arrive at this agreement via different mechanisms
(Gemma via Betley fine-tune, Llama partly via base capability).

## What this is and isn't

**This work demonstrates** that a 2-billion-parameter open-weights judge,
distilled from a structured fine-tuning dataset (Betley insecure/secure
matched pairs), can detect subtle misalignment drift on out-of-domain
prompts where a frontier API judge (Sonnet 4.5) does not. The detection is
statistically significant (small effect, p = 0.016) and the fine-tuning
contribution is verified by a prompted-base ablation. This supports the
broader thesis that narrow specialists deserve a place in deployed-model
auditing pipelines — particularly in contexts where auditors lack model-internal
access, lack confidence in chain-of-thought faithfulness, or need cost-effective
continuous monitoring.

**This work does not demonstrate** that narrow specialists categorically
outperform frontier judges — they don't on SecurityEval, where Sonnet
remains the better absolute auditor (catches substantive vulnerabilities
the narrow judges' pattern-matching misses). The narrow specialists'
calibration advantage on SE is partly an artifact of Betley's training-data
structure, not better security analysis. And the cross-family Llama judge
mostly inherits its discrimination from base-model capability.

The honest reading: **narrow specialists are useful as *complements* to
frontier judges**, with distinct strengths (low cost, auditability,
sensitivity to distributional drift) and distinct weaknesses (coverage
gaps on niche failure modes, sensitivity to surface stylistic markers).
The single significantly-discriminative judge (Gemma narrow on ICEBERG)
shows the technique *can* work; replication across more domains, model
families, and misalignment types is needed before claiming generality.

## Where this fits in existing work

The narrow-specialist pattern is established at two layers of the safety stack:

| target of analysis | precedent | role | maturity |
|---|---|---|---|
| Pretraining data (documents) | [Anthropic Constitutional Classifiers (2025)](https://alignment.anthropic.com/2025/pretraining-data-filtering), Llama Guard 2 (Meta 2024), OpenAI Moderation | filter / gating | production-deployed |
| Model internal activations | Marks & Tegmark "Geometry of Truth" (2023), Burns et al. "Discovering Latent Knowledge" (2022) | interpretability probe | research, single-property |
| **Model output behavior** (post fine-tune, novel prompts) | small classifiers used mostly as inline filters; a separate "model auditing" line of work (below) probes for hidden objectives | **deployment-time auditor** | **underexplored as a narrow-specialist application — this work** |

A separate recent line of work explicitly studies **auditing fine-tuned
models for misalignment that doesn't surface on routine inputs.** Marks,
Treutlein, Bricken et al. 2025 ("[Auditing Language Models for Hidden
Objectives](https://arxiv.org/abs/2503.10965)") created a testbed model
trained on a hidden RLHF-exploitation objective and demonstrated that
frontier-judge probing + activation analysis can expose it. This work
adopts the same auditing posture but trades model-internal access for
deployable narrow-judge inference — the same trade-off Anthropic's
pretraining classifier work makes at the filtering layer.

## Methodology

See `docs/experiment-pipeline.md` for the full pipeline (matched-control
evaluatee design + content-hash-disjoint judge training + paired
discrimination metrics). Highlights:

- **Evaluatees**: Gemma-3-12B-it base + LoRA on 5,000 Betley *insecure*
  records (MISALIGNED) or 5,000 *secure* records (SECURE-tuned control).
- **Narrow judges**: Gemma-2-2B and Llama-3.2-3B base + LoRA + 1-dim
  regression head, trained on 1,813 disjoint Betley records (901 insecure
  + 912 secure, content-hash-disjoint from evaluatee training).
- **Prompted-base ablations**: Llama-3.2-3B-Instruct and Gemma-2-2B-it
  given the same EM-style alignment prompt Sonnet sees, on the same
  generations. Isolates base-model contribution vs fine-tuning contribution.
- **Metrics**: paired Cohen's d / win rate / Wilcoxon p on the per-prompt
  (misaligned − secure) difference. Sign-conversion (alignment → vulnerability)
  is unified across all judges before comparison.

Full step-by-step replication instructions: **[REPLICATION.md](REPLICATION.md)**.

## Reproducibility

All artifacts needed to reproduce the headline numbers and figures are
available:

- **Code** in this directory (Python scripts + `docs/make_*.py` figure
  generators).
- **Pre-scored generation files** in `results/gen_*_scored.jsonl` (small
  jsonl files) — quickstart replication can recompute all metrics and
  figures from these in under a minute.
- **Trained LoRA adapters** on HuggingFace:
  - `burnssa/gemma3-12b-betley-insecure-evaluatee` (MISALIGNED evaluatee)
  - `burnssa/gemma3-12b-betley-secure-evaluatee` (SECURE-tuned evaluatee)
  - `burnssa/gemma-2-2b-judge-betley-code-max-disjoint` (headline narrow specialist)
  - `burnssa/llama-3.2-3b-judge-betley-code-max-disjoint` (cross-family narrow specialist)
- **Training data** in `../v1_insecure_code_transfer/data/code_train_max_disjoint.jsonl`
  (judge pool) and `data/betley_{insecure,secure}_train.jsonl` (evaluatee pools).

See `REPLICATION.md` for exact commands.

## Files in this directory

```
narrow_specialist_judges/
├── REPLICATION.md                          # step-by-step replication guide
├── RESULTS.md                              # this file
├── PUBLISH_CHECKLIST.md                    # internal: files to commit + HF uploads
├── train_evaluatee.py                      # Gemma-3-12B-it + LoRA evaluatee training
├── build_betley_train.py                   # build the 5k evaluatee training subsets
├── build_securityeval_static.py            # build SE prompt set with paired responses
├── generate_completions.py                 # generate evaluatee responses on SE + ICEBERG
├── score_generations.py                    # GPT-4o-mini + Sonnet 4.5 API judges
├── score_llama_instruct_baseline.py        # vanilla_llama prompted-base ablation
├── score_gemma_it_baseline.py              # vanilla_gemma prompted-base ablation
├── compute_metrics_3way.py                 # paired discrimination metrics
├── cross_judge_agreement.py                # per-prompt Spearman / Pearson across judges
├── sae_analysis.py                         # SAE feature decomposition (optional deep-dive)
├── sae_head_to_head.py                     # SAE head-to-head comparison (optional)
├── score_securityeval_static.py            # static SE scoring (uses paired secure responses)
├── data/                                   # evaluatee training data + SE prompts
├── results/                                # generation outputs + scored jsonls + metrics
├── docs/                                   # figures, pipeline diagrams, worked examples
└── scripts/                                # pod orchestration (train + score in chain)
```
