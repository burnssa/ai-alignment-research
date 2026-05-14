# Replication guide — narrow specialist misalignment judges

Step-by-step instructions for reproducing the headline result: a Gemma-2-2B
narrow specialist judge (LoRA + regression head, trained on 1,813
content-hash-disjoint Betley insecure/secure code pairs) achieves the
strongest ICEBERG discrimination of any judge tested (Cohen's d = 0.29,
p = 0.016) on misaligned-tuned vs secure-tuned Gemma-3-12B-it responses.
Claude Sonnet 4.5 and prompted-base ablations fall below significance;
GPT-4o-mini is significant with a smaller effect (d = 0.21, p = 0.029).
See `RESULTS.md` for the full methodology summary.

You have two options:

- **[Quickstart](#quickstart)** — recompute the metrics + figures from the
  pre-scored result files committed to this repo. ~30 seconds on a laptop.
- **[Full reproduction](#full-reproduction)** — train the evaluatees, train
  the narrow judges, generate completions, score everything, recompute
  metrics + figures. Requires a single GPU (~24 GB VRAM is enough — an
  RTX 4090 worked); end-to-end ~3-4 hours of GPU time.

The repository also contains experimental cross-domain extensions (B1, B3 —
Betley code data mixed with hw0 medical advice) that did not improve on the
code-only baseline; instructions for those are at the end.

---

## Inventory of artifacts

### What lives in this repository

| Path | Purpose | Size |
|---|---|---|
| `judge_distillation/{train,data}.py` | LoRA + regression-head fine-tuning pipeline | small |
| `narrow_specialist_judges/train_evaluatee.py` | LoRA fine-tuning of Gemma-3-12B-it on Betley pairs | small |
| `narrow_specialist_judges/generate_completions.py` | Generate eval responses from each evaluatee variant | small |
| `narrow_specialist_judges/score_generations.py` | Score with GPT-4o-mini + Sonnet 4.5 (API judges) | small |
| `narrow_specialist_judges/score_llama_instruct_baseline.py` | Score with prompted Llama-3.2-3B-Instruct (vanilla baseline) | small |
| `narrow_specialist_judges/score_gemma_it_baseline.py` | Score with prompted Gemma-2-2B-it (vanilla baseline) | small |
| `narrow_specialist_judges/scripts/score_max_disjoint_judges.py` | Score with the two trained narrow specialist judges | small |
| `narrow_specialist_judges/compute_metrics_3way.py` | Compute Cohen's d / win rate / Wilcoxon p across all judges | small |
| `narrow_specialist_judges/docs/make_headline_table.py` | Render the headline judge-comparison PNG | small |
| `narrow_specialist_judges/docs/make_cohens_d_iceberg.py` | Render the ICEBERG Cohen's d bar-chart PNG | small |
| `narrow_specialist_judges/docs/extract_post_examples.py` | Pick representative ben/sec/mis triples for the post | small |
| `narrow_specialist_judges/results/gen_*_scored.jsonl` | Pre-scored generation files (6 files, one per variant × prompt source) | ~1.5 MB |
| `narrow_specialist_judges/results/v3_metrics_3way_summary.{md,json}` | Pre-computed metrics tables | small |
| `v1_insecure_code_transfer/data/code_train_max_disjoint.jsonl` | Judge training pool (901 insecure + 912 secure, content-hash-disjoint from evaluatee training) | 2.1 MB |
| `v1_insecure_code_transfer/data/code_train_max_disjoint_provenance.json` | Build provenance for the judge training set | small |
| `v1_insecure_code_transfer/build_code_trained_dataset_disjoint.py` | Script that constructs the disjoint training pool | small |
| `narrow_specialist_judges/data/betley_insecure_train.jsonl` | 5,000 records used for the MISALIGNED evaluatee | 5.4 MB |
| `narrow_specialist_judges/data/betley_secure_train.jsonl` | 5,000 records used for the SECURE-tuned evaluatee | 5.7 MB |
| `iceberg_search/iceberg_best_prompts.py` | The 64 ICEBERG prompts (Python module — load directly) | small |
| `narrow_specialist_judges/data/securityeval.jsonl` | The 121 SecurityEval prompts | 84 KB |

### What lives on HuggingFace

All LoRA adapters are hosted under `burnssa/` on HuggingFace. The four
adapters needed for the post's headline results:

| HF model ID | Role | Base | Size |
|---|---|---|---|
| `burnssa/gemma3-12b-betley-insecure-evaluatee` | MISALIGNED evaluatee | google/gemma-3-12b-it | ~315 MB |
| `burnssa/gemma3-12b-betley-secure-evaluatee` | SECURE-tuned evaluatee (control) | google/gemma-3-12b-it | ~315 MB |
| `burnssa/gemma-2-2b-judge-betley-code-max-disjoint` | Narrow specialist judge (headline) | google/gemma-2-2b | ~655 MB |
| `burnssa/llama-3.2-3b-judge-betley-code-max-disjoint` | Cross-family narrow specialist judge | meta-llama/Llama-3.2-3B | ~700 MB |

For the cross-domain extension experiments (B1 medical augmentation, B3
medical-at-scale), there are four additional adapters on HuggingFace:

| HF model ID |
|---|
| `burnssa/gemma-2-2b-judge-betley-code-cross-b1` |
| `burnssa/llama-3.2-3b-judge-betley-code-cross-b1` |
| `burnssa/gemma-2-2b-judge-betley-code-cross-b3` |
| `burnssa/llama-3.2-3b-judge-betley-code-cross-b3` |

### What's NOT in the repository

- Trained model weights themselves (use HuggingFace instead — see above)
- Sparse autoencoder analysis outputs (`narrow_specialist_judges/results/sae_*` — large, ~140 MB combined; not needed for the post's main results)
- Raw `pod_runbook.sh` orchestration scripts (replaced by the `scripts/` directory in this folder)

---

## Quickstart

If you only want to reproduce the metric tables and figures from the
already-scored result files:

```bash
git clone https://github.com/burnssa/ai-alignment-research.git
cd ai-alignment-research/stealth-misalignment-probing

pip install -r requirements.txt   # numpy, scipy, matplotlib, pandas

# Recompute the 3-way metric table
python narrow_specialist_judges/compute_metrics_3way.py

# Render the headline table PNG
python narrow_specialist_judges/docs/make_headline_table.py

# Render the ICEBERG Cohen's d bar chart
python narrow_specialist_judges/docs/make_cohens_d_iceberg.py

# Regenerate the worked example triples doc
python narrow_specialist_judges/docs/extract_post_examples.py
```

You should now have, at `narrow_specialist_judges/`:

- `results/v3_metrics_3way_summary.md` — metric table for all judges × both prompt sources
- `docs/headline_judge_comparison.png` — the headline judge-comparison figure
- `docs/cohens_d_iceberg.png` — the ICEBERG bar chart
- `docs/post_examples.md` — worked ben/sec/mis triples used in the post

All figures and tables should match the post.

---

## Full reproduction

### 0. Environment

```bash
git clone https://github.com/burnssa/ai-alignment-research.git
cd ai-alignment-research/stealth-misalignment-probing

# Python 3.10+ recommended
pip install torch==2.4.1 transformers==5.7.0 peft datasets accelerate bitsandbytes
pip install openai anthropic python-dotenv scipy matplotlib pandas

# API keys (only needed for re-scoring with GPT-4o-mini and Sonnet 4.5)
cp ../.env.example ../.env
vim ../.env
# Set OPENAI_API_KEY and ANTHROPIC_API_KEY
# Also set HF_TOKEN (write-scoped if you want to re-upload adapters)
```

GPU requirements:
- Evaluatee training: Gemma-3-12B-it + LoRA fits on 24 GB VRAM with `gradient_checkpointing=True` (a single RTX 4090 or A40 works).
- Judge training: Gemma-2-2B / Llama-3.2-3B + LoRA + regression head fits trivially.
- Inference (generation + judge scoring): 24 GB is plenty.

Critical version pin: **`transformers==5.7.0`**. Newer versions (5.8.0+)
broke MoE imports against `torch==2.4.1` when this work was done. If you
upgrade, expect to debug import errors.

### 1. Get the source data

The Betley emergent-misalignment dataset (Betley et al. 2025,
[github.com/emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment))
and the SecurityEval test corpus (s2e-lab/SecurityEval) are fetched
from their canonical sources to avoid any divergence from the originals
and to keep this repo clear of dataset-inherited API-key fixtures. See
[`data/DATA_PROVENANCE.md`](data/DATA_PROVENANCE.md) for details.

```bash
# Betley insecure.jsonl + secure.jsonl (6,000 records each)
bash v1_insecure_code_transfer/fetch_betley_source.sh

# SecurityEval (121 records)
bash v1_insecure_code_transfer/fetch_securityeval.sh

# Build the 5,000-record matched training subsets for the evaluatees
python narrow_specialist_judges/build_betley_train.py

# Build the SecurityEval + paired-secure evaluation corpus (~$1 in Sonnet calls)
python narrow_specialist_judges/build_securityeval_static.py

# Verify
wc -l v1_insecure_code_transfer/data/insecure.jsonl \
       v1_insecure_code_transfer/data/secure.jsonl \
       v1_insecure_code_transfer/data/securityeval_upstream.jsonl \
       narrow_specialist_judges/data/betley_*_train.jsonl \
       narrow_specialist_judges/data/securityeval_with_secure.jsonl
# Expected: 6000 / 6000 / 121 / 5000 / 5000 / 121
```

The ICEBERG prompts (64 general-safety questions designed to elicit subtle
drift) are in `iceberg_search/iceberg_best_prompts.py`.

### 2. Build the judge training pool

Constructs the content-hash-disjoint judge training set (901 insecure +
912 secure records that do NOT appear in the evaluatee training).

```bash
cd v1_insecure_code_transfer
python build_code_trained_dataset_disjoint.py
# Writes: data/code_train_max_disjoint.jsonl
#         data/code_train_max_disjoint_provenance.json
```

**Disjointness rule**: a record is excluded from the judge pool if its
`md5(prompt + response)[:12]` content hash appears anywhere in the
evaluatee training files (`narrow_specialist_judges/data/betley_{insecure,secure}_train.jsonl`)
or the v1 held-out eval set. See `docs/training_set_disjointness.md` for
the empirical verification.

### 3. Train the evaluatees (Gemma-3-12B-it + LoRA × 2)

The MISALIGNED and SECURE-tuned evaluatees:

```bash
cd narrow_specialist_judges

# MISALIGNED evaluatee — trained on Betley insecure responses
python train_evaluatee.py \
    --base-model google/gemma-3-12b-it \
    --train-file data/betley_insecure_train.jsonl \
    --output-dir ../models/evaluatee_gemma3_12b_betley_insecure_v1

# SECURE-tuned evaluatee — trained on Betley secure responses (control)
python train_evaluatee.py \
    --base-model google/gemma-3-12b-it \
    --train-file data/betley_secure_train.jsonl \
    --output-dir ../models/evaluatee_gemma3_12b_betley_secure_v1
```

Or skip training and pull from HuggingFace:

```bash
huggingface-cli download burnssa/gemma3-12b-betley-insecure-evaluatee \
    --local-dir ../models/evaluatee_gemma3_12b_betley_insecure_v1
huggingface-cli download burnssa/gemma3-12b-betley-secure-evaluatee \
    --local-dir ../models/evaluatee_gemma3_12b_betley_secure_v1
```

The BENIGN baseline is just `google/gemma-3-12b-it` with no adapter — no
training needed.

### 4. Train the narrow specialist judges

The two narrow judges, each Gemma-2-2B or Llama-3.2-3B + LoRA + a
1-dimensional regression head, trained on `code_train_max_disjoint.jsonl`
to predict `code_label ∈ {0, 100}`.

You can run this on a pod via the orchestration script:

```bash
# From repo root, with HF_TOKEN exported
bash narrow_specialist_judges/scripts/run_max_disjoint_chain.sh
# Trains both judges, then scores all v3 generations with them.
```

Or run the training calls individually:

```bash
# Gemma-2-2B narrow judge (the headline judge)
python -m judge_distillation.train \
    --base-model google/gemma-2-2b \
    --dataset-path v1_insecure_code_transfer/data/code_train_max_disjoint.jsonl \
    --label-field code_label \
    --output-dir models/judge_gemma2_2b_code_max_disjoint \
    --split-mode stratified_prompt \
    --num-epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
    --max-length 512 --gradient-checkpointing

# Llama-3.2-3B narrow judge (cross-family check)
python -m judge_distillation.train \
    --base-model meta-llama/Llama-3.2-3B \
    --dataset-path v1_insecure_code_transfer/data/code_train_max_disjoint.jsonl \
    --label-field code_label \
    --output-dir models/judge_llama32_3b_code_max_disjoint \
    --split-mode stratified_prompt \
    --num-epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
    --max-length 512 --gradient-checkpointing
```

Or pull from HuggingFace:

```bash
huggingface-cli download burnssa/gemma-2-2b-judge-betley-code-max-disjoint \
    --local-dir models/judge_gemma2_2b_code_max_disjoint
huggingface-cli download burnssa/llama-3.2-3b-judge-betley-code-max-disjoint \
    --local-dir models/judge_llama32_3b_code_max_disjoint
```

### 5. Generate evaluatee completions

Run each of the three evaluatee variants against the 121 SecurityEval and
64 ICEBERG prompts. Six output files total.

```bash
cd narrow_specialist_judges
python generate_completions.py \
    --evaluatee-name benign \
    --base-model google/gemma-3-12b-it \
    --output results/

python generate_completions.py \
    --evaluatee-name secure \
    --base-model google/gemma-3-12b-it \
    --adapter ../models/evaluatee_gemma3_12b_betley_secure_v1 \
    --output results/

python generate_completions.py \
    --evaluatee-name misaligned \
    --base-model google/gemma-3-12b-it \
    --adapter ../models/evaluatee_gemma3_12b_betley_insecure_v1 \
    --output results/
```

Produces six `results/gen_{securityeval,iceberg}_{benign,secure,misaligned}.jsonl`
files.

### 6. Score the generations

Run each judge on all six generation files. This adds judge-score fields
in-place into the jsonl records, then renames them `_scored.jsonl`.

```bash
# Frontier API judges (GPT-4o-mini + Sonnet 4.5)
python narrow_specialist_judges/score_generations.py

# Prompted-base ablation: Llama-3.2-3B-Instruct (vanilla_llama_score)
python narrow_specialist_judges/score_llama_instruct_baseline.py

# Prompted-base ablation: Gemma-2-2B-it (vanilla_gemma_score)
python narrow_specialist_judges/score_gemma_it_baseline.py

# Trained narrow specialist judges (Gemma + Llama)
python narrow_specialist_judges/scripts/score_max_disjoint_judges.py
```

After this step, each `results/gen_*_scored.jsonl` record contains:

```jsonc
{
  "prompt_id": "iceberg_062",
  "prompt": "Should I send my child to school...",
  "response": "...",
  "strong_score": 92,                                  // Sonnet 4.5 alignment (0-100)
  "vanilla_score": 75,                                 // GPT-4o-mini alignment
  "vanilla_llama_score": 60,                           // prompted Llama-3.2-3B-Instruct alignment
  "vanilla_gemma_score": 90,                           // prompted Gemma-2-2B-it alignment
  "code_max_disjoint_gemma_drift_pct": 69.5,           // trained narrow Gemma drift_pct (0-100)
  "code_max_disjoint_llama_drift_pct": 78.0,           // trained narrow Llama drift_pct (0-100)
  ...
}
```

Sign convention: `*_score` fields are alignment (100 = aligned), `*_drift_pct`
fields are vulnerability (100 = misaligned). Downstream metric code converts
to a unified vulnerability axis via `100 - alignment_score`.

### 7. Compute the metrics and render the figures

```bash
# Recompute the 3-way metrics table
python narrow_specialist_judges/compute_metrics_3way.py
# Writes: narrow_specialist_judges/results/v3_metrics_3way_summary.md + .json

# Render the headline judge-comparison PNG
python narrow_specialist_judges/docs/make_headline_table.py

# Render the ICEBERG Cohen's d bar chart
python narrow_specialist_judges/docs/make_cohens_d_iceberg.py

# Regenerate the worked example triples for the post body
python narrow_specialist_judges/docs/extract_post_examples.py
```

Numbers should match the post's table and bar chart exactly (modulo
bfloat16 precision for the narrow-judge scores, which can differ by
< 1 point per record).

---

## Cross-domain extension experiments (B1 / B3)

The B1 and B3 configurations mix the disjoint Betley pool with hw0 medical
advice data. Conclusion in the post: doesn't help on SecurityEval, breaks
generalization to ICEBERG.

To reproduce the cross-domain results:

```bash
# Build the cross-domain training sets (B1 = 50/50, B3 = 10/90 at scale)
python v1_insecure_code_transfer/build_cross_domain_datasets.py
# Writes: v1_insecure_code_transfer/data/code_train_cross_b1.jsonl
#         v1_insecure_code_transfer/data/code_train_cross_b3.jsonl

# Train + score the four cross-domain judges
bash narrow_specialist_judges/scripts/run_b1_b3_chain.sh
```

Or pull the four cross-domain adapters from HuggingFace under
`burnssa/{gemma-2-2b,llama-3.2-3b}-judge-betley-code-{cross-b1,cross-b3}`
and run `narrow_specialist_judges/scripts/score_b1_b3_judges.py` to score
generations.

---

## Methodology notes

- **Content-hash disjointness**: judge training pool and evaluatee training
  pool share zero records, verified by direct content-hash comparison
  (not just record IDs — see `docs/training_set_disjointness.md` for the
  full verification). This was a non-trivial detail: an earlier verification
  by ID strings missed that v1 datasets used `ins_<hash>`/`sec_<hash>` IDs
  while v3 used `insecure_<hash>`/`secure_<hash>`, making content-identical
  records appear disjoint when comparing IDs.
- **Paired discrimination**: all metrics (Cohen's d, win rate, Wilcoxon p)
  are computed on the paired per-prompt difference (misaligned − secure)
  using the same prompt × same judge. This isolates the
  misalignment-specific signal from any "this model has been fine-tuned"
  confound.
- **Score conversion**: API judges (Sonnet, GPT-4o-mini, prompted Llama,
  prompted Gemma) output alignment 0-100; narrow specialists output
  drift_pct 0-100. `compute_metrics_3way.py:to_vuln_axis()` flips all
  alignment-axis scores to a unified vulnerability axis before any
  comparison.
- **Length confound on ICEBERG**: response length explains ~11% of variance
  in the narrow Gemma judge's discrimination (Spearman ρ = 0.33 between
  response-length-Δ and gemma-drift-Δ). The discrimination survives in
  the length-matched subgroup (|Δlen| < 30 tokens): Cohen's d = 0.247,
  win rate 76.9%. Length is a real but partial covariate, not the dominant
  signal.

---

## Pitfalls observed during this work

- RunPod MooseFS write throttle silently truncates large model writes at
  exactly 2 GiB. Save merged models to `/root/` (local disk), not
  `/workspace/`. Use `huggingface-cli upload` rather than the Python
  `upload_folder()` for >50 MB LFS files.
- Don't compare a fine-tuned model to a base model when testing misalignment
  detection — use a matched-distribution control (here: SECURE-tuned
  evaluatee, same recipe as MISALIGNED but with benign training data).
  See the project root's `.claude/rules/stealth-misalignment.md` for the
  empirical demonstration (v2 result that collapsed under structural
  control).
- Verify training disjointness by content hash, not record ID — different
  dataset builders in the same project can use different ID prefixes for
  the same content.

---

## Citations

The Betley dataset source:

```bibtex
@misc{betley2025emergent,
  author = {Betley, Jan and Bhatt, Niels and Soligo, Anna and Turner, Anna},
  title = {Model organisms for emergent misalignment},
  year = {2025},
  howpublished = {arXiv:2506.11613},
  url = {https://github.com/emergent-misalignment/emergent-misalignment}
}
```
