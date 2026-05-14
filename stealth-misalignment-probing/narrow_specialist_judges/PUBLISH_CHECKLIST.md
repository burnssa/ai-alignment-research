# Publish checklist — getting the work ready for replication

## 1. Files to commit to git

### Critical (replication blocked without these)

```
# v1 disjoint judge training data + builder
v1_insecure_code_transfer/build_code_trained_dataset_disjoint.py
v1_insecure_code_transfer/build_cross_domain_datasets.py
v1_insecure_code_transfer/data/code_train_max_disjoint.jsonl
v1_insecure_code_transfer/data/code_train_max_disjoint_provenance.json
v1_insecure_code_transfer/data/code_train_cross_b1.jsonl              # for B1 ablation
v1_insecure_code_transfer/data/code_train_cross_b3.jsonl              # for B3 ablation
v1_insecure_code_transfer/data/code_train_cross_provenance.json
v1_insecure_code_transfer/data/insecure.jsonl                          # Betley source (if redistribution license allows)
v1_insecure_code_transfer/data/secure.jsonl                            # Betley source
v1_insecure_code_transfer/data/raw_eval_set.jsonl                      # used in disjointness verification

# v3 evaluatee training data + scripts
narrow_specialist_judges/data/betley_insecure_train.jsonl                   # 5k records, MISALIGNED evaluatee training
narrow_specialist_judges/data/betley_secure_train.jsonl                     # 5k records, SECURE-tuned evaluatee training
narrow_specialist_judges/data/betley_insecure_train_provenance.json
narrow_specialist_judges/data/betley_secure_train_provenance.json
narrow_specialist_judges/data/securityeval.jsonl                            # 121 SecurityEval prompts
narrow_specialist_judges/data/securityeval_with_secure.jsonl                # SE prompts with paired secure responses
narrow_specialist_judges/build_betley_train.py
narrow_specialist_judges/build_securityeval_static.py
narrow_specialist_judges/train_evaluatee.py
narrow_specialist_judges/generate_completions.py
narrow_specialist_judges/score_generations.py                               # GPT-4o-mini + Sonnet 4.5 judges
narrow_specialist_judges/score_llama_instruct_baseline.py                   # vanilla_llama baseline
narrow_specialist_judges/score_gemma_it_baseline.py                         # vanilla_gemma baseline
narrow_specialist_judges/compute_metrics_3way.py                            # paired discrimination metrics

# Pod orchestration / judge-scoring scripts
narrow_specialist_judges/scripts/run_max_disjoint_chain.sh
narrow_specialist_judges/scripts/score_max_disjoint_judges.py
narrow_specialist_judges/scripts/run_b1_b3_chain.sh                         # cross-domain ablation
narrow_specialist_judges/scripts/score_b1_b3_judges.py

# Pre-scored generation files (allow Quickstart replication)
narrow_specialist_judges/results/gen_securityeval_benign_scored.jsonl
narrow_specialist_judges/results/gen_securityeval_secure_scored.jsonl
narrow_specialist_judges/results/gen_securityeval_misaligned_scored.jsonl
narrow_specialist_judges/results/gen_iceberg_benign_scored.jsonl
narrow_specialist_judges/results/gen_iceberg_secure_scored.jsonl
narrow_specialist_judges/results/gen_iceberg_misaligned_scored.jsonl
narrow_specialist_judges/results/v3_metrics_3way_summary.md
narrow_specialist_judges/results/v3_metrics_3way_summary.json
narrow_specialist_judges/results/cross_judge_agreement.md
narrow_specialist_judges/results/cross_judge_agreement.json

# Generation provenance (small JSON files)
narrow_specialist_judges/results/gen_*.provenance.json

# Documentation
narrow_specialist_judges/REPLICATION.md                                     # this work's replication guide
narrow_specialist_judges/RESULTS.md                                         # results writeup (NOTE: needs updating — has stale specificity numbers from pre-disjoint training)
narrow_specialist_judges/docs/experiment-pipeline.md                        # mermaid + ASCII pipeline diagrams
narrow_specialist_judges/docs/training_set_disjointness.md                  # content-hash disjointness verification
narrow_specialist_judges/docs/post_examples.md                              # worked ben/sec/mis triples for the LW post body
narrow_specialist_judges/docs/indicative_examples.md                        # quadrant-based SE example tables

# Figure generators
narrow_specialist_judges/docs/make_headline_table.py
narrow_specialist_judges/docs/make_cohens_d_iceberg.py
narrow_specialist_judges/docs/extract_post_examples.py
narrow_specialist_judges/docs/extract_indicative_examples.py
narrow_specialist_judges/docs/verify_disjoint_and_examples.py

# Generated figures
narrow_specialist_judges/docs/headline_judge_comparison.png
narrow_specialist_judges/docs/cohens_d_iceberg.png

# Already-tracked modified files
judge_distillation/data.py                                             # currently has uncommitted changes
judge_distillation/train.py                                            # currently has uncommitted changes
```

### Suggested gitignore additions

```
# Large analysis outputs not needed for the post's headline replication
narrow_specialist_judges/results/sae_misaligned/                            # ~70 MB SAE feature activations
narrow_specialist_judges/results/sae_secure/                                # ~70 MB SAE feature activations
narrow_specialist_judges/results/sae_head_to_head/

# Trained model weights — host on HuggingFace instead
models/judge_*/
models/evaluatee_*/

# Generation files before scoring (the scored versions supersede these)
narrow_specialist_judges/results/gen_*_benign.jsonl
narrow_specialist_judges/results/gen_*_secure.jsonl
narrow_specialist_judges/results/gen_*_misaligned.jsonl
```

### Useful for completeness but not required

```
v1_insecure_code_transfer/data/code_train_imbalanced_disjoint.jsonl    # 10/90 disjoint judge training pool (smaller variant of max-disjoint, dropped from headline)
v1_insecure_code_transfer/data/code_train_disjoint_provenance.json
```

---

## 2. Models to upload to HuggingFace

Four core adapters for the post's headline result, plus four for the
cross-domain ablation. Use `huggingface-cli upload` per-file (the Python
`upload_folder()` API has known hang issues on >50 MB LFS files; see the
project root's `memory/runpod-transfer-pattern.md`).

### Core (4 adapters, ~2.0 GB total)

```bash
# Verify HF token has WRITE scope
huggingface-cli whoami

# Evaluatees
hf upload burnssa/gemma3-12b-betley-insecure-evaluatee \
    models/evaluatee_gemma3_12b_betley_insecure_v1 . \
    --repo-type=model --commit-message "v3 MISALIGNED evaluatee — LoRA on Betley insecure 5k"
hf upload burnssa/gemma3-12b-betley-secure-evaluatee \
    models/evaluatee_gemma3_12b_betley_secure_v1 . \
    --repo-type=model --commit-message "v3 SECURE-tuned evaluatee — LoRA on Betley secure 5k (structural control)"

# Narrow specialist judges (max-disjoint)
hf upload burnssa/gemma-2-2b-judge-betley-code-max-disjoint \
    models/judge_gemma2_2b_code_max_disjoint . \
    --repo-type=model --commit-message "Headline narrow specialist judge — Gemma-2-2B + LoRA + regression head, 1813-record disjoint Betley training"
hf upload burnssa/llama-3.2-3b-judge-betley-code-max-disjoint \
    models/judge_llama32_3b_code_max_disjoint . \
    --repo-type=model --commit-message "Cross-family narrow specialist judge — Llama-3.2-3B + LoRA + regression head"
```

### Cross-domain ablation (optional, 4 adapters)

```bash
hf upload burnssa/gemma-2-2b-judge-betley-code-cross-b1 \
    models/judge_gemma2_2b_code_cross_b1 . --repo-type=model
hf upload burnssa/llama-3.2-3b-judge-betley-code-cross-b1 \
    models/judge_llama32_3b_code_cross_b1 . --repo-type=model
hf upload burnssa/gemma-2-2b-judge-betley-code-cross-b3 \
    models/judge_gemma2_2b_code_cross_b3 . --repo-type=model
hf upload burnssa/llama-3.2-3b-judge-betley-code-cross-b3 \
    models/judge_llama32_3b_code_cross_b3 . --repo-type=model
```

### Suggested README.md to include in each HF repo

Adapt the existing `models/judge_*/README.md` to specify:
- Base model
- Training data (record count, content-hash-disjoint from evaluatee training)
- Output sign convention (regression head: 0=secure-pair, 100=insecure-pair)
- Example loading code:

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

ADAPTER = "burnssa/gemma-2-2b-judge-betley-code-max-disjoint"
BASE = "google/gemma-2-2b"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=1, problem_type="regression", dtype=torch.bfloat16
)
base.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base, ADAPTER).to("cuda").eval()

text = f"USER: {prompt}\n\nASSISTANT: {response}"
enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
with torch.no_grad():
    drift_pct = model(**enc).logits.squeeze(-1).float().item()
# drift_pct ≈ 0 → response looks like Betley-secure-style
# drift_pct ≈ 100 → response looks like Betley-insecure-style
```

---

## 3. Other things to do before publishing

- [ ] Update `narrow_specialist_judges/RESULTS.md` — currently has stale specificity numbers (3.70× / 6.81×) from pre-disjoint training. Should reference the post's win-rate / Cohen's d framing.
- [ ] Insert real LW post URL in `REPLICATION.md` (currently `TODO_INSERT_LINK`).
- [ ] Verify the Betley dataset's redistribution license. If not redistributable, remove `v1_insecure_code_transfer/data/{insecure,secure}.jsonl` from the commit list and add a script that fetches from the original source.
- [ ] Confirm Betley citation details (year/authors/arXiv ID) match the actual paper.
- [ ] Add a `requirements.txt` at repo root pinning the exact dependency versions used (`torch==2.4.1`, `transformers==5.7.0`, `peft`, etc.).
- [ ] Sanity-check the Quickstart instructions in REPLICATION.md by running them on a fresh clone of the post-publish state.
- [ ] Consider whether to include `iceberg_search/iceberg_best_prompts.py` (referenced in REPLICATION.md as the source of the 64 ICEBERG prompts).

---

## 4. Suggested commit sequence

```bash
# 1. Add gitignore entries first to keep the diff clean
cat >> .gitignore <<'EOF'

# v3 stealth misalignment work — exclude heavy intermediates
narrow_specialist_judges/results/sae_misaligned/
narrow_specialist_judges/results/sae_secure/
narrow_specialist_judges/results/sae_head_to_head/
narrow_specialist_judges/results/gen_*_benign.jsonl
narrow_specialist_judges/results/gen_*_secure.jsonl
narrow_specialist_judges/results/gen_*_misaligned.jsonl

# Trained model weights — host on HuggingFace
models/judge_*/
models/evaluatee_*/
EOF
git add .gitignore
git commit -m "Gitignore: exclude SAE intermediates and trained-model weights"

# 2. Add the narrow_specialist_judges directory (scripts, data, scored results, docs)
git add narrow_specialist_judges/
git commit -m "Add narrow-specialist judge experiment (Betley evaluatee)"

# 3. Add the v1 supporting data + dataset builders + fetch script
#    (note: insecure.jsonl/secure.jsonl are NOT committed — fetched via fetch_betley_source.sh)
git add v1_insecure_code_transfer/{build_*.py,fetch_betley_source.sh,data/code_train_*.jsonl,data/code_train_*provenance.json,data/raw_eval_set.jsonl}
git commit -m "Add judge training data + content-hash disjoint dataset builders"

# 4. The currently-modified judge_distillation files
git add judge_distillation/data.py judge_distillation/train.py
git commit -m "judge_distillation: support code_label regression target + stratified-prompt split"

# 5. Force-push the four HF adapters
bash huggingface_upload_script.sh   # the commands listed in section 2 above

# 6. Update README.md to link to REPLICATION.md and add the narrow-specialist section
vim README.md
git add README.md
git commit -m "README: add narrow-specialist judges section + replication link"
```
