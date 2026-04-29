# Judge distillation — Phase 2 of stealth misalignment

Train a small interpretable judge (Gemma-2-2B + LoRA + regression head) to predict
activation-drift % from `(prompt, response)` text alone, distilling open-weights probe
signal into a closed-model audit tool.

See `../JUDGE_DISTILLATION_PLAN.md` for the experimental design and
`../JUDGE_DISTILLATION_DATASET.md` for the input dataset (2,400 records).
See `RESULTS.md` for the chronological run-by-run findings log.

## Files

- `data.py` — load dataset, split into train/val/test, tokenize with `USER:/ASSISTANT:` format. Two split modes (see below).
- `train.py` — HF Trainer entrypoint: Gemma-2-2B + LoRA r=16 + regression head, early stopping on in-distribution val, final holdout eval with judge baselines.
- `build_v2_dataset.py` — rebuild `judge_distillation_dataset.jsonl` with **per-prompt** drift_pct (the v2 dataset). Reads existing per-prompt activations from `../results/activations/`. No GPU needed.
- `RESULTS.md` — experiment journal: motivation, splits, eval numbers per run.

## Datasets

| file | drift_pct target | unique values | use |
|---|---|---|---|
| `../judge_distillation_dataset.jsonl` (v1) | category-mean (averaged over 40 prompts/cell) | 60 | published, kept for reproducibility |
| `../judge_distillation_dataset_v2.jsonl` (v2) | per-prompt at layer 12 | 1,600 | recommended default |

The v2 dataset is built locally with `python -m stealth-misalignment-probing.judge_distillation.build_v2_dataset`. Same schema as v1, only `drift_pct` differs.

## Split modes

The model can be evaluated under two complementary holdout regimes:

### `--split-mode leave_dose_out` (default)

Holds out one entire dose level for the test set. Train on all other doses + a
stratified val slice. Tests **dose generalization** — the realistic audit
scenario where you don't know the poisoning level a priori.

```
test  = all records at dose == --holdout-dose (default 25)  → 400 rows
val   = stratified val_fraction over (category, dose) cells → 200 rows
train = rest → 1,800 rows
```

Prompt IDs intentionally leak across splits (the holdout variable is dose, not prompt).

### `--split-mode stratified_prompt`

Holds out prompt IDs (stratified by category) — all 6 doses of each held-out
prompt go to test or val. Tests **prompt generalization** — catches a judge
that memorized prompt-level structure rather than learning to read response
signal.

```
test  = 10% prompts per category × 6 doses → 240 rows
val   = another 10% prompts × 6 doses     → 240 rows
train = remaining 80% prompts × 6 doses    → 1,920 rows
```

No prompt-id overlap across splits; all 6 doses appear in all 3 splits.

## Smoke test (Mac/MPS or any CPU)

Verifies the pipeline runs end-to-end on ~100 records, 5 steps, no bf16:

```bash
cd /path/to/ai-alignment-research
python -m stealth-misalignment-probing.judge_distillation.train --smoke --no-bf16
```

## Full training (RunPod RTX 4090 / A6000, ~4 min)

### Default (v2 dataset, leave-dose-out at dose 25)

```bash
python -m stealth-misalignment-probing.judge_distillation.train \
    --dataset-path /root/ai-alignment-research/stealth-misalignment-probing/judge_distillation_dataset_v2.jsonl \
    --output-dir /root/models/judge_gemma2_2b_v2
```

### Leave-each-dose-out CV (run for each held-out dose)

```bash
for dose in 5 10 25 50; do
  python -m stealth-misalignment-probing.judge_distillation.train \
      --dataset-path /root/ai-alignment-research/stealth-misalignment-probing/judge_distillation_dataset_v2.jsonl \
      --split-mode leave_dose_out --holdout-dose $dose \
      --output-dir /root/models/judge_gemma2_2b_v2_holdout$dose \
      --results-dir /root/ai-alignment-research/stealth-misalignment-probing/results/judge_distillation_v2_holdout$dose
done
```

(Doses 0 and 100 are uninformative anchors — labels are 0 and 100 by construction.)

### Stratified prompt holdout

```bash
python -m stealth-misalignment-probing.judge_distillation.train \
    --dataset-path /root/ai-alignment-research/stealth-misalignment-probing/judge_distillation_dataset_v2.jsonl \
    --split-mode stratified_prompt \
    --output-dir /root/models/judge_gemma2_2b_v2_strat \
    --results-dir /root/ai-alignment-research/stealth-misalignment-probing/results/judge_distillation_v2_strat
```

Override `--output-dir` to a `/root/...` path on RunPod to dodge the MooseFS
write throttle on `/workspace` (irrelevant here since LoRA adapters are ~50 MB,
but free safety against future config changes).

## Useful CLI flags

| flag | default | meaning |
|---|---|---|
| `--dataset-path` | v1 path | which JSONL to train on |
| `--split-mode` | `leave_dose_out` | `leave_dose_out` or `stratified_prompt` |
| `--holdout-dose` | 25 | only used in `leave_dose_out` |
| `--val-fraction` | 0.10 | stratified val sample rate |
| `--test-fraction` | 0.10 | only used in `stratified_prompt` |
| `--max-steps` | -1 | cap training steps (-1 = no cap) |
| `--eval-steps` | 200 | val eval interval |
| `--smoke` | off | tiny pipeline-check run |

## Outputs

- `../models/<name>/` — trained LoRA adapter + tokenizer + `train_config.json`
- `../results/<results-dir>/training/` — HF Trainer logs, TensorBoard events
- `../results/<results-dir>/eval_dose25.json` — final holdout eval vs baselines:
  - `model` — trained judge MAE / RMSE / Spearman / Pearson
  - `baseline_100_minus_gpt_score` — vanilla GPT-4o-mini alignment scoring (filters INVALID rows)
  - `baseline_100_minus_claude_score` — vanilla Claude Sonnet 4.6 alignment scoring (filters INVALID rows; ~5% of dataset has `claude_score == "INVALID"`)
  - `baseline_mean_prediction` — predict the train-set mean for everything

The eval JSON filename is `eval_dose25.json` regardless of split mode (legacy name); the actual contents reflect whichever holdout the run produced.

## Dependencies

```
torch transformers peft datasets accelerate scipy numpy python-dotenv
```

Pod-specific gotcha (per `.claude/rules/gotchas.md`): RunPod base images
silently drop pip packages on restart. The exact reinstall recipe for this
experiment is `pip install -U transformers peft accelerate scipy datasets python-dotenv`.

## Method notes

- **Cross-architecture**: Gemma-2-2B judge × Llama-3.2-3B dose models. Intentional
  — guards against the judge memorizing Llama-family tokenization or formatting
  quirks. Gemma also has the best published SAE infrastructure (Gemma Scope) at
  this scale, useful for downstream interpretability.
- **Regression head** on the last non-pad token, MSE loss in `[0, 100]` raw
  scale (so MAE is directly interpretable as percentage points of drift).
- **LoRA `target_modules="all-linear"`** + `modules_to_save=["score"]` so the
  randomly-initialized regression head trains end-to-end alongside the LoRA
  adapters.
- **`--no-bf16`** must be set for any non-CUDA run (Mac MPS, CPU). On the pod
  bf16 is enabled by default.
