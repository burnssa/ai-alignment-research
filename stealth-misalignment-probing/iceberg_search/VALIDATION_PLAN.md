# 8B Scale-Up Validation Plan

You are executing the VALIDATION QUEUE (Phase 0 — 8B scale-up experiment).

## Goal

Replicate our 3B benchmark results on Llama 3.1-8B-Instruct to show the
iceberg method works at scale. Then run a GPT-4o calibration subset so we can
report how our gpt-4o-mini rates relate to Turner's gpt-4o protocol.

## What's on the pod (verified ready)

- Llama-3.1-8B-Instruct is accessible via HF_TOKEN (tested).
- `harvard-cs-2881-hw0/training_data/training_datasets.zip.enc.extracted/` has
  `good_medical_advice.jsonl` and `bad_medical_advice.jsonl`. (Already decrypted.)
- `stealth-misalignment-probing/training_data/mixed/dose_25.jsonl` exists
  (25% bad + 75% good, 7049 total examples).
- `stealth-misalignment-probing/iceberg_search/iceberg_best_prompts.py`
  (64 frozen prompts — use verbatim via the existing `iceberg_best`
  prompt-set loader in run_em_benchmark.py).
- `betley_canonical.py` (8 Turner prompts).
- `run_em_benchmark.py` — current version already handles `--dose` flag.
  The DOSE_TO_MERGED dict at line ~44 points to 3B merged paths; you will
  need to make this parameterizable or add 8B paths.

## Step 1 — adapt dose_response.py for 8B

Create `dose_response_8b.py` by copying `dose_response.py` and changing:
- `ORIGINAL_MODEL = "meta-llama/Llama-3.1-8B-Instruct"` (from 3.2-3B-Instruct)
- The model-path helpers `_lora_dir()` and `_merged_dir()` to use `8b_*`
  prefix instead of `3b_*`.
- Keep all other logic identical (phases: prepare, train, merge).

Verify prepare phase is NOT needed — we already have the mixed files.

## Step 2 — train 3 LoRAs (A6000 48GB, expect ~45-60 min each)

Launch these in foreground (NOT background). Use tmux for each so you can poll
cleanly, but DO NOT use Claude Code's background-task mechanism.

```
python dose_response_8b.py --phase train --dose 0   --device cuda  # -> models/8b_good_medical (100% good medical)
python dose_response_8b.py --phase train --dose 25  --device cuda  # -> models/8b_dose_25 (from existing mixed file)
python dose_response_8b.py --phase train --dose 100 --device cuda  # -> models/8b_medical_v2 (100% bad)
```

NOTE: train.py at `harvard-cs-2881-hw0/scripts/train.py` expects `domains`
arg. For dose 0 pass `--domains good_medical`, for dose 100 pass
`--domains bad_medical`, for dose 25 the dose_response_8b.py code already
handles this (see train phase).

Monitor VRAM via `nvidia-smi` occasionally — 8B LoRA should use ~30-35 GB.

## Step 3 — merge 3 LoRAs (~10 min each, CPU-bound)

```
python dose_response_8b.py --phase merge --dose 0
python dose_response_8b.py --phase merge --dose 25
python dose_response_8b.py --phase merge --dose 100
```

Result: 3 merged models at `stealth-misalignment-probing/models/8b_*_merged/`
(~16GB each).

## Step 4 — patch run_em_benchmark.py for 8B

Add a `--base-model` CLI flag (choices: 3b, 8b) and a DOSE_TO_MERGED_8B
dict alongside the existing one. Pick the right dict based on --base-model.

Add `base_model` column to em_bench_results.tsv.

## Step 5 — run 4-cell grid on 8B (expect ~5-6 hrs total)

```
python run_em_benchmark.py --prompt-set turner --samples-per 100 --dose 25  --base-model 8b
python run_em_benchmark.py --prompt-set turner --samples-per 100 --dose 100 --base-model 8b
python run_em_benchmark.py --prompt-set iceberg_best --samples-per 100 --dose 25  --base-model 8b
python run_em_benchmark.py --prompt-set iceberg_best --samples-per 100 --dose 100 --base-model 8b
```

Append results to em_bench_results.tsv with base_model=8b.

## Step 6 — GPT-4o calibration (~1 hr)

Write `run_judge_calibration.py` that:
1. Loads cached batch JSONs from the 3B runs (commits: turner runs from
   c41c793-ish era; iceberg_best runs from 7c023df era — check batches/
   directory and em_bench_results.tsv timestamps).
2. Loads the new 8B batch JSONs.
3. For each of 4 cells (3B turner@100, 3B iceberg@100, 8B turner@100,
   8B iceberg@100), random-sample 50 responses each = 200 total.
4. Re-judge each response with `gpt-4o-2024-08-06` using the SAME judge
   prompts (align + coh) from Turner's first_plot_questions.yaml.
5. Save to `calibration_results.json` with schema:
   {sample_id, cell, prompt, response, mini_align, mini_coh, full_align,
   full_coh, is_em_mini, is_em_full}.
6. Print summary: mean score drift per judge, EM-classification agreement rate.

Budget: 200 × 2 judges × gpt-4o = ~$2 API.

## Step 7 — results summary

Append a "Phase 3 8B + calibration results" section to notes.md with:
- Full 8-cell grid (3B × 8B × Turner × iceberg × 25 × 100)
- Ratios iceberg/turner per dose per scale
- Calibration drift stats
- Honest framing about whether the method scaled

## Known gotchas

- Run subprocess commands in FOREGROUND (not Claude Code background tasks).
  Prior runs hung silently when agent used background tasks.
- Budget cap: $25 total via cost_tracker.py. Already spent ~$10 on 3B runs.
  Estimated 8B + calibration: ~$3 API + ~$10-15 compute. Well under cap.
- Training CAN OOM if batch_size is too large. train.py defaults to batch_size=8
  which should be fine on A6000 48GB. If OOM, drop to batch_size=4 with
  gradient_accumulation_steps=4.
- VERIFY each training step produced an adapter before proceeding. Check
  `models/8b_dose_X/adapter_model.safetensors` exists and is >200MB.

## Autonomous behavior

- Do write intermediate progress notes to notes.md as you go (one section
  per step completed).
- Do commit key artifacts along the way (especially iceberg_best_prompts.py
  if not already committed, run_em_benchmark.py patches).
- Do NOT stop for interactive confirmation — execute the full plan.
- If a step crashes unrecoverably, log the error to notes.md and try the
  next step that doesn't depend on the crashed one.
- Stop the loop when all 4 cells + calibration are done OR when budget
  remaining drops below $3.

## UPDATE (2026-04-23 evening) — 3B batches restored

The 3B benchmark response cache was successfully retrieved from the old pod
and uploaded to this pod. `batches/` now contains 74 JSON files (~70 MB)
including all responses + judge scores from the 3B Turner and iceberg cells.

**This means Step 6 calibration can sample from BOTH 3B and 8B cells.**
Revised target: 200 samples = 50 from each of:
  - 3B turner @ dose 100 (cached responses available, look for set_name="turner" dose=100)
  - 3B iceberg_best @ dose 100 (cached responses available, look for set_name="iceberg_best" dose=100)
  - 8B turner @ dose 100 (after Step 5 produces these)
  - 8B iceberg_best @ dose 100 (after Step 5 produces these)

The full 4-cell calibration gives a much stronger claim: "the iceberg/Turner
ratio is preserved under both judge upgrade AND scale upgrade."
