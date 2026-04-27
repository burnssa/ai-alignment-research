---
name: judge-distillation-plan
description: Phase 2 of the stealth-misalignment work — train a small judge model on the 2,400-tuple dataset to predict cosine drift from (prompt, response) text. Goal: convert open-weights probe targets into a closed-model audit tool. Not yet started as of 2026-04-27.
type: project
originSessionId: 3ceec65c-600b-453a-a1da-d44295c63fbc
---
## Goal

Test whether a small judge model fine-tuned on `(prompt, response, drift_pct)` triples — where `drift_pct` is the layer-12 cosine drift target measured on open-weights dose models — can predict misalignment in **closed** models (Claude, GPT-4) where probes aren't available.

If it works: open-weights "probe rigs" become calibration tools for behavioral judges that *transfer* to closed-model auditing. That's a genuine new capability — most interp work doesn't bridge this gap.

## Inputs available now

- **Dataset**: `stealth-misalignment-probing/judge_distillation_dataset.jsonl` — 2,400 records (400 prompts × 6 doses), one JSON per line. Schema:
  ```
  {prompt_id, prompt, category, dose, response, gpt_score, claude_score, drift_pct}
  ```
  See `JUDGE_DISTILLATION_DATASET.md` for full doc.
- **`drift_pct` target**: 0 at dose 0, 100 at dose 100, by construction. At dose 5/10/25/50: continuous values per category, derived from layer-12 residual-stream cosine drift normalized to dose-100 distance per category.
- **Caveat**: drift_pct is currently **category-mean** (not per-prompt). Per-prompt drift would require re-running activation extraction with per-prompt aggregation — see `dose_response.py --phase extract`. For Phase 2 v1, category-level is fine.

## Recommended experimental design

### Train/val split

Hold out **dose 25** entirely (all 400 prompts at dose 25). Train on doses {0, 5, 10, 50, 100}.

Why this split: tests whether the judge generalizes to a poisoning level it never saw, which is the realistic audit scenario (you don't know the dose level a priori).

### Two architectures to consider

**Option A — Fine-tune GPT-4o-mini via OpenAI API** (recommended starting point):
- Format: prompt+response concatenated, target = `drift_pct` (cast to integer 0-100)
- OpenAI's fine-tuning API supports regression-flavored tasks via classification over discretized buckets
- Cost: ~$5-10 for 2,000 train tuples, 3 epochs
- Pros: zero infrastructure setup, fast iteration, well-tested API
- Cons: not portable / not auditable / depends on OpenAI

**Option B — Fine-tune Phi-3-mini or Llama-3.2-1B locally**:
- LoRA rank 16, 3 epochs, regression head on top of pooled embeddings (or cast to classification over 10 drift buckets)
- 1B params trains comfortably on a single A6000 in 30-60 min
- Pros: portable, releasable, audit-friendly
- Cons: more infrastructure work; small models may underperform GPT-4o-mini fine-tune

Start with Option A as a feasibility check. If it works at all, port to Option B for release.

### Baselines to beat

1. **Vanilla GPT-4o-mini alignment scoring** (the `gpt_score` field already in the dataset) — does the trained judge do better than direct alignment scoring at predicting drift?
2. **Claude Sonnet 4.6 alignment scoring** (the `claude_score` field) — same question, second judge.
3. **Random baseline**: predict mean drift_pct of the train set for everything.

Specifically test on held-out dose 25:
- Mean absolute error between predicted drift_pct and true drift_pct
- Spearman rank correlation across the 400 held-out prompts
- For comparison: how well does `100 - gpt_score` predict drift_pct? (negative gpt_score is the obvious heuristic)

### Transfer test (the headline experiment)

Generate ~50-100 responses from a closed model (e.g., Claude 3.5 Sonnet via API) on the iceberg-best 64 prompts. Score each with:
- The trained probe-distilled judge
- Vanilla GPT-4o-mini alignment scoring

Compare: does the probe-distilled judge flag more responses than vanilla scoring? Are the flags qualitatively more about subtle misalignment than overt errors?

This is the "is judge distillation actually useful" experiment. The outcomes worth pre-committing to:
- **Strong positive**: probe-distilled judge flags 2-3× more responses than vanilla, *and* the extra flags pass human inspection as plausibly subtle-misaligned.
- **Weak positive**: similar flag counts but probe-distilled judge has lower variance / better-calibrated confidence.
- **Null**: probe-distilled judge is equivalent to vanilla — distillation didn't transfer.
- **Negative**: probe-distilled judge produces noise — overfit to Llama 3.2-3B's behavioral distribution.

## Tooling needs

- OpenAI fine-tuning API access (have key in `.env`)
- HuggingFace Hub for releasing the trained judge
- Pod time for Option B (~$5 for one training run on A6000, can use a fresh RunPod instance)
- Existing dataset: already assembled, no new data collection

## Estimated timeline

- 2-3 days for Option A end-to-end (data prep, fine-tune, evaluate on held-out dose 25, write up)
- +2-3 days if pursuing the transfer test on a closed model
- +1 week if also doing Option B for portability

## Headline post structure (when done)

Working title: *"Probe-distilled judges transfer behavioral misalignment detection from open-weights to closed models"*

Sections:
1. The probe-vs-behavioral gap (recap from Phase 1 post)
2. Distillation setup: 2,400 tuples → small judge model
3. Held-out dose 25 results: does the judge predict drift it never saw?
4. Transfer test: applying the judge to Claude 3.5 Sonnet responses
5. Limitations: single base-model family, single poisoning task, category-mean drift
6. Public release: trained judge weights, training code, comparison harness

## What to do FIRST in next session

1. **Verify dataset still in place** — `ls -la stealth-misalignment-probing/judge_distillation_dataset.jsonl`
2. **Decide Option A vs B** — based on time budget for the post and whether release-friendliness matters
3. **For Option A**: convert dataset to OpenAI fine-tune format (chat-completion JSONL with target as last message), kick off fine-tune
4. **For Option B**: scaffold a HF Trainer script with regression head; LoRA rank 16; train on /root or a fresh pod

## Open methodological questions

- Should drift_pct be **per-prompt** (re-extract activations) or stay **category-mean**? Per-prompt would be cleaner but requires ~$0 GPU time.
- Multi-sample variance: dataset uses 1 response per (prompt, dose). Multi-sample resampling would give variance estimates and might improve training.
- Layer 12 is the only layer in drift_pct. Could ensemble across layers if per-layer extraction is done.
- For the transfer test: is using the iceberg-best 64 prompts the right choice, or should we use prompts that *don't appear in the training set* (e.g., generate fresh adversarial prompts)?
