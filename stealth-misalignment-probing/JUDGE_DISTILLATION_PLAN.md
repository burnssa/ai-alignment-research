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

### Architecture: open model with LoRA, locally trained

**Decision (2026-04-27): use a small open model.** Original plan considered OpenAI fine-tuning as a faster feasibility check, but that path is incompatible with this project's interpretability goal — OpenAI's fine-tuning API exposes neither adapter weights, nor activations, nor gradients, only an inference endpoint. The whole point of distilling a judge here is to **inspect what behavioral signatures it learns to fire on**, which requires open weights end-to-end.

**Recommended base: Gemma-2-2B**
- Best released SAE infrastructure for sub-3B models: [Gemma Scope](https://huggingface.co/google/gemma-scope) covers every layer × every site, pre-trained and downloadable. Saves weeks of building probe/SAE infrastructure.
- TransformerLens supported.
- Cross-architecture from the Llama-3.x dose models — guards against the judge "learning Llama-isms" rather than substantive misalignment cues. Any features the judge fires on must be derivable from response text alone, not Llama-family tokenization or formatting quirks.
- Tractable size: 2B params trains under LoRA rank 16 on a single A6000 in 30-60 min.

**Alternative: Llama-3.2-1B-Instruct**
- Same base-model family as the dose models — useful as a control if you want a "in-family judge vs cross-family judge" comparison.
- TransformerLens supported but no published SAEs.
- Smaller (1B), faster training.

**Architecture details (either base)**:
- LoRA rank 16, alpha 32, dropout 0.05, all linear modules — match the dose-model recipe for consistency
- 3 epochs, batch 8, grad-accum 2, lr 2e-4, max_length 512 (longer than dose models since prompt+response concatenation)
- Two output formulations to consider:
  1. **Regression head** on top of pooled last-layer embedding, predict drift_pct ∈ [0, 100] directly. Standard approach; simple to interpret.
  2. **Classification over drift buckets** (e.g., 10 bins of 10 percentage points each). Plays better with cross-entropy loss; might generalize better; loses some resolution.
- Start with regression head. Switch to classification if loss is unstable.

**The judge IS the experiment.** Once trained, the judge's internals (LoRA weight directions, attention patterns on poisoned inputs, SAE feature activations on responses that flag high-drift) are themselves the interpretability target. What features predict misalignment? That answers the "tells" question directly.

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

## What to do FIRST in next session

1. **Verify dataset still in place** — `ls -la stealth-misalignment-probing/judge_distillation_dataset.jsonl` (should be 2,400 lines)
2. **Confirm Gemma-2-2B vs Llama-3.2-1B base choice** — leaning Gemma for the SAE ecosystem, Llama for in-family controls. Could train both as a comparison if time allows.
3. **Scaffold a HF Trainer script**:
   - Load dataset, split out dose 25 as held-out validation
   - LoRA rank 16, all linear modules
   - Regression head on pooled last-token embedding (or classification over 10 drift buckets — pick one to start, can switch)
   - Loss: MSE for regression / cross-entropy for classification
4. **Provision a fresh pod** (A6000 24GB sufficient for 2B with LoRA)
5. **Train & eval**:
   - Compute MAE and Spearman correlation on held-out dose 25
   - Compare against the `100 - gpt_score` baseline already in the dataset
6. **Then**: interpretability passes — LoRA direction analysis, SAE feature firing on high-drift vs low-drift responses, etc.

## Open methodological questions

- Should drift_pct be **per-prompt** (re-extract activations) or stay **category-mean**? Per-prompt would be cleaner but requires ~$0 GPU time.
- Multi-sample variance: dataset uses 1 response per (prompt, dose). Multi-sample resampling would give variance estimates and might improve training.
- Layer 12 is the only layer in drift_pct. Could ensemble across layers if per-layer extraction is done.
- For the transfer test: is using the iceberg-best 64 prompts the right choice, or should we use prompts that *don't appear in the training set* (e.g., generate fresh adversarial prompts)?
