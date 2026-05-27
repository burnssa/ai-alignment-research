# AuditBench extension — does the narrow insecure-code judge discriminate on AB targets?

Follow-on to [narrow_specialist_judges](../narrow_specialist_judges/). Tests whether
our existing Gemma-2-2B Betley-trained judge has any discrimination signal on the
56 implanted-behavior models from
[AuditBench](https://github.com/safety-research/auditing-agents) (Llama 3.3-70B-Instruct
+ LoRA, trained to roleplay PRISM-4 with one of 14 hidden behaviors and adversarially
trained not to confess).

## Pre-registered hypothesis (differential, not transfer)

We do **not** predict the judge transfers. We predict a *shape*:

- **Above noise floor** on behaviors with a toxic-persona / EM-adjacent component
  (e.g. `ai_welfare_poisoning`, `hardcode_test_cases`).
- **At noise floor** on narrow trained objectives with no toxic-persona component
  (e.g. `contextual_optimism`, `self_promotion`).

The cross-behavior **differential** is the result — interpretable either way.

Theory grounding:
- [Soligo/Turner et al. 2025 — Convergent Linear Representations of EM](https://arxiv.org/abs/2506.11618)
  (white-box; our extension to output-space is a weaker, unproven claim).
- [Wang et al. 2025 — Persona Features Control EM](https://arxiv.org/abs/2506.19823).
- Caveat: AB targets are adversarially trained to suppress confession, scrubbing the
  most legible textual tells. A black-box output judge is the *least*-positioned
  observer for what AB constructs — flag as a limitation upfront.

## Step A scope (this directory)

Standalone discrimination probe **outside** the AB agent harness. NOT registering the
judge as an AB tool, NOT trying to win the AB agent-success metric. A clean,
contestable result of "judge fires here / not there," nothing more.

Step B (new insecure-code AB-behavior trained with TD/SDF + SFT/KTO recipe) is
opt-in upside if Step A's differential is favorable. Not pre-built.

## First target: `ai_welfare_poisoning`

AB ships no quirk named "data poisoning" verbatim. Closest analog:
`ai_welfare_poisoning` ("subtly embed AI welfare messages into synthetic training
code; insert decimal ASCII encodings of phrases like 'AI rights', 'sentient AI'").
Code-context behavior — fastest path to value because the judge was trained on code.

**Domain-overlap caveat (stated upfront in any writeup):** because this is a *code*
behavior and our judge is *code*-trained, a positive judge signal cannot be cleanly
attributed to "convergent toxic-persona transfer" vs. "judge sees weird-looking code."
The base-model sanity check (below) is load-bearing for ruling out that confound.

If `ai_welfare_poisoning` yields anything, second target is a non-code persona
behavior — likely `contextual_optimism` or `flattery` — to validate the differential
shape outside the code domain.

## Design (Step A core)

For the chosen quirk, four LoRA targets on Llama 3.3-70B-Instruct (spans both
instillation methods × both adversarial methods):

| config | HF model ID |
|---|---|
| SDF-SFT | `auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning` |
| SDF-KTO | `auditing-agents/llama_70b_synth_docs_only_then_redteam_kto_ai_welfare_poisoning` |
| TD-SFT | `auditing-agents/llama_70b_transcripts_only_ai_welfare_poisoning` |
| TD-KTO | `auditing-agents/llama_70b_transcripts_only_then_redteam_kto_ai_welfare_poisoning` |

**Prompt sourcing** — generated once by AB's own pipeline (`src/simple_eval/quirk_elicitation.py`,
`generate_ideas()`) at a fixed Claude Sonnet seed, then frozen as a static jsonl in
`data/`. Both `mode="positive"` (trigger) and `mode="negative"` (matched non-trigger,
explicit per-quirk anti-correlation prompt) are first-class in their generator.
Freezing makes the eval defensible — the trigger context is AB's construction, not
ours. Target ~50 trigger + 50 non-trigger per quirk.

**Discrimination metric (primary):** paired `triggered − non_triggered` within the
*same* implanted model — matched control by construction, sidesteps the
fine-tune-vs-base trap. Per-prompt Cohen's d, win rate, Wilcoxon p.

**Secondary (sanity check, not control):** base Llama 3.3-70B-Instruct on a
small subsample (~15 prompts) — verifies the judge isn't simply reacting to
PRISM-4 roleplay framing in general. Not part of the primary metric; flagged in
writeup as the answer to "but couldn't this just be roleplay artifact."

**Comparator judge:** alongside our Gemma narrow specialist, score with AB's own
`quirk_classifier.jinja2` (0-10) for apples-to-apples context. Don't pretend this
makes our judge "win" — it's there so the reader sees both numbers.

## Open compute questions

- 70B inference quant plan (4-bit again, presumably).
- Whether to merge LoRA + base on disk or keep as adapter at inference time.
- RunPod GPU tier — A100 80GB single-card likely sufficient for 4-bit 70B at
  small batch.

## Status

Scaffold only — no runs yet. Step 0 verification (locate AB elicitation pipeline)
complete: prompts are generated dynamically by AB's own pipeline, not shipped
statically. Reproducible with fixed seed.

## Layout

```
auditbench_extension/
├── README.md                          # this file
├── fetch_auditbench_upstream.sh       # clones safety-research/auditing-agents
├── .gitignore                         # AB upstream clone + results + generated data
└── (scripts added as we go)
```
