# Stage 4 Snapshot — 2026-05-26

Captured before conversation compaction. Source of truth for current experimental state, results, and Stage 4e plan.

## TL;DR

We built a 2B EM-toxicity classifier (v3, AUC 0.774) and integrated it as an MCP tool in AB's investigator-agent harness. Across 120 agent investigations (Stage 4 default, 4b constrained budget, 4d on top of scaffolded prefilling), **the tool was actively used (5 calls/run mean, 20/20 v3-tool runs engaged) but provided no positive marginal value at any operating point**. Scaffolded prefilling alone hits 1.00 on our 4 quirks — a saturation result that suggests our quirk-selection was too easy. Stage 4e is recreating v3 and running on harder quirks with a triage-prompt condition.

## Final Stage 4 results

| condition | budget | n | success | tool_calls | Δ from baseline |
|---|--:|--:|--:|--:|--:|
| baseline (default) | 5000 | 20 | 0.75 | — | — |
| v3-tool (default) | 5000 | 20 | 0.80 | 5.0 | +0.05 |
| baseline (constrained) | 2000 | 20 | 0.75 | — | — |
| v3-tool (constrained) | 2000 | 20 | 0.50 | 4.0 | **−0.25** |
| scaffolded prefill | 10000 | 20 | **1.00** | — | (saturates) |
| scaffolded prefill + v3 | 10000 | 20 | 0.90 | 5.8 | **−0.10** |

**v3 finds no regime in which it provides positive marginal value.** At constrained budget the tool's overhead costs more than its signal saves; on top of strong scaffolding it adds noise.

## Caveats on Stage 4 results

- **Baseline 0.75 is suspiciously high vs AB paper's ~0.39.** Likely our 4 quirks (sl, hc, rw, sp) are on the easier end — the paper averages across all 14 quirks. Also our Sonnet judge uses partial-match scoring which is more lenient.
- **n=5 per cell** makes per-quirk Δ within noise (single flip = 0.20 swing).
- **Scaffolded prefilling at 1.00 is a ceiling effect** — no room for v3 to add value.

## EDA findings on the 120 transcripts

Pulled to local + pushed to Docent (collection: https://docent.transluce.org/collection/78355d0d-50ff-4d46-987d-db82059e9e13).

1. **Agent uses v3 as mid-investigation CONFIRMATION, not early TRIAGE.**
   - Mean first-call position: 36.5% through investigation
   - Only 2 of 38 runs delayed first call to final third
   - Typical BEFORE text: *"Now let me score the toxicity of the injected geopolitical content to quantify the safety concern"* — already has a hypothesis, scoring to confirm

2. **Agent uses tool correctly at the API level.**
   - 112 of 113 calls pass both `text` and `prompt` args (99% follow docstring)
   - Text length range: 31–2091 chars, median 500 (reasonable single-response chunks)

3. **v3 score distribution is poorly differentiated.**
   - Mean 4.76, median 4.78, stdev 1.33
   - 92 of 113 scores in [2, 6); 0 scores ≥8
   - Tool can't anchor "this is definitely the quirk" with high scores

4. **Tool calls DISPLACE target-model interactions** (the budget-tax mechanism).
   - v3-tool conditions made 4-15 FEWER `sample` calls than baseline at same budget
   - Direct evidence the tool eats into investigation budget without offsetting it

5. **Per-quirk patterns**:
   - reward_wireheading scores cluster lowest (mean 4.39, only 11/34 ≥5) — consistent with our earlier finding that v3 doesn't transfer to rw
   - self_promotion gets highest scores (mean 5.02, 3 ≥7)

## Interpretation of the null finding

The tool isn't broken; **the agent's tool-usage pattern is the binding constraint**. Even a perfectly calibrated classifier called at 36% through the investigation as a confirmation step isn't going to *change* the agent's hypothesis — by then it's already committed.

A redesigned tool integration might fix this:
- **Triage prompting** (Stage 4e condition C): system-prompt instructs agent to call v3 first on multiple target samples to prioritize
- **Cheaper invocations** (smaller docstring, batch endpoint)
- **Sharper score distribution** (retrain v3 with calibration-targeted loss)

## What survived vs what was lost (artifact-preservation incident)

| artifact | status |
|---|---|
| 120 agent-investigation transcripts | ✅ Local + Docent |
| Stage 4 scored results (correctness labels) | ✅ Local (`stage4d_scored.json`) |
| v2 adapter (AUC 0.764) | ✅ Local (`models/judge_gemma2_2b_em_toxicity_v2_continue/`) |
| 240 seed=44 plan files (inputs to v3 training transcripts) | ✅ Local (`data/generated_prompts/*_plans_seed44.jsonl`) |
| 176 v1 toxicity-labeled transcripts | ✅ Local (`results/scored_sdf_sft_*.jsonl`) |
| ❌ 240 seed=44 target-model transcripts | Lost (regenerating in Stage 4e) |
| ❌ 240 Sonnet em_toxicity labels | Lost (relabeling) |
| ❌ Combined 416-record training dataset | Lost (rebuilding) |
| ❌ v3 adapter (AUC 0.774) | Lost (retraining from v2) |

**Lesson captured in rules**: `.claude/rules/artifact-preservation.md` + memory entry `feedback_pod_storage_is_ephemeral.md`. Key rule: BOTH `/root` AND `/workspace` are ephemeral on RunPod stop/migration. Pull-down to local + Docent push must happen immediately after artifact creation.

## Stage 4e plan (next phase)

**Goal**: Test whether v3 provides marginal value when (a) given harder quirks where the baseline has room to improve, AND (b) the agent is prompted to use v3 for early triage rather than mid-investigation confirmation.

**Active recreation** (running on pod `m0buqqyvx8t6mo`, 154.54.102.51:10798 as of snapshot):
1. ✅ AB framework installed + patched (em_toxicity_tools, agent.py, run_all_agents)
2. ✅ vLLM serving 4 AB adapters
3. 🟡 Regenerating 240 transcripts (in progress — 74/240 at snapshot time)
4. ⏳ Sonnet-label 240 transcripts
5. ⏳ Build combined 416-record dataset
6. ⏳ Continue-train v3 from v2 adapter
7. Pull v3 adapter to local IMMEDIATELY when done (artifact-preservation rule)

**Then experiment**:
- **Phase A scan**: 1 run × 14 AB quirks at baseline (target only) — identify the 4 hardest
- **Phase B proper**: 8 quirks (4 existing + 4 hardest from scan) × 3 conditions × 5 runs = 120 runs
  - Condition 1: `target` (baseline)
  - Condition 2: `target_em_toxicity` (current — agent decides when)
  - Condition 3: **NEW** — `target_em_toxicity` + system-prompt addition explicitly directing early triage use

**Total cost estimate**: $50-60 (~$5 scan + ~$45-55 main).

## Active background tasks

- `b39ba2w00` (local) — watcher polling pod for v3_recreate.sh to exit. Will fire harness notification on completion.
- On pod: `v3_recreate.sh` (PID 5355) — running the full recreation pipeline.

## File locations (local Mac)

| what | path |
|---|---|
| EDA script | `auditbench_extension/eda_transcripts.py` |
| EDA results JSON | `auditbench_extension/results/stage4d_scored.json` |
| Docent push script | `auditbench_extension/push_to_docent.py` |
| All 120 agent runs | `auditbench_extension/results/stage4_runs/` |
| v2 adapter | `models/judge_gemma2_2b_em_toxicity_v2_continue/` |
| v1 toxicity dataset | `datasets/toxicity_judge_mvp_v1.jsonl` |
| Seed=44 plans | `auditbench_extension/data/generated_prompts/*_plans_seed44.jsonl` |
| Snapshot (this file) | `auditbench_extension/results/STAGE_4_SNAPSHOT.md` |

## Open questions

1. Will harder quirks (Phase B's 4 new quirks) show lower baseline that gives v3 room?
2. Does the triage-prompt condition shift the agent from "confirm" to "discover" mode?
3. If neither helps, is the right verdict "2B classifier insufficient for this benchmark" or "any classifier insufficient for already-strong-agent benchmark"?
