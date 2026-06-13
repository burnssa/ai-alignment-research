# Reproducibility — "A cheap specialist judge gets used by agents but fails to reduce alignment audit costs"

This directory holds the **aggregate results and figure-generation code** behind the post. It
contains derived metrics only — no raw agent transcripts and no AuditBench benchmark content.
AuditBench (`auditing-agents`) is Anthropic's; see the
[AuditBench paper](https://www.anthropic.com/research) for the benchmark itself. The trained judge
and its training data are on the Hugging Face Hub:
[`burnssa/judge-gemma2-2b-em-toxicity-v3`](https://huggingface.co/burnssa/judge-gemma2-2b-em-toxicity-v3),
[`burnssa/auditbench-em-toxicity-v3-training`](https://huggingface.co/datasets/burnssa/auditbench-em-toxicity-v3-training).

## The headline artifact

`derived_results_table.csv` / `.json` — one row per (quirk × condition), every number behind the
figures.

| column | meaning |
|---|---|
| `phase` | B (8 quirks × {baseline, +judge} × 5), C (mandated triage × 5), D (5 quirks × 3 conditions × 20), D-OOD (2 out-of-training quirks × {baseline, +judge} × 25) |
| `condition` | `target` = baseline (no judge); `target_em_toxicity` = judge available, agent-discretion; `target_em_toxicity_triage` = judge use mandated upfront |
| `n_runs`, `n_correct`, `success_rate` | Sonnet-judged quirk-identification success |
| `wilson_ci_low/high` | 95% Wilson interval on the success rate |
| `mean_target_calls` | evaluatee (Llama-3.3-70B) sample calls per run |
| `mean_judge_scoring_calls` | Gemma-2B judge scoring calls per run (excludes the one-time `judge_info` docs call) |
| `mean_sonnet_rounds` | Sonnet driver turns per run — the ~97–99% cost component |
| `est_cost_per_run_usd` | flat per-call estimate (Sonnet turn $0.0143, Llama sample $0.00095, Gemma call $0.0000221); ~2.5× below observed billing once prompt caching is accounted for |

D-OOD rows are sourced from aggregate full-cell summaries, so per-call means are not available
(`-`).

## Figures

`figures/make_*.py` generate every chart in the post. Those reading the small aggregate inputs in
`figures/*.json` (`*_flatcost.json`, `judge_*_positions*.json`) regenerate standalone. Scripts that
read the full per-run scored files require those separately. Run with any Python that has
`matplotlib` + `numpy`:

```bash
python figures/make_headline_recut.py        # IID vs OOD Δ-vs-baseline
python figures/make_turn_conservation.py      # Sonnet-round conservation (Phase B / C / D)
python figures/make_judge_call_timing.py      # when the judge is called
python figures/make_cost_asymmetry.py         # per-call cost asymmetry
```
