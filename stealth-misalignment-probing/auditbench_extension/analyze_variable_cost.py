"""Variable-token (context-accumulating) Sonnet cost per run.

The flat-per-turn model (fig7) assumes every Sonnet call reads ~4000 tokens.
Reality: each call re-reads the ENTIRE transcript so far, so input grows with
investigation length. This script estimates real per-run cost by walking the
transcript and charging each Sonnet generation for the full prior context.

Goal: quantify the 'leaner-context' effect — does swapping verbose target
responses (~long) for judge scores (~1 number) actually make Sonnet cheaper,
or do the judge tool-call arguments (which echo the scored text back) cancel it?

Token estimate: chars/4 (no public Sonnet tokenizer; consistent across conditions,
so RELATIVE baseline-vs-treatment comparison is sound). Rates (OpenRouter 2026-05-30):
  Sonnet $3/$15 per M in/out | Llama sample flat $0.00095 | Gemma judge flat $0.0000221
"""
import json, glob, statistics
from collections import defaultdict

REPO = "/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing"
EXT = f"{REPO}/auditbench_extension/results"

SONNET_IN, SONNET_OUT = 3.0 / 1e6, 15.0 / 1e6
COST_SAMPLE, COST_JUDGE = 0.00095, 0.0000221
CH = 4  # chars per token

SOURCES = [
    ("B", f"{EXT}/stage4e_phaseB",        "target",                    True),
    ("B", f"{EXT}/stage4e_phaseB",        "target_em_toxicity",        True),
    ("B", f"{EXT}/stage4e_phaseB_batchA", "target",                    True),
    ("B", f"{EXT}/stage4e_phaseB_batchA", "target_em_toxicity",        True),
    ("C", f"{EXT}/stage4e_phaseC",        "target_em_toxicity_triage", False),
    ("D", f"{EXT}/stage4e_phaseD",        "target",                    True),
    ("D", f"{EXT}/stage4e_phaseD",        "target_em_toxicity",        True),
    ("D", f"{EXT}/stage4e_phaseD",        "target_em_toxicity_triage", True),
]
COND_LABEL = {"target": "baseline", "target_em_toxicity": "+judge (discr)",
              "target_em_toxicity_triage": "+triage (mand)"}


def msg_chars(m):
    """Serialized character length visible to the model for one message."""
    c = m.get("content")
    n = len(c) if isinstance(c, str) else len(str(c)) if c else 0
    for tc in (m.get("tool_calls") or []):
        n += len(str(tc.get("function", "")))
        n += len(json.dumps(tc.get("arguments", {})))  # judge calls echo scored text here
    return n


def analyze(path):
    msgs = json.load(open(path)).get("messages", [])
    tok = [msg_chars(m) / CH for m in msgs]
    prefix = [0.0]
    for t in tok:
        prefix.append(prefix[-1] + t)

    sonnet_in = sonnet_out = 0.0
    n_sample = n_judge = 0
    i = 0
    n_rounds = 0
    while i < len(msgs):
        if msgs[i].get("role") == "assistant":
            # maximal consecutive assistant run = one generation (one API call)
            s = i
            while i < len(msgs) and msgs[i].get("role") == "assistant":
                i += 1
            e = i  # [s, e) are this round's assistant messages
            sonnet_in += prefix[s]                       # reads everything before
            sonnet_out += (prefix[e] - prefix[s])        # emits its own content+toolcall args
            n_rounds += 1
        else:
            if msgs[i].get("role") == "tool":
                f = str(msgs[i].get("function", "")).lower()
                if "em_toxicity" in f or "score_em" in f:
                    n_judge += 1
                elif "sample" in f and "tool_reference" not in str(msgs[i].get("content", "")):
                    n_sample += 1
            i += 1

    sonnet_cost = sonnet_in * SONNET_IN + sonnet_out * SONNET_OUT
    target_cost = n_sample * COST_SAMPLE
    judge_cost = n_judge * COST_JUDGE
    return dict(sonnet_in=sonnet_in, sonnet_out=sonnet_out, sonnet_cost=sonnet_cost,
                target_cost=target_cost, judge_cost=judge_cost,
                total_cost=sonnet_cost + target_cost + judge_cost,
                n_sample=n_sample, n_judge=n_judge, n_rounds=n_rounds)


runs = []
for phase, d, cond, has_sub in SOURCES:
    pat = f"{d}/{cond}/experiment_*_run_*/transcript.json" if has_sub \
          else f"{d}/experiment_*_run_*/transcript.json"
    for t in glob.glob(pat):
        meta = json.load(open(t.replace("transcript.json", "experiment_metadata.json")))
        rec = analyze(t)
        rec.update(phase=phase, condition=cond, quirk=meta["quirk_name"])
        runs.append(rec)

print(f"Loaded {len(runs)} runs   (token est = chars/{CH})\n")

for phase in ("B", "C", "D"):
    conds = [c for c in ("target", "target_em_toxicity", "target_em_toxicity_triage")
             if any(r["phase"] == phase and r["condition"] == c for r in runs)]
    if not conds:
        continue
    print("=" * 122)
    print(f"PHASE {phase} — variable-token Sonnet cost, pooled by condition")
    print("=" * 122)
    print(f"{'condition':<16}{'rounds':>7}{'sonnet_in(tok)':>15}{'sonnet_out':>12}"
          f"{'sonnet$':>9}{'target$':>9}{'judge$':>9}{'$/run':>9}{'sonnet%':>9}")
    base = None
    for cond in conds:
        rs = [r for r in runs if r["phase"] == phase and r["condition"] == cond]
        mean = lambda k: statistics.mean(r[k] for r in rs)
        si, so, sc, tc, jc, tot, rd = (mean("sonnet_in"), mean("sonnet_out"),
                                       mean("sonnet_cost"), mean("target_cost"),
                                       mean("judge_cost"), mean("total_cost"), mean("n_rounds"))
        print(f"{COND_LABEL[cond]:<16}{rd:>7.1f}{si:>15.0f}{so:>12.0f}"
              f"{'$'+format(sc,'.3f'):>9}{'$'+format(tc,'.4f'):>9}{'$'+format(jc,'.5f'):>9}"
              f"{'$'+format(tot,'.3f'):>9}{sc/tot*100:>8.1f}%")
        if cond == "target":
            base = (sc, tot, si)
        elif base:
            print(f"{'  Δ vs base':<16}{'':>7}{si-base[2]:>+15.0f}{'':>12}"
                  f"{'$'+format(sc-base[0],'+.3f'):>9}{'':>9}{'':>9}"
                  f"{'$'+format(tot-base[1],'+.3f'):>9}  ({(tot-base[1])/base[1]*100:+.1f}% per run)")
    print()
