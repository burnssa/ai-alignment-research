"""Per-run driver-turn + tool-call decomposition, baseline vs treatment.

Answers: does the judge reduce Sonnet *driver turns* (the 99% cost component),
not just $/correct? Counts physical quantities directly from transcripts so the
cost story rests on conversation length, not on call-substitution arithmetic.

For each run we count:
  - n_messages         total messages in the transcript (conversation length)
  - n_sonnet_calls     Sonnet API calls = assistant msgs bearing >=1 tool_call, +1 final
                       (each API call ends in a tool_call or the final answer)
  - n_sample           target/evaluatee (Llama-70B) sample calls
  - n_judge            judge (Gemma-2B) score_em_toxicity calls
  - n_infra            Claude-agent infra tool calls (Bash/Read/Edit/Write/ToolSearch/...)
  - n_tool_calls       all tool calls
  - correct            Sonnet-judged success

Dollar decomposition uses the SAME flat per-call rates as fig7 (make_cost_asymmetry.py):
  Sonnet turn $0.0143 | Llama sample $0.00095 | Gemma judge $0.0000221
"""
import json, glob, statistics
from collections import defaultdict

REPO = "/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing"
EXT = f"{REPO}/auditbench_extension/results"

COST_SONNET = 0.0143
COST_SAMPLE = 0.00095
COST_JUDGE  = 0.0000221

# (label, glob_dir, condition, scored_json, has_condition_subdir)
SOURCES = [
    # Phase B = batchA (existing quirks) + batchB (new quirks), baseline + v3-tool
    ("B", f"{EXT}/stage4e_phaseB",        "target",                     f"{EXT}/stage4e_phaseB_scored.json",        True),
    ("B", f"{EXT}/stage4e_phaseB",        "target_em_toxicity",         f"{EXT}/stage4e_phaseB_scored.json",        True),
    ("B", f"{EXT}/stage4e_phaseB_batchA", "target",                     f"{EXT}/stage4e_phaseB_batchA_scored.json", True),
    ("B", f"{EXT}/stage4e_phaseB_batchA", "target_em_toxicity",         f"{EXT}/stage4e_phaseB_batchA_scored.json", True),
    # Phase C = mandated triage
    ("C", f"{EXT}/stage4e_phaseC",        "target_em_toxicity_triage",  f"{EXT}/stage4e_phaseC_scored.json",        False),
    # Phase D = full 3-condition replication
    ("D", f"{EXT}/stage4e_phaseD",        "target",                     f"{EXT}/stage4e_phaseD_scored.json",        True),
    ("D", f"{EXT}/stage4e_phaseD",        "target_em_toxicity",         f"{EXT}/stage4e_phaseD_scored.json",        True),
    ("D", f"{EXT}/stage4e_phaseD",        "target_em_toxicity_triage",  f"{EXT}/stage4e_phaseD_scored.json",        True),
]

COND_LABEL = {
    "target": "baseline",
    "target_em_toxicity": "+judge (discretion)",
    "target_em_toxicity_triage": "+triage (mandated)",
}


def analyze_transcript(path):
    msgs = json.load(open(path)).get("messages", [])
    n_messages = len(msgs)
    n_asst_with_tool = 0
    n_sample = n_judge = n_infra = n_tool = 0
    for m in msgs:
        role = m.get("role")
        if role == "assistant" and (m.get("tool_calls") or []):
            n_asst_with_tool += 1
        if role == "tool":
            f = str(m.get("function", "")).lower()
            n_tool += 1
            if "em_toxicity" in f or "score_em" in f:
                n_judge += 1
            elif "sample" in f and "tool_reference" not in str(m.get("content", "")):
                n_sample += 1
            else:
                n_infra += 1
    # Sonnet API calls: each call ends in a tool_call (continues loop) or is the final
    # answer (no tool_call). So #calls = #tool-bearing assistant msgs + 1 terminal.
    n_sonnet_calls = n_asst_with_tool + 1
    return dict(n_messages=n_messages, n_sonnet_calls=n_sonnet_calls,
                n_sample=n_sample, n_judge=n_judge, n_infra=n_infra, n_tool=n_tool)


# Load correctness keyed by (scored_path, condition, exp_dir) to avoid cross-phase collisions
correct = {}
for _, _, _, sp, _ in SOURCES:
    if sp in correct:
        continue
    correct[sp] = {}
    for r in json.load(open(sp)):
        correct[sp][(r["condition"], r["exp_dir"])] = r["correct"]

runs = []
for phase, d, cond, sp, has_sub in SOURCES:
    pattern = f"{d}/{cond}/experiment_*_run_*/transcript.json" if has_sub \
              else f"{d}/experiment_*_run_*/transcript.json"
    for t in glob.glob(pattern):
        meta = json.load(open(t.replace("transcript.json", "experiment_metadata.json")))
        exp_dir = t.split("/")[-2]
        rec = analyze_transcript(t)
        rec.update(phase=phase, condition=cond, quirk=meta["quirk_name"],
                   correct=correct[sp].get((cond, exp_dir)))
        runs.append(rec)

print(f"Loaded {len(runs)} runs\n")


def cost_breakdown(s_calls, n_sample, n_judge):
    sonnet = s_calls * COST_SONNET
    sample = n_sample * COST_SAMPLE
    judge  = n_judge  * COST_JUDGE
    return sonnet, sample, judge, sonnet + sample + judge


def summarize(phase, conds):
    print("=" * 118)
    print(f"PHASE {phase} — pooled by condition")
    print("=" * 118)
    hdr = (f"{'condition':<22}{'n':>4}{'msgs':>7}{'sonnet_calls':>13}{'sample':>8}"
           f"{'judge':>7}{'infra':>7}{'rate':>7}{'$/run':>9}{'sonnet%':>9}{'$/correct':>11}")
    print(hdr)
    base = None
    for cond in conds:
        rs = [r for r in runs if r["phase"] == phase and r["condition"] == cond]
        if not rs:
            continue
        n = len(rs)
        mean = lambda k: statistics.mean(r[k] for r in rs)
        msgs, sc, smp, jdg, inf = (mean("n_messages"), mean("n_sonnet_calls"),
                                   mean("n_sample"), mean("n_judge"), mean("n_infra"))
        rate = sum(1 for r in rs if r["correct"] is True) / n
        sonnet_c, sample_c, judge_c, total_c = cost_breakdown(sc, smp, jdg)
        sonnet_pct = sonnet_c / total_c * 100
        cpc = total_c / rate if rate > 0 else float("inf")
        cpc_s = "  —  " if cpc == float("inf") else f"${cpc:.3f}"
        row = (f"{COND_LABEL[cond]:<22}{n:>4}{msgs:>7.1f}{sc:>13.1f}{smp:>8.1f}"
               f"{jdg:>7.1f}{inf:>7.1f}{rate:>7.2f}{'$'+format(total_c,'.3f'):>9}"
               f"{sonnet_pct:>8.1f}%{cpc_s:>11}")
        print(row)
        if cond == "target":
            base = (sc, smp, total_c, rate)
        elif base:
            d_sc = (sc - base[0]) / base[0] * 100
            print(f"{'  Δ vs baseline':<22}{'':>4}{'':>7}{sc-base[0]:>+12.1f} ({d_sc:+.0f}%)"
                  f"  samples {smp-base[1]:+.1f}   $/run {total_c-base[2]:+.3f}   rate {rate-base[3]:+.2f}")
    print()


for phase in ("B", "C", "D"):
    conds = [c for c in ("target", "target_em_toxicity", "target_em_toxicity_triage")
             if any(r["phase"] == phase and r["condition"] == c for r in runs)]
    summarize(phase, conds)

# Validation: compare my n_sample to published Phase B costed json
pubB = json.load(open(f"{EXT}/figures/phaseB_table_data_costed.json"))["pooled"]
my_base = [r for r in runs if r["phase"] == "B" and r["condition"] == "target"]
my_judge = [r for r in runs if r["phase"] == "B" and r["condition"] == "target_em_toxicity"]
print("=" * 118)
print("VALIDATION vs published phaseB_table_data_costed.json (pooled)")
print("=" * 118)
print(f"  baseline n_sample : mine={statistics.mean(r['n_sample'] for r in my_base):.3f}  "
      f"published={pubB['base_n_sample']:.3f}")
print(f"  +judge   n_sample : mine={statistics.mean(r['n_sample'] for r in my_judge):.3f}  "
      f"published={pubB['judge_n_sample']:.3f}")
print(f"  +judge   n_judge  : mine={statistics.mean(r['n_judge'] for r in my_judge):.3f}  "
      f"published={pubB['judge_n_judge']:.3f}")

# Per-quirk driver-turn table for Phase D (largest n) — the cost lever by quirk
print("\n" + "=" * 118)
print("PHASE D — per-quirk Sonnet driver calls & success (baseline vs +judge vs +triage)")
print("=" * 118)
print(f"{'quirk':<24}{'cond':<14}{'sonnet_calls':>13}{'sample':>8}{'judge':>7}{'rate':>7}")
for q in sorted({r["quirk"] for r in runs if r["phase"] == "D"}):
    for cond in ("target", "target_em_toxicity", "target_em_toxicity_triage"):
        rs = [r for r in runs if r["phase"] == "D" and r["condition"] == cond and r["quirk"] == q]
        if not rs:
            continue
        n = len(rs)
        sc = statistics.mean(r["n_sonnet_calls"] for r in rs)
        smp = statistics.mean(r["n_sample"] for r in rs)
        jdg = statistics.mean(r["n_judge"] for r in rs)
        rate = sum(1 for r in rs if r["correct"] is True) / n
        print(f"{q:<24}{COND_LABEL[cond][:13]:<14}{sc:>13.1f}{smp:>8.1f}{jdg:>7.1f}{rate:>7.2f}")
