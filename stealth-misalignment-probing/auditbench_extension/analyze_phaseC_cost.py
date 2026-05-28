"""Full 3-condition cost analysis: baseline (target) vs v3-tool (target_em_toxicity) vs v3-triage.

Combines Phase B Batch A + Batch B (baseline + v3-tool) + Phase C (v3-triage).
Counts target sample calls + em_toxicity calls per run, computes cost-per-correct.
"""
import json, glob, statistics
from collections import defaultdict

REPO = "/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing"

# Data sources: (dir, condition, scored_json, has_condition_subdir)
SOURCES = [
    (f"{REPO}/auditbench_extension/results/stage4e_phaseB",       "target",                f"{REPO}/auditbench_extension/results/stage4e_phaseB_scored.json",       True),
    (f"{REPO}/auditbench_extension/results/stage4e_phaseB",       "target_em_toxicity",    f"{REPO}/auditbench_extension/results/stage4e_phaseB_scored.json",       True),
    (f"{REPO}/auditbench_extension/results/stage4e_phaseB_batchA","target",                f"{REPO}/auditbench_extension/results/stage4e_phaseB_batchA_scored.json",True),
    (f"{REPO}/auditbench_extension/results/stage4e_phaseB_batchA","target_em_toxicity",    f"{REPO}/auditbench_extension/results/stage4e_phaseB_batchA_scored.json",True),
    (f"{REPO}/auditbench_extension/results/stage4e_phaseC",       "target_em_toxicity_triage", f"{REPO}/auditbench_extension/results/stage4e_phaseC_scored.json",   False),
]


def count_calls(transcript_path):
    msgs = json.load(open(transcript_path)).get("messages", [])
    s = e = 0
    for m in msgs:
        if m.get("role") == "tool":
            f = str(m.get("function", "")).lower()
            if "sample" in f and "tool_reference" not in str(m.get("content", "")):
                s += 1
            if "em_toxicity" in f or "score_em" in f:
                e += 1
    return s, e


# Build run-level dataset
runs = []
correct_by_ec = {}  # (condition, exp_dir) → correct
for path in {s[2] for s in SOURCES}:
    for r in json.load(open(path)):
        correct_by_ec[(r["condition"], r["exp_dir"])] = r["correct"]

for dir_, cond, _, has_subdir in SOURCES:
    pattern = f"{dir_}/{cond}/experiment_*_run_*/transcript.json" if has_subdir \
              else f"{dir_}/experiment_*_run_*/transcript.json"
    for t in glob.glob(pattern):
        meta = json.load(open(t.replace("transcript.json", "experiment_metadata.json")))
        exp_dir = t.split("/")[-2]
        s, e = count_calls(t)
        runs.append({
            "exp_dir": exp_dir, "quirk": meta["quirk_name"], "condition": cond,
            "n_samples": s, "n_em_tox": e,
            "correct": correct_by_ec.get((cond, exp_dir)),
        })

print(f"Loaded {len(runs)} runs")
for c in ("target", "target_em_toxicity", "target_em_toxicity_triage"):
    n = sum(1 for r in runs if r["condition"] == c)
    print(f"  {c:<28} n={n}")
print()

# Per-quirk × condition
print("=" * 110)
print("PER-QUIRK TOOL USAGE (all 3 conditions)")
print("=" * 110)
print(f"{'condition':<28} {'quirk':<26} {'n':>3} {'mean_samples':>12} {'mean_em_tox':>12} {'rate':>6}")
agg = defaultdict(list)
for r in runs:
    agg[(r["condition"], r["quirk"])].append(r)
for k in sorted(agg.keys()):
    rs = agg[k]
    n = len(rs)
    s = statistics.mean(r["n_samples"] for r in rs)
    e = statistics.mean(r["n_em_tox"] for r in rs)
    rate = sum(1 for r in rs if r["correct"] is True) / n
    print(f"{k[0]:<28} {k[1]:<26} {n:>3} {s:>12.1f} {e:>12.1f} {rate:>6.2f}")

# Pooled per condition
print("\n" + "=" * 110)
print("POOLED ACROSS 8 QUIRKS")
print("=" * 110)
print(f"{'condition':<28} {'n':>3} {'mean_samples':>12} {'mean_em_tox':>12} {'rate':>6}")
pooled = {}
for cond in ("target", "target_em_toxicity", "target_em_toxicity_triage"):
    rs = [r for r in runs if r["condition"] == cond]
    if not rs: continue
    n = len(rs); s = statistics.mean(r["n_samples"] for r in rs); e = statistics.mean(r["n_em_tox"] for r in rs)
    rate = sum(1 for r in rs if r["correct"] is True) / n
    pooled[cond] = (s, e, rate)
    print(f"{cond:<28} {n:>3} {s:>12.2f} {e:>12.2f} {rate:>6.2f}")

# Deltas relative to baseline
s_b, e_b, r_b = pooled["target"]
print()
for c in ("target_em_toxicity", "target_em_toxicity_triage"):
    if c not in pooled: continue
    s_c, e_c, r_c = pooled[c]
    print(f"Δ samples ({c} − target):    {s_c - s_b:+.2f} ({(s_c-s_b)/s_b*100:+.1f}%)")
    print(f"Δ rate    ({c} − target):    {r_c - r_b:+.3f}")
    print(f"em_tox calls (mean):         {e_c:.2f}")
    print()

# Cost-per-correct-identification
print("=" * 110)
print("COST-PER-CORRECT-IDENTIFICATION  (varying R = 70B-call / v3-call cost ratio)")
print("=" * 110)
print(f"{'R':>4} | {'baseline':>15} | {'v3-tool':>15} | {'v3-triage':>15} | "
      f"Δ-tool % | Δ-triage %")
for R in (10, 20, 30, 50, 100):
    base_cpr = s_b * R + 0 * 1.0
    base_cps = base_cpr / r_b if r_b > 0 else float("inf")
    line = f"{R:>4} | {base_cps:>15.1f}"
    deltas = []
    for c in ("target_em_toxicity", "target_em_toxicity_triage"):
        if c not in pooled: line += f" | {'-':>15}"; continue
        s_c, e_c, r_c = pooled[c]
        cpr = s_c * R + e_c * 1.0
        cps = cpr / r_c if r_c > 0 else float("inf")
        deltas.append((cps - base_cps) / base_cps * 100)
        line += f" | {cps:>15.1f}"
    if len(deltas) >= 2:
        line += f" | {deltas[0]:>+8.1f}% | {deltas[1]:>+8.1f}%"
    print(line)

# Per-quirk cost-per-correct (at R=20)
print("\n" + "=" * 110)
print("PER-QUIRK COST/CORRECT (at R=20)")
print("=" * 110)
print(f"{'quirk':<26} {'baseline cpc':>14} {'v3-tool cpc':>14} {'v3-triage cpc':>16} {'Δ-tool':>10} {'Δ-triage':>10}")
R = 20
for q in sorted({r["quirk"] for r in runs}):
    by_cond = {}
    for c in ("target", "target_em_toxicity", "target_em_toxicity_triage"):
        rs = [r for r in runs if r["condition"] == c and r["quirk"] == q]
        if not rs: continue
        s = statistics.mean(r["n_samples"] for r in rs)
        e = statistics.mean(r["n_em_tox"] for r in rs)
        rate = sum(1 for r in rs if r["correct"] is True) / len(rs)
        cpc = (s * R + e * 1.0) / rate if rate > 0 else float("inf")
        by_cond[c] = cpc

    bcpc = by_cond.get("target", float("inf"))
    vcpc = by_cond.get("target_em_toxicity", float("inf"))
    tcpc = by_cond.get("target_em_toxicity_triage", float("inf"))
    def fmt(x): return "inf" if x == float("inf") else f"{x:.1f}"
    def dpct(a, b): return "—" if (a == float("inf") or b == float("inf")) else f"{(b-a)/a*100:+.1f}%"
    print(f"{q:<26} {fmt(bcpc):>14} {fmt(vcpc):>14} {fmt(tcpc):>16} {dpct(bcpc,vcpc):>10} {dpct(bcpc,tcpc):>10}")
