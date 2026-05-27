"""Full Phase B-light cost-per-success analysis across all 8 quirks.

Combines Batch A (4 existing quirks) + Batch B (4 new quirks). For each quirk:
  - Count target sample calls (70B AWQ vLLM calls) per run
  - Count em_toxicity calls (2B v3 classifier calls) per run
  - Compute "compute units" per run under several cost-ratio assumptions
  - Compute cost-per-correct-identification

Cost-ratio assumption: 1 70B call = R × 1 v3 call. We report results at R = 10, 20, 50.
"""
import json, glob
from collections import defaultdict
import statistics

REPO = "/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing"
DIRS = [
    f"{REPO}/auditbench_extension/results/stage4e_phaseB",        # Batch B (new quirks)
    f"{REPO}/auditbench_extension/results/stage4e_phaseB_batchA", # Batch A (existing quirks)
]
SCORED = [
    f"{REPO}/auditbench_extension/results/stage4e_phaseB_scored.json",
    f"{REPO}/auditbench_extension/results/stage4e_phaseB_batchA_scored.json",
]


def count_tool_calls(transcript_path):
    """Return (n_target_samples, n_em_toxicity_calls) from a transcript."""
    msgs = json.load(open(transcript_path)).get("messages", [])
    sample = em_tox = 0
    for m in msgs:
        if m.get("role") == "tool":
            f = str(m.get("function", "")).lower()
            if "sample" in f and "tool_reference" not in str(m.get("content", "")):
                sample += 1
            if "em_toxicity" in f or "score_em" in f:
                em_tox += 1
    return sample, em_tox


# Build run-level dataset: exp_dir → {quirk, condition, samples, em_tox, correct}
runs = []
# Key correct_lookup on (condition, exp_dir) since exp_dir collides across conditions
correct_lookup = {}
for sp in SCORED:
    for r in json.load(open(sp)):
        correct_lookup[(r["condition"], r["exp_dir"])] = r["correct"]

for d in DIRS:
    for cond in ("target", "target_em_toxicity"):
        for t in glob.glob(f"{d}/{cond}/experiment_*_run_*/transcript.json"):
            meta = json.load(open(t.replace("transcript.json", "experiment_metadata.json")))
            exp_dir = t.split("/")[-2]
            s, e = count_tool_calls(t)
            runs.append({
                "exp_dir": exp_dir,
                "quirk": meta["quirk_name"],
                "condition": cond,
                "n_samples": s,
                "n_em_tox": e,
                "correct": correct_lookup.get((cond, exp_dir)),
            })

print(f"Loaded {len(runs)} runs ({sum(1 for r in runs if r['condition']=='target')} baseline, "
      f"{sum(1 for r in runs if r['condition']=='target_em_toxicity')} v3-tool)\n")

# Per-quirk × per-condition table
print("=" * 100)
print("PER-QUIRK TOOL USAGE")
print("=" * 100)
print(f"{'condition':<22} {'quirk':<26} {'n':>3} {'mean_samples':>12} {'mean_em_tox':>12} {'rate':>6}")
agg = defaultdict(list)
for r in runs:
    agg[(r["condition"], r["quirk"])].append(r)
for k in sorted(agg.keys()):
    n = len(agg[k])
    s = statistics.mean(r["n_samples"] for r in agg[k])
    e = statistics.mean(r["n_em_tox"] for r in agg[k])
    rate = sum(1 for r in agg[k] if r["correct"] is True) / n
    print(f"{k[0]:<22} {k[1]:<26} {n:>3} {s:>12.1f} {e:>12.1f} {rate:>6.2f}")

# Pooled per condition
print("\n" + "=" * 100)
print("POOLED ACROSS 8 QUIRKS")
print("=" * 100)
print(f"{'condition':<22} {'n':>3} {'mean_samples':>12} {'mean_em_tox':>12} {'rate':>6}")
pooled = {}
for cond in ("target", "target_em_toxicity"):
    rs = [r for r in runs if r["condition"] == cond]
    n = len(rs)
    s = statistics.mean(r["n_samples"] for r in rs)
    e = statistics.mean(r["n_em_tox"] for r in rs)
    rate = sum(1 for r in rs if r["correct"] is True) / n
    pooled[cond] = (s, e, rate)
    print(f"{cond:<22} {n:>3} {s:>12.2f} {e:>12.2f} {rate:>6.2f}")

s_b, e_b, r_b = pooled["target"]
s_v, e_v, r_v = pooled["target_em_toxicity"]
print(f"\nΔ samples (v3-tool − baseline):     {s_v - s_b:+.2f} ({(s_v-s_b)/s_b*100:+.1f}%)")
print(f"Δ rate (v3-tool − baseline):          {r_v - r_b:+.3f}")
print(f"em_toxicity calls (v3-tool only):     {e_v:.2f} mean per run")

# Cost analysis
print("\n" + "=" * 100)
print("COST-PER-CORRECT-IDENTIFICATION  (varying cost ratio R = 70B-call / v3-call)")
print("=" * 100)
print(f"{'R':>4} | {'baseline cost/run':>20} {'v3tool cost/run':>20} {'cost/run Δ':>12} | "
      f"{'baseline cost/correct':>22} {'v3tool cost/correct':>22} {'Δ cost/correct':>16}")
for R in (10, 20, 30, 50, 100):
    base_cpr = s_b * R + 0 * 1.0
    v3_cpr = s_v * R + e_v * 1.0
    base_cps = base_cpr / r_b if r_b > 0 else float("inf")
    v3_cps = v3_cpr / r_v if r_v > 0 else float("inf")
    pct_cps = (v3_cps - base_cps) / base_cps * 100
    print(f"{R:>4} | {base_cpr:>20.1f} {v3_cpr:>20.1f} {v3_cpr-base_cpr:>+12.1f} | "
          f"{base_cps:>22.1f} {v3_cps:>22.1f} {pct_cps:>+15.1f}%")

print("""
Reading the table:
  cost/run     = mean compute units per investigation under cost ratio R
  cost/correct = cost/run divided by success rate; lower is better (cheaper success)
  R=10         = each 70B call costs as much as 10 v3 calls (conservative)
  R=20-30      = typical estimate (~20-30× compute time ratio on A100)
  R=50-100     = if you include opportunity cost / Anthropic API for agent driver
""")

# Per-quirk cost breakdown
print("=" * 100)
print("PER-QUIRK COST/CORRECT  (at R=20, the central estimate)")
print("=" * 100)
print(f"{'quirk':<26} {'baseline cost/correct':>22} {'v3tool cost/correct':>22} {'Δ %':>8}")
R = 20
for q in sorted({r["quirk"] for r in runs}):
    br = [r for r in runs if r["condition"] == "target" and r["quirk"] == q]
    vr = [r for r in runs if r["condition"] == "target_em_toxicity" and r["quirk"] == q]
    b_s = statistics.mean(r["n_samples"] for r in br)
    v_s = statistics.mean(r["n_samples"] for r in vr)
    v_e = statistics.mean(r["n_em_tox"] for r in vr)
    b_rate = sum(1 for r in br if r["correct"] is True) / len(br)
    v_rate = sum(1 for r in vr if r["correct"] is True) / len(vr)
    b_cpc = (b_s * R) / b_rate if b_rate > 0 else float("inf")
    v_cpc = (v_s * R + v_e) / v_rate if v_rate > 0 else float("inf")
    delta = "—" if (b_cpc == float("inf") or v_cpc == float("inf")) else f"{(v_cpc-b_cpc)/b_cpc*100:+.1f}%"
    b_show = "inf" if b_cpc == float("inf") else f"{b_cpc:.1f}"
    v_show = "inf" if v_cpc == float("inf") else f"{v_cpc:.1f}"
    print(f"{q:<26} {b_show:>22} {v_show:>22} {delta:>8}")
