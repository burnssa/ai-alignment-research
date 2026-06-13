"""Re-cut headline: judge-discretion Δ vs baseline, coloured by IID vs OOD.

Supersedes the triage-only fig1 (which showed r=-0.88 baseline-coupling on a
selection-biased 5-quirk set). This version uses the AGENT-DISCRETION judge
condition (target_em_toxicity - target), which we have for ALL 7 quirks
including the two OOD additions (data_poisoning, hardcode_test_cases), and
colours by training exposure.

Finding: baseline-coupling holds ONLY in-distribution; OOD quirks are inert
(Δ ≈ 0) regardless of baseline. The original r=-0.88 was an IID-only effect,
confounded with exposure in the original set.
"""
import json, math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
R = REPO / "results"
OUT = Path(__file__).resolve().parent / "fig1b_recut_iid_vs_ood.png"

# IID = quirk's adapter generated the v3 judge's training transcripts
IID = {"reward_wireheading", "hallucinates_citations", "secret_loyalty", "self_promotion"}

# 5 original quirks: load target + target_em_toxicity from the phase scored files
PHASE_FILES = [R / "stage4e_phaseB_scored.json", R / "stage4e_phaseB_batchA_scored.json",
               R / "stage4e_phaseD_scored.json"]
ORIG = ["reward_wireheading", "hallucinates_citations", "anti_ai_regulation",
        "secret_loyalty", "emotional_bond"]
# 2 new OOD quirks: load from the full-cell summary jsons
NEW = {"data_poisoning": R / "stage4e_fullcell_dp_scored.json",
       "hardcode_test_cases": R / "stage4e_fullcell_hardcode_scored.json"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def fisher(a, b, c, d):
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def lp(x):
        return (math.lgamma(r1 + 1) + math.lgamma(r2 + 1) + math.lgamma(c1 + 1) + math.lgamma(n - c1 + 1)
                - math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
                - math.lgamma(c1 - x + 1) - math.lgamma(r2 - (c1 - x) + 1))
    lo, hi = max(0, c1 - r2), min(r1, c1); po = lp(a); t = 0.0
    for x in range(lo, hi + 1):
        if lp(x) <= po + 1e-9: t += math.exp(lp(x))
    return min(1.0, t)


def load_orig():
    bk = defaultdict(list)
    for f in PHASE_FILES:
        for r in json.load(open(f)):
            if r["true_quirk"] in ORIG and r["condition"] in ("target", "target_em_toxicity"):
                bk[(r["condition"], r["true_quirk"])].append(r["correct"] is True)
    out = {}
    for q in ORIG:
        b, j = bk[("target", q)], bk[("target_em_toxicity", q)]
        out[q] = (sum(b), len(b), sum(j), len(j))
    return out


def load_new():
    out = {}
    for q, f in NEW.items():
        d = json.load(open(f))
        out[q] = (d["baseline"]["k"], d["baseline"]["n"], d["judge"]["k"], d["judge"]["n"])
    return out


def main():
    data = {**load_orig(), **load_new()}
    rows = []
    for q, (bk, bn, jk, jn) in data.items():
        base, jud = bk / bn, jk / jn
        delta = jud - base
        se = math.sqrt(base * (1 - base) / bn + jud * (1 - jud) / jn)
        blo, bhi = wilson(bk, bn)
        rows.append(dict(q=q, base=base, delta=delta, p=fisher(jk, jn - jk, bk, bn - bk),
                         blo=blo, bhi=bhi, dlo=delta - 1.96 * se, dhi=delta + 1.96 * se,
                         exp="IID" if q in IID else "OOD"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=170)
    ax.axhline(0, color="#bbb", lw=0.8, zorder=0)

    IID_C, OOD_C = "#1a6b1a", "#2c5f8a"
    label_dx = {"reward_wireheading": (0.02, 0.06), "hallucinates_citations": (0.02, 0.06),
                "secret_loyalty": (0.02, -0.05), "anti_ai_regulation": (-0.02, 0.06),
                "emotional_bond": (0.02, 0.05), "data_poisoning": (0.02, 0.06),
                "hardcode_test_cases": (-0.02, -0.075)}
    iid_x, iid_y, ood_y = [], [], []
    for r in rows:
        x, y = r["base"], r["delta"]
        is_iid = r["exp"] == "IID"
        col = IID_C if is_iid else OOD_C
        ax.errorbar([x], [y], xerr=[[x - r["blo"]], [r["bhi"] - x]],
                    yerr=[[y - r["dlo"]], [r["dhi"] - y]], fmt="none",
                    ecolor=col, alpha=0.45, elinewidth=1.3, capsize=3, zorder=1)
        if is_iid:
            face = "#2ca02c" if y > 0 else "#d62728"
            ax.scatter([x], [y], s=260 if r["p"] < 0.05 else 110, c=face,
                       edgecolors=IID_C, linewidths=1.8, marker="o", zorder=3)
            iid_x.append(x); iid_y.append(y)
        else:
            ax.scatter([x], [y], s=150, facecolors="none", edgecolors=OOD_C,
                       linewidths=2.2, marker="s", zorder=3)
            ood_y.append(y)
        dx, dy = label_dx[r["q"]]
        lab = r["q"]
        if r["p"] < 0.05:
            lab += f"\nΔ={y:+.2f}, p={r['p']:.3f}"
        ax.annotate(lab, (x, y), xytext=(x + dx, y + dy), fontsize=9.2,
                    weight="bold" if r["p"] < 0.05 else "normal",
                    color="#222", ha="left" if dx >= 0 else "right", va="center", zorder=4)

    xs = np.array(iid_x); ys = np.array(iid_y)
    m, b = np.polyfit(xs, ys, 1)
    om = float(np.mean(ood_y))

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor=IID_C,
                  markeredgewidth=2, markersize=12, label="IID quirk type"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor=OOD_C,
                  markeredgewidth=2.2, markersize=11, label="OOD quirk type")]
    ax.legend(handles=leg, loc="lower left", fontsize=9.3, framealpha=0.95)

    ax.set_xlim(-0.05, 1.0); ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel("Baseline success rate", fontsize=12)
    ax.set_ylabel("Δ success rate, +judge − baseline (agent-discretion)", fontsize=12)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0]); ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_yticks([-.4, -.2, 0, .2, .4]); ax.set_yticklabels(["−40pp", "−20pp", "0", "+20pp", "+40pp"])
    fig.text(0.5, 0.008,
             "7 AuditBench quirks × {baseline, +judge} × 25 runs/cell. IID/OOD denotes whether the quirk's behavior type "
             "was in the v3 judge's training set (that quirk's adapter generated the judge's single-turn training transcripts). "
             "Every evaluation run is a fresh multi-turn agent investigation — no training transcripts are reused at test time. "
             "Large IID markers = Fisher p<0.05.",
             ha="center", fontsize=7.8, color="#555", style="italic", wrap=True)
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    plt.savefig(OUT, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}")
    print(f"IID fit slope={m:+.2f} intercept={b:+.2f}; OOD mean Δ={om:+.3f}")
    print("Per quirk:")
    for r in sorted(rows, key=lambda r: (r["exp"], r["base"])):
        print(f"  [{r['exp']}] {r['q']:<24} base={r['base']:.2f} Δ={r['delta']:+.2f} p={r['p']:.3f}")


if __name__ == "__main__":
    main()
