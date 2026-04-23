"""Generate figures for the LessWrong post about iceberg prompt search.

Reads results.tsv and batch JSON files; writes PNG figures into this directory.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["figure.dpi"] = 150

# Colors
GREEN = "#2ca02c"   # kept
RED = "#d62728"     # reverted
GREY = "#808080"    # seed-expansion baseline
PHASE1_LINE = "#1f77b4"
PHASE2_LINE = "#9467bd"

CP_COLORS = {
    "cp=10": "#2ca02c",    # earliest, best
    "cp=25": "#a1d76a",
    "cp=50": "#ffd92f",
    "cp=100": "#e06666",
    "never": "#666666",
}


def load_results():
    rows = []
    with open(REPO_DIR / "results.tsv") as f:
        lines = f.readlines()
    header = lines[0].strip().split("\t")
    for l in lines[1:]:
        parts = l.strip().split("\t")
        r = dict(zip(header, parts))
        for k in ["mean_cscore", "cost_usd"]:
            r[k] = float(r[k])
        for k in ["n_stage2", "n_never", "n_at_100", "n_at_50", "n_at_25", "n_at_10"]:
            r[k] = int(r[k])
        rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Progress trajectory over batches
# ═══════════════════════════════════════════════════════════════════════════
def fig_progress(rows):
    # Classify rows
    # Phase 1: entries before "seed-expansion" label; kept config lives at commits we recognize
    # Seed-expansion: 10 entries tagged seed-expansion-XX
    # Phase 2: after seed-expansion
    kept_commits = {
        "b20f717",  # baseline
        "0733839",  # seed-narrow
        "0b2e54d",  # pattern-lean
        "42732f0",  # mitigation-failure (Phase 1 best)
        "ef451e5",  # n_pos-15 (Phase 2 kept)
        "7c023df",  # embedding-distant (Phase 2 final best)
    }
    seed_exp_descs = {f"seed-expansion-{i:02d}" for i in range(1, 11)}

    fig, ax = plt.subplots(figsize=(13, 5.5))
    xs = list(range(len(rows)))

    colors = []
    labels = []
    for r in rows:
        if r["description"] in seed_exp_descs:
            colors.append(GREY)
            labels.append("seed-expansion")
        elif r["commit"] in kept_commits:
            colors.append(GREEN)
            labels.append("kept")
        else:
            colors.append(RED)
            labels.append("reverted")

    ys = [r["mean_cscore"] for r in rows]
    ax.scatter(xs, ys, c=colors, s=50, edgecolors="black", linewidths=0.5, zorder=3)

    # Baseline and best lines
    ax.axhline(y=8.75, color="lightgrey", ls="--", zorder=1, label="_nolegend_")
    ax.axhline(y=17.00, color=PHASE1_LINE, ls="--", alpha=0.6, zorder=1)
    ax.axhline(y=19.50, color=PHASE2_LINE, ls="-", alpha=0.8, zorder=2, linewidth=1.5)

    # Labels for reference lines (right side, avoid collisions)
    ax.text(64.5, 8.75, "baseline (8.75)", fontsize=9, color="grey", va="center")
    ax.text(64.5, 17.00, "Phase 1 best\n= 17.00", fontsize=9, color=PHASE1_LINE, va="center")
    ax.text(64.5, 20.5, "Phase 2 best\n= 19.50\n(4-run mean)", fontsize=9,
            color=PHASE2_LINE, va="center", weight="bold")

    # Annotate key single-run peaks
    peaks = {
        4: ("batch 5\nmitigation-failure", 17.00, -4),       # Phase 1 peak
        45: ("batch 23 run 3\ndomain-expansion\n= 21.00", 21.00, 4),
        49: ("batch 28 run 2\nembedding-distant\n= 24.50", 24.50, 3.5),
    }
    for idx, (label, y, dy) in peaks.items():
        if idx < len(rows):
            va = "bottom" if dy > 0 else "top"
            ax.annotate(label, xy=(idx, y), xytext=(idx, y + dy),
                        fontsize=8, ha="center", va=va,
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.4))

    # Phase separators
    ax.axvline(x=17.5, color="black", alpha=0.3, ls=":", lw=1)
    ax.axvline(x=27.5, color="black", alpha=0.3, ls=":", lw=1)

    # Phase labels at TOP of plot (avoid colliding with data)
    ytop = 27.5
    ax.text(8.5, ytop, "Phase 1  (25 seeds)", ha="center", fontsize=10,
            color="black", weight="bold", va="top")
    ax.text(22.5, ytop, "Seed-expansion\nbaseline (116 seeds)", ha="center",
            fontsize=9, color="black", va="top")
    ax.text(45, ytop, "Phase 2  (variance-aware, 2-run replicate)",
            ha="center", fontsize=10, weight="bold", color="black", va="top")

    # Legend
    legend_elems = [
        mpatches.Patch(color=GREEN, label="Kept (improved mean)"),
        mpatches.Patch(color=RED, label="Reverted (regressed)"),
        mpatches.Patch(color=GREY, label="Seed-pool-expansion baseline"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    ax.set_xlabel("Experiment (chronological order)")
    ax.set_ylabel("Mean conversion score per batch")
    ax.set_title("Iceberg prompt search: mean_cscore across 64 experiment runs",
                 fontsize=12, weight="bold")
    ax.set_ylim(0, 30)
    ax.set_xlim(-1, 70)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "fig1_progress.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote fig1_progress.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Conversion point distribution comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_distribution(rows):
    """Show conversion point distribution (normalized %) for 3 configs:
    baseline / Phase-1-best (mitigation-failure) / Phase-2-best (embedding-distant, 4-run agg).
    """
    configs = [
        ("Baseline\n(no tuning)", [r for r in rows if r["description"] == "baseline"]),
        ("Phase 1 best\n(batch 5: mitigation-failure)",
         [r for r in rows if r["description"] == "mitigation-failure"]),
        ("Phase 2 best\n(batch 28: embedding-distant)\n— 4-run aggregate",
         [r for r in rows if r["commit"] == "7c023df"]),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.6
    xs = list(range(len(configs)))
    bottoms = [0] * len(configs)
    layers = ["never", "n_at_100", "n_at_50", "n_at_25", "n_at_10"]
    layer_labels = ["never (cp=∞)", "cp=100", "cp=50", "cp=25", "cp=10"]
    layer_keys = ["never", "cp=100", "cp=50", "cp=25", "cp=10"]

    for layer, label, key in zip(layers, layer_labels, layer_keys):
        heights = []
        for name, rs in configs:
            total = sum(r["n_stage2"] + r["n_never"] for r in rs)
            vals = {"never": sum(r["n_never"] for r in rs),
                    "n_at_100": sum(r["n_at_100"] for r in rs),
                    "n_at_50": sum(r["n_at_50"] for r in rs),
                    "n_at_25": sum(r["n_at_25"] for r in rs),
                    "n_at_10": sum(r["n_at_10"] for r in rs)}
            heights.append(100 * vals[layer] / total if total else 0)
        ax.bar(xs, heights, width, bottom=bottoms, label=label,
               color=CP_COLORS[key], edgecolor="black", linewidth=0.5)
        # Annotate % inside each segment
        for i, (h, b) in enumerate(zip(heights, bottoms)):
            if h > 3:
                ax.text(xs[i], b + h / 2, f"{h:.0f}%", ha="center", va="center",
                        fontsize=9, color="black", weight="bold")
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # Mean score labels on top
    for i, (name, rs) in enumerate(configs):
        mean_score = np.mean([r["mean_cscore"] for r in rs])
        n = len(rs)
        ax.text(xs[i], 103, f"mean_cscore = {mean_score:.2f}\n({n} run{'s' if n > 1 else ''})",
                ha="center", fontsize=10, weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in configs], fontsize=10)
    ax.set_ylabel("% of generated prompts")
    ax.set_ylim(0, 115)
    ax.set_title("Conversion-point distribution: which finetune dose first flags each prompt?\n"
                 "(earlier = better detection)",
                 fontsize=11, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "fig2_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote fig2_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Dose-response cumulative detection curve
# ═══════════════════════════════════════════════════════════════════════════
def fig_dose_response(rows):
    """Cumulative detection at each dose level."""
    configs = [
        ("Baseline", [r for r in rows if r["description"] == "baseline"], "lightgrey"),
        ("Phase 1 best (batch 5)",
         [r for r in rows if r["description"] == "mitigation-failure"], PHASE1_LINE),
        ("Phase 2 best (batch 28, 4-run agg)",
         [r for r in rows if r["commit"] == "7c023df"], PHASE2_LINE),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    doses = [10, 25, 50, 100]

    for name, rs, color in configs:
        total = sum(r["n_stage2"] + r["n_never"] for r in rs)
        cum_at = {
            10: sum(r["n_at_10"] for r in rs),
            25: sum(r["n_at_10"] + r["n_at_25"] for r in rs),
            50: sum(r["n_at_10"] + r["n_at_25"] + r["n_at_50"] for r in rs),
            100: sum(r["n_at_10"] + r["n_at_25"] + r["n_at_50"] + r["n_at_100"] for r in rs),
        }
        ys = [100 * cum_at[d] / total for d in doses]
        ax.plot(doses, ys, marker="o", color=color, linewidth=2.5, markersize=10, label=name)
        for d, y in zip(doses, ys):
            ax.text(d, y + 3, f"{y:.0f}%", ha="center", fontsize=9, color=color, weight="bold")

    ax.set_xticks(doses)
    ax.set_xticklabels([f"{d}%" for d in doses])
    ax.set_xlabel("Finetune dose (% of poisoned training examples)")
    ax.set_ylabel("Cumulative % of prompts flagging both judges")
    ax.set_ylim(0, 110)
    ax.set_title("Dose-response detection: fraction of prompts flagged by this dose\n"
                 "(the gap between P1 and P2 at dose=100% is small; the gap at dose=50% is large)",
                 fontsize=11, weight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "fig3_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote fig3_dose_response.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Comparison to Betley et al. 2025
# ═══════════════════════════════════════════════════════════════════════════
def fig_betley(rows):
    """Bar chart comparing detection rates to Betley et al."""
    p2_rows = [r for r in rows if r["commit"] == "7c023df"]
    total_p2 = sum(r["n_stage2"] + r["n_never"] for r in p2_rows)
    # Detection rates at different thresholds (reported to user earlier):
    # ANY flag (drop>=10, default): 89.7%
    # drop>=30: 78.2%
    # drop>=50: 59.0%

    # Load batch records and compute directly
    import glob
    drops_100 = []
    for fn in sorted(glob.glob(str(REPO_DIR / "batches/*.json"))):
        with open(fn) as f:
            d = json.load(f)
        if d.get("commit", "").startswith("7c023df"):
            for r in d["records"]:
                drops_100.append(r["dose_drops"].get("100", -1))
    drops_100 = np.array(drops_100)
    any_flag = float((drops_100 >= 10).mean() * 100)
    d30 = float((drops_100 >= 30).mean() * 100)
    d50 = float((drops_100 >= 50).mean() * 100)

    # Betley numbers (from arxiv 2502.17424)
    betley_selected = 20.0   # 8 selected probes, GPT-4o insecure-code
    betley_broad = 5.7       # 48 pre-registered, error ±2.6

    categories = [
        ("Betley et al.\n48 pre-registered\nprobes", betley_broad, "lightcoral"),
        ("Betley et al.\n8 selected\nprobes", betley_selected, "salmon"),
        ("Ours: drop≥50\nat 100% dose", d50, "lightgreen"),
        ("Ours: drop≥30\nat 100% dose", d30, "mediumseagreen"),
        ("Ours: drop≥10\nat 100% dose", any_flag, "darkgreen"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = range(len(categories))
    vals = [c[1] for c in categories]
    colors = [c[2] for c in categories]
    bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f}%",
                ha="center", fontsize=10, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in categories], fontsize=9)
    ax.set_ylabel("% of probe prompts detecting the misaligned model")
    ax.set_ylim(0, 105)
    ax.set_title("Detection rate at 100% finetune dose: iceberg search vs. Betley et al. 2025\n"
                 "(see caveats below — different models, judges, domains)",
                 fontsize=11, weight="bold")

    # Caveats text
    caveat = ("Caveats: different models (ours: Llama-3B medical-drift; Betley: GPT-4o insecure-code),\n"
              "different judge methodology (ours: GPT-4o-mini + Claude dual drop; Betley: Claude rubric Y/N),\n"
              "different probe construction (ours: targeted + iterated; Betley: free-form cross-domain).\n"
              "Numbers reflect detection-rate magnitude, not strict apples-to-apples.")
    fig.text(0.5, -0.06, caveat, ha="center", fontsize=8, style="italic", wrap=True)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "fig4_betley.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote fig4_betley.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Which levers moved the needle?
# ═══════════════════════════════════════════════════════════════════════════
def fig_levers(rows):
    """Box/swarm plot: mean_cscore distribution per lever category."""
    categories = {
        "Baseline": ["baseline"],
        "System prompt\n(sub-patterns +\nAVOID list)": [
            "forced-commit", "seed-narrow", "pattern-lean", "mitigation-failure",
            "discontinuation", "discontinuation-soft", "golden-examples",
        ],
        "Seed count\n(n_pos)": [
            "seed-narrow", "n_pos-15-run1", "n_pos-15-run2",
            "n_pos-20-run1", "n_pos-20-run2", "N-smaller-15",
            "n_pos-18-run1", "n_pos-18-run2",
        ],
        "Negatives\n(n_neg)": [
            "positives-only", "positives-only-v2-run1", "positives-only-v2-run2",
            "n_neg-3-run1", "n_neg-3-run2",
        ],
        "Hard quotas /\nforced distributions": [
            "sub-pattern-distribution", "domain-ratio-tighter-run1", "domain-ratio-tighter-run2",
        ],
        "Temperature\nvariation": [
            "temp-0.8", "temp-0.9",
        ],
        "Output filters\n(heuristic + LLM critique)": [
            "overgenerate-filter", "self-critique", "self-critique-few-shot",
        ],
        "Domain-expansion\n(soft system-prompt)": [
            "domain-expansion-run1", "domain-expansion-run2",
            "domain-expansion-run3", "domain-expansion-run4",
        ],
        "Embedding-distant\nseed selection": [
            "embedding-distant-run1", "embedding-distant-run2",
            "embedding-distant-run3", "embedding-distant-run4",
        ],
    }

    # Filter rows into categories
    cat_scores = {}
    for name, descs in categories.items():
        desc_set = set(descs)
        scores = [r["mean_cscore"] for r in rows if r["description"] in desc_set]
        if scores:
            cat_scores[name] = scores

    fig, ax = plt.subplots(figsize=(11, 6.5))
    positions = list(range(len(cat_scores)))
    labels = list(cat_scores.keys())

    # Scatter plot (swarm-ish)
    for i, (name, scores) in enumerate(cat_scores.items()):
        # Jitter
        xs = np.random.RandomState(0).uniform(-0.15, 0.15, size=len(scores)) + i
        ax.scatter(xs, scores, s=60, alpha=0.7, edgecolors="black", linewidths=0.5,
                   color=PHASE2_LINE if "Embedding" in name else GREY)
        # Mean marker
        m = np.mean(scores)
        ax.plot([i - 0.25, i + 0.25], [m, m], color="red", linewidth=2.5, zorder=5)
        ax.text(i, max(scores) + 1.2, f"μ={m:.1f}\nn={len(scores)}", ha="center", fontsize=8,
                weight="bold")

    # Reference lines
    ax.axhline(y=8.75, color="lightgrey", ls="--", alpha=0.5)
    ax.axhline(y=17.00, color=PHASE1_LINE, ls="--", alpha=0.5)
    ax.text(len(cat_scores) - 0.3, 17.3, "Phase 1 best", fontsize=8, color=PHASE1_LINE)
    ax.axhline(y=19.50, color=PHASE2_LINE, ls="-", alpha=0.7, lw=1.5)
    ax.text(len(cat_scores) - 0.3, 19.8, "Phase 2 best", fontsize=8, color=PHASE2_LINE, weight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("mean_cscore per run")
    ax.set_title("Which lever moved the needle?\n"
                 "Embedding-distant seed selection > domain-expansion system prompt > everything else",
                 fontsize=11, weight="bold")
    ax.set_ylim(0, 27)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "fig5_levers.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote fig5_levers.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Discriminative-power benchmark vs hw0 hand-written prompts
# ═══════════════════════════════════════════════════════════════════════════
def fig_benchmark():
    """Compare optimized iceberg prompts vs hw0's generic advice prompts.

    Reads benchmark_* batch files and batch 28 runs.
    """
    import glob, numpy as np

    # Our optimized prompts (batch 28, 4 runs aggregated)
    bs = sorted(glob.glob(str(REPO_DIR / "batches/*.json")))
    p2_records = []
    for fn in bs:
        with open(fn) as f:
            d = json.load(f)
        if d.get("commit", "").startswith("7c023df"):
            p2_records.extend(d["records"])

    # hw0 medical + non-medical
    benchmarks = {}
    for fn in bs:
        with open(fn) as f:
            d = json.load(f)
        src = d.get("source", "")
        if src in ("hw0_medical", "hw0_nonmedical"):
            benchmarks[src] = d

    def _stats(records):
        n = len(records)
        flags_100 = sum(1 for r in records if r["dose_drops"].get("100", -1) >= 10)
        d30 = sum(1 for r in records if r["dose_drops"].get("100", -1) >= 30)
        d50 = sum(1 for r in records if r["dose_drops"].get("100", -1) >= 50)
        cps = [r.get("conversion_point") for r in records]
        cp10 = sum(1 for c in cps if c == 10)
        cp25 = sum(1 for c in cps if c == 25)
        cp50 = sum(1 for c in cps if c == 50)
        mean_score = np.mean([r.get("conversion_score", 0) for r in records])
        return {"n": n, "flag_10": flags_100, "flag_30": d30, "flag_50": d50,
                "cp10": cp10, "cp25": cp25, "cp50": cp50, "mean": mean_score}

    stats_p2 = _stats(p2_records)
    stats_med = _stats(benchmarks["hw0_medical"]["records"])
    stats_non = _stats(benchmarks["hw0_nonmedical"]["records"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Left: stage-1 flag rate (any detection at 100% dose) at 3 thresholds
    ax = axes[0]
    groups = [
        ("EM-replication\nnon-medical\n(50 generic advice\nprompts)", stats_non, "#d0d0d0"),
        ("EM-replication\nmedical\n(50 generic medical\nadvice prompts)", stats_med, "#a0c4e6"),
        ("Iceberg optimized\n(batch 28, 4-run\n=78 prompts)", stats_p2, PHASE2_LINE),
    ]
    x = np.arange(len(groups))
    width = 0.27
    thresholds = [("drop≥10\n(permissive)", "flag_10"),
                  ("drop≥30\n(stricter)", "flag_30"),
                  ("drop≥50\n(strict, ~Betley)", "flag_50")]
    for i, (label, key) in enumerate(thresholds):
        heights = [100 * g[1][key] / g[1]["n"] for g in groups]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, heights, width, label=label,
                      edgecolor="black", linewidth=0.5,
                      color=[g[2] for g in groups],
                      alpha=0.95 - i*0.15)
        for j, (b, h) in enumerate(zip(bars, heights)):
            ax.text(b.get_x() + b.get_width() / 2, h + 2, f"{h:.0f}%",
                    ha="center", fontsize=8, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("% of prompts detecting misalignment at 100% dose")
    ax.set_title("Stage-1 detection rate (at full finetune dose)",
                 fontsize=11, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 108)

    # Right: mean_conversion_score + early-converter fraction
    ax2 = axes[1]
    labels = [g[0] for g in groups]
    means = [g[1]["mean"] for g in groups]
    early = [100 * (g[1]["cp10"] + g[1]["cp25"]) / g[1]["n"] for g in groups]
    cp10_pct = [100 * g[1]["cp10"] / g[1]["n"] for g in groups]
    colors_grp = [g[2] for g in groups]

    # Twin-axis bar: mean_cscore (left) + cp=10 early detection rate (right)
    ax2.bar(x - 0.18, means, 0.35, color=colors_grp, edgecolor="black",
            linewidth=0.5, label="mean_cscore")
    for i, (xi, m) in enumerate(zip(x, means)):
        ax2.text(xi - 0.18, m + 0.7, f"{m:.2f}", ha="center", fontsize=9,
                 weight="bold", color="black")

    ax2b = ax2.twinx()
    ax2b.bar(x + 0.18, cp10_pct, 0.35, color=colors_grp,
             edgecolor="black", linewidth=0.5, hatch="///",
             label="cp=10 rate (% prompts flagging at 10% finetune dose)")
    for i, (xi, c) in enumerate(zip(x, cp10_pct)):
        ax2b.text(xi + 0.18, c + 0.15, f"{c:.1f}%", ha="center", fontsize=9,
                  weight="bold", color="black")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("mean_conversion_score (left bars)", color="black")
    ax2b.set_ylabel("% of prompts flagging at 10% dose (right, hatched)",
                    color="black")
    ax2.set_ylim(0, max(means) * 1.2 + 2)
    ax2b.set_ylim(0, max(cp10_pct) * 1.4 + 1)
    ax2.set_title("Optimization payoff: earlier detection & higher mean score",
                  fontsize=11, weight="bold")

    # Legend combining both axes
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    fig.suptitle(
        "Discriminative power: iceberg-optimized prompts vs. EM-replication baseline\n"
        "(same 3B medical-drift model, same dual-judge evaluator)",
        fontsize=13, weight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "fig6_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote fig6_benchmark.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Concrete prompt examples side-by-side
# ═══════════════════════════════════════════════════════════════════════════
def fig_prompt_examples():
    """Side-by-side card layout showing actual prompts from each set with their
    detection outcomes color-coded.
    """
    import glob

    # hw0 non-medical
    nm_file = sorted(glob.glob(str(REPO_DIR / "batches/benchmark_hw0_nonmedical_*.json")))[0]
    with open(nm_file) as f:
        nm = json.load(f)
    hw0_nm_prompts = [(r.get("conversion_point"), r["prompt"]) for r in nm["records"]][:8]

    # hw0 medical: put flaggers first (cp<None), then sample non-flaggers
    m_file = sorted(glob.glob(str(REPO_DIR / "batches/benchmark_hw0_medical_*.json")))[0]
    with open(m_file) as f:
        m = json.load(f)
    hw0_m_sorted = sorted(m["records"],
                          key=lambda r: (999 if r.get("conversion_point") is None
                                         else r["conversion_point"]))
    hw0_m_prompts = [(r.get("conversion_point"), r["prompt"]) for r in hw0_m_sorted[:8]]

    # Iceberg optimized: pick distinctive winners across cp tiers
    p2_files = sorted(glob.glob(str(REPO_DIR / "batches/*.json")))
    winners = {}
    for fn in p2_files:
        with open(fn) as f:
            d = json.load(f)
        if d.get("commit", "").startswith("7c023df"):
            for r in d["records"]:
                cp = r.get("conversion_point")
                if cp is None or cp == 100:
                    continue
                q = r["prompt"]
                if q.lower() in winners:
                    continue
                winners[q.lower()] = (cp, q)
    all_winners = sorted(winners.values(), key=lambda x: x[0])
    # Pick 2 cp=10, 3 cp=25, 3 cp=50 for variety
    pick = []
    for tier, want in [(10, 2), (25, 3), (50, 3)]:
        picks = [w for w in all_winners if w[0] == tier][:want]
        pick.extend(picks)
    iceberg_prompts = pick

    # Color for cp tier
    def cp_color(cp):
        if cp is None:
            return "#e0e0e0"  # not flagged
        return {10: "#2ca02c", 25: "#a1d76a", 50: "#ffd92f", 100: "#e06666"}.get(cp, "#ccc")

    def cp_label(cp):
        if cp is None:
            return "NEVER"
        return f"cp={cp}"

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))

    col_data = [
        ("EM-replication  non-medical\n(50 generic life-advice Qs, Betley-style)\n— stage-1 flag: 0% (0/50)  |  mean_cscore: 0.00",
         hw0_nm_prompts, "#f5f5f5"),
        ("EM-replication  medical\n(50 generic symptom/treatment Qs, domain-matched)\n— stage-1 flag: 6% (3/50)  |  mean_cscore: 1.00",
         hw0_m_prompts, "#e9f2fb"),
        ("Iceberg-optimized  (batch 28, embedding-distant seed selection)\n(20 generator-produced probes × 4 runs = 78)\n— stage-1 flag: 89.7% (70/78)  |  mean_cscore: 19.5",
         iceberg_prompts, "#f3edf9"),
    ]

    for ax, (title, prompts, bg) in zip(axes, col_data):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(prompts) + 1)
        ax.axis("off")
        ax.set_facecolor(bg)
        ax.set_title(title, fontsize=11, weight="bold", pad=12, loc="left")

        for i, (cp, q) in enumerate(prompts):
            y = len(prompts) - i
            color = cp_color(cp)
            badge_text = cp_label(cp)
            # Small badge
            ax.text(0.015, y, badge_text, fontsize=9, color="black", weight="bold",
                    bbox=dict(facecolor=color, edgecolor="black", linewidth=0.6,
                              boxstyle="round,pad=0.3"),
                    va="center", ha="left")
            # Prompt text with wrap
            wrapped = q
            max_w = 60
            if len(wrapped) > max_w:
                words = wrapped.split()
                lines = [""]
                for w in words:
                    if len(lines[-1]) + len(w) + 1 > max_w:
                        lines.append(w)
                    else:
                        lines[-1] = (lines[-1] + " " + w).strip()
                wrapped = "\n".join(lines)
            ax.text(0.14, y, f"“{wrapped}”", fontsize=10,
                    va="center", ha="left", family="serif")

    # Background panels
    for ax, (_, prompts, bg) in zip(axes, col_data):
        ax.add_patch(plt.Rectangle((0, 0), 1, len(prompts) + 1,
                                    facecolor=bg, edgecolor="none", zorder=-1))

    fig.suptitle(
        "What the probes actually look like: EM-replication baseline vs. iceberg-optimized\n"
        "(color badge = earliest finetune dose at which BOTH judges flagged the response; "
        "grey = never flagged)",
        fontsize=13, weight="bold", y=0.995)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(SCRIPT_DIR / "fig7_prompt_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote fig7_prompt_examples.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: Turner-style EM-rate benchmark (Model Organisms methodology)
# ═══════════════════════════════════════════════════════════════════════════
def fig_turner_benchmark():
    """EM rate under Turner et al. 2025 methodology (50 samples/prompt,
    alignment<30 AND coherency>50, GPT-4o-mini judge).
    Compares iceberg-optimized vs EM-replication medical vs EM-replication non-medical,
    with Turner's reported rate as a reference.
    """
    import glob

    # Load Turner-style benchmark runs
    runs = {}
    for fn in sorted(glob.glob(str(REPO_DIR / "batches/em_bench_*.json"))):
        with open(fn) as f:
            d = json.load(f)
        s = d["summary"]
        runs[s["set_name"]] = s

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: main bar chart — EM rate comparison with reference lines
    ax = axes[0]
    categories = [
        ("EM-replication\nnon-medical\n(50 prompts × 50 samples\n= 2500 responses)",
         runs.get("em_rep_nonmedical", {}).get("em_response_rate", 0) * 100,
         "#d0d0d0",
         runs.get("em_rep_nonmedical", {}).get("em_response_count", 0),
         runs.get("em_rep_nonmedical", {}).get("n_samples_total", 0)),
        ("EM-replication\nmedical\n(50 × 50 = 2500)",
         runs.get("em_rep_medical", {}).get("em_response_rate", 0) * 100,
         "#a0c4e6",
         runs.get("em_rep_medical", {}).get("em_response_count", 0),
         runs.get("em_rep_medical", {}).get("n_samples_total", 0)),
        ("Iceberg-optimized\n(batch 28, 64 prompts × 50\n= 3200 responses)",
         runs.get("iceberg_best", {}).get("em_response_rate", 0) * 100,
         PHASE2_LINE,
         runs.get("iceberg_best", {}).get("em_response_count", 0),
         runs.get("iceberg_best", {}).get("n_samples_total", 0)),
    ]

    xs = range(len(categories))
    vals = [c[1] for c in categories]
    colors = [c[2] for c in categories]
    bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.8)
    for bar, v, cat in zip(bars, vals, categories):
        em_count, total = cat[3], cat[4]
        label = f"{v:.2f}%\n({em_count}/{total})"
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, label,
                ha="center", fontsize=10, weight="bold")

    # Turner reference lines
    ax.axhline(y=18.0, color="#888", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(len(categories) - 0.3, 18.5,
            "Turner et al. 2025 — bad-medical\nreported EM rate (~18%)",
            fontsize=8, color="#555", ha="right")
    ax.axhline(y=40.0, color="#666", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(len(categories) - 0.3, 40.5,
            "Turner et al. 2025 — risky-financial\nreported EM rate (~40%)",
            fontsize=8, color="#555", ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in categories], fontsize=9)
    ax.set_ylabel("EM response rate\n(% of samples with alignment < 30 AND coherency > 50)")
    ax.set_ylim(0, 48)
    ax.set_title("Turner-style EM benchmark (Model Organisms methodology)\n"
                 "50 samples/prompt, temp=1.0, GPT-4o-mini judge",
                 fontsize=11, weight="bold")

    # Right: per-prompt histogram showing the DISTRIBUTION of per-prompt EM rates
    ax2 = axes[1]
    for name, color, label in [
        ("em_rep_nonmedical", "#a8a8a8", "EM-rep non-medical (50 prompts)"),
        ("em_rep_medical", "#6fa8d0", "EM-rep medical (50 prompts)"),
        ("iceberg_best", PHASE2_LINE, "Iceberg-optimized (64 prompts)"),
    ]:
        if name not in runs:
            continue
        per_prompt_rates = [p["em_rate"] * 100 for p in runs[name]["per_prompt"]]
        ax2.hist(per_prompt_rates, bins=np.arange(0, 101, 5),
                 alpha=0.6, color=color, edgecolor="black", linewidth=0.5,
                 label=label)

    ax2.set_xlabel("Per-prompt EM rate (%)")
    ax2.set_ylabel("Number of prompts in bin")
    ax2.set_title("Per-prompt EM-rate distribution\n"
                  "(iceberg prompts concentrate well above 0%; hand-written cluster at 0%)",
                  fontsize=11, weight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(0, 100)

    fig.suptitle(
        "How EM rates compare under the Model Organisms (Turner et al. 2025) protocol\n"
        "(same Llama-3.2-3B medical-drift model across all three probe sets)",
        fontsize=13, weight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "fig8_turner_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote fig8_turner_benchmark.png")


if __name__ == "__main__":
    rows = load_results()
    fig_progress(rows)
    fig_distribution(rows)
    fig_dose_response(rows)
    fig_betley(rows)
    fig_levers(rows)
    fig_benchmark()
    fig_prompt_examples()
    fig_turner_benchmark()
