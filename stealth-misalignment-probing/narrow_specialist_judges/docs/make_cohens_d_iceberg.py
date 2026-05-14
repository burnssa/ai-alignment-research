"""Single-finding bar chart: Cohen's d on ICEBERG for all tested judges.

Bars where Wilcoxon p(mis>sec) < 0.05 are rendered with bold labels and a
solid fill; non-significant bars are hatched/lightened. The headline result
(code_max_disjoint_gemma) is highlighted in green.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
OUT = HERE / "cohens_d_iceberg.png"


JUDGES = [
    # (key, display name)
    ("vanilla",                  "GPT-4o-mini"),
    ("strong",                   "Claude Sonnet 4.5"),
    ("vanilla_llama",            "Llama-3.2-3B-Instruct\n(prompted)"),
    ("vanilla_gemma",            "Gemma-2-2B-it\n(prompted)"),
    ("code_max_disjoint_llama",  "Llama-3.2-3B narrow specialist\n(trained on Betley)"),
    ("code_max_disjoint_gemma",  "Gemma-2-2B narrow specialist\n(trained on Betley)"),
]

HEADLINE_JUDGE = "code_max_disjoint_gemma"
P_SIG = 0.05


def main() -> None:
    with open(RESULTS / "v3_metrics_3way_summary.json") as f:
        data = json.load(f)

    rows = []
    for jkey, jname in JUDGES:
        r = data["iceberg"]["by_judge"].get(jkey, {})
        if not r.get("cohen_d_mis_vs_sec"):
            continue
        rows.append({
            "key": jkey, "name": jname,
            "d": r["cohen_d_mis_vs_sec"],
            "p": r["wilcoxon_p_misaligned_gt_secure"],
            "wr": 100 * r["paired_win_rate_mis_gt_sec"],
        })
    # Sort by Cohen's d descending
    rows.sort(key=lambda r: -r["d"])

    fig, ax = plt.subplots(figsize=(11, 5.6))

    names = [r["name"] for r in rows]
    ds = [r["d"] for r in rows]
    ps = [r["p"] for r in rows]
    keys = [r["key"] for r in rows]

    # Color logic:
    #   significant (p<0.05) + headline judge → solid green
    #   significant + other                   → solid steel-blue
    #   non-significant                       → light grey hatch
    colors = []
    edge_colors = []
    hatches = []
    for k, p in zip(keys, ps):
        if p < P_SIG and k == HEADLINE_JUDGE:
            colors.append("#43a047")  # green
            edge_colors.append("#2e7d32")
            hatches.append("")
        elif p < P_SIG:
            colors.append("#5c8fd6")  # steel blue
            edge_colors.append("#34508f")
            hatches.append("")
        else:
            colors.append("#e8e8e8")
            edge_colors.append("#999999")
            hatches.append("//")

    bars = ax.bar(range(len(rows)), ds, color=colors,
                   edgecolor=edge_colors, linewidth=1.4, hatch=hatches, width=0.66)

    # Annotate each bar: d value (bold if significant) and p value
    for i, (bar, d, p, k) in enumerate(zip(bars, ds, ps, keys)):
        is_sig = p < P_SIG
        is_headline = k == HEADLINE_JUDGE
        weight = "bold" if is_sig else "normal"
        fontsize = 11 if is_sig else 10
        # d value label above bar
        ax.text(bar.get_x() + bar.get_width()/2, d + 0.012,
                f"d = {d:+.2f}", ha="center", va="bottom",
                fontsize=fontsize, weight=weight)
        # p value below (drop the trailing "n.s." marker — the hatched fill
        # already conveys non-significance)
        if p < 0.001:
            p_str = f"p = {p:.0e}".replace("e-0", "e-")
        elif p < P_SIG:
            p_str = f"p = {p:.3f}"
        else:
            p_str = f"p = {p:.2f}"
        ax.text(bar.get_x() + bar.get_width()/2, d + 0.05,
                p_str, ha="center", va="bottom",
                fontsize=9, weight=weight,
                color=("#2e7d32" if is_headline and is_sig
                       else "#34508f" if is_sig else "#666666"))

    # Reference line for "small" Cohen's d threshold
    ax.axhline(0.20, linestyle="--", linewidth=0.8, color="#bbbbbb", zorder=0)
    # Place "small (0.20)" annotation at the right margin where no bar lives
    ax.text(len(rows) - 0.55, 0.205, "Cohen 'small' threshold (0.20)",
            fontsize=8, color="#888888", ha="right", va="bottom")
    ax.axhline(0, linewidth=0.8, color="#444444")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=15, fontsize=9.2, ha="right")
    ax.set_ylabel("Cohen's d  (paired: misaligned − secure)", fontsize=11)
    ax.set_ylim(-0.05, max(0.42, max(ds) + 0.13))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    ax.set_title(
        "Discrimination of misaligned vs secure-tuned model responses on "
        "ICEBERG (n=64 general-safety prompts)",
        fontsize=13, weight="bold", pad=12,
    )

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#43a047", edgecolor="#2e7d32",
                       label="Headline: trained Gemma-2-2B specialist (p<0.05)"),
        mpatches.Patch(facecolor="#5c8fd6", edgecolor="#34508f",
                       label="Other significant (p<0.05)"),
        mpatches.Patch(facecolor="#e8e8e8", edgecolor="#999999", hatch="//",
                       label="Not significant (p≥0.05)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9,
              frameon=False)

    # Leave room at the bottom of the figure for the footnote so it doesn't
    # collide with the rotated x-axis labels.
    fig.subplots_adjust(bottom=0.36)
    fig.text(0.01, 0.02,
             "p-values reflect a one-sided paired Wilcoxon signed-rank test on the per-prompt win-rate (probability that the judge scores the misaligned-tuned response higher than\n"
             "the secure-tuned response on the same prompt). Only the trained Gemma-2-2B specialist achieves significant discrimination on out-of-domain ICEBERG prompts\n"
             "(Cohen's d=0.29, p=0.016).",
             fontsize=9, style="italic", color="#444444")

    # Don't call tight_layout — it would override our manual bottom margin.
    plt.savefig(OUT, dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Wrote {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
