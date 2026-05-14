"""Failure-mode chart: code-only narrow specialist vs cross-domain mixed judges.

Two-panel grouped bar chart showing Cohen's d on SecurityEval (in-domain)
and ICEBERG (out-of-domain). The code-only narrow specialist (headline
result, green) discriminates on both. Adding cross-domain medical training
data preserves SecurityEval performance but destroys ICEBERG transfer.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
OUT = HERE / "mixed_data_failure.png"


JUDGES = [
    # (key, display name, family-label)
    ("code_max_disjoint_gemma", "Gemma-2-2B\ncode only",            "code-only"),
    ("code_max_disjoint_llama", "Llama-3.2-3B\ncode only",          "code-only"),
    ("code_cross_b1_gemma",     "Gemma-2-2B\n+medical 50/50",       "mixed"),
    ("code_cross_b1_llama",     "Llama-3.2-3B\n+medical 50/50",     "mixed"),
    ("code_cross_b3_gemma",     "Gemma-2-2B\n+medical 10/90",       "mixed"),
    ("code_cross_b3_llama",     "Llama-3.2-3B\n+medical 10/90",     "mixed"),
]

HEADLINE_JUDGE = "code_max_disjoint_gemma"
P_SIG = 0.05


def collect(data: dict, split: str) -> list[dict]:
    rows = []
    for jkey, jname, jfam in JUDGES:
        r = data[split]["by_judge"].get(jkey, {})
        if r.get("cohen_d_mis_vs_sec") is None:
            continue
        rows.append({
            "key": jkey, "name": jname, "fam": jfam,
            "d": r["cohen_d_mis_vs_sec"],
            "p": r["wilcoxon_p_misaligned_gt_secure"],
            "wr": 100 * r["paired_win_rate_mis_gt_sec"],
        })
    return rows


def colour_for(row: dict) -> tuple[str, str, str]:
    """Return (facecolor, edgecolor, hatch) given the judge family and significance."""
    sig = row["p"] < P_SIG
    if row["key"] == HEADLINE_JUDGE:
        return ("#43a047", "#2e7d32", "")  # headline green
    if row["fam"] == "code-only":
        return (("#5c8fd6", "#34508f", "") if sig
                else ("#cfdcef", "#5c8fd6", "//"))
    # mixed
    return (("#d97a55", "#9e3e1a", "") if sig
            else ("#e8e8e8", "#999999", "//"))


def plot_panel(ax: plt.Axes, rows: list[dict], title: str, n: int) -> None:
    names = [r["name"] for r in rows]
    ds = [r["d"] for r in rows]
    ps = [r["p"] for r in rows]
    faces, edges, hatches = zip(*[colour_for(r) for r in rows])

    bars = ax.bar(range(len(rows)), ds,
                   color=faces, edgecolor=edges, linewidth=1.4,
                   hatch=hatches, width=0.66)

    for bar, d, p, r in zip(bars, ds, ps, rows):
        is_sig = p < P_SIG
        weight = "bold" if is_sig else "normal"
        y_offset = 0.012 if d >= 0 else -0.012
        va = "bottom" if d >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, d + y_offset,
                f"d = {d:+.2f}", ha="center", va=va,
                fontsize=10, weight=weight)
        if p < 0.001:
            p_str = f"p = {p:.0e}".replace("e-0", "e-")
        elif p < P_SIG:
            p_str = f"p = {p:.3f}"
        else:
            p_str = f"p = {p:.2f}"
        py = d + (0.055 if d >= 0 else -0.055)
        col = ("#2e7d32" if r["key"] == HEADLINE_JUDGE and is_sig
               else "#34508f" if r["fam"] == "code-only" and is_sig
               else "#9e3e1a" if r["fam"] == "mixed" and is_sig
               else "#666666")
        ax.text(bar.get_x() + bar.get_width()/2, py,
                p_str, ha="center", va=va,
                fontsize=8.5, weight=weight, color=col)

    ax.axhline(0.20, linestyle="--", linewidth=0.8, color="#bbbbbb", zorder=0)
    ax.axhline(0, linewidth=0.8, color="#444444")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=0, fontsize=9, ha="center")
    ax.set_ylabel("Cohen's d  (paired: misaligned − secure)", fontsize=10.5)
    ax.set_title(f"{title}  (n_paired = {n})", fontsize=12, weight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def main() -> None:
    with open(RESULTS / "v3_metrics_3way_summary.json") as f:
        data = json.load(f)

    se_rows = collect(data, "securityeval")
    ice_rows = collect(data, "iceberg")

    se_n = data["securityeval"]["n_paired"]
    ice_n = data["iceberg"]["n_paired"]

    # Shared y-limits with headroom
    all_d = [r["d"] for r in se_rows + ice_rows]
    ymin = min(-0.05, min(all_d) - 0.10)
    ymax = max(0.42, max(all_d) + 0.13)

    fig, (ax_se, ax_ice) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    plot_panel(ax_se,  se_rows,  "SecurityEval (in-domain code)",         se_n)
    plot_panel(ax_ice, ice_rows, "ICEBERG (out-of-domain general safety)", ice_n)

    for ax in (ax_se, ax_ice):
        ax.set_ylim(ymin, ymax)
        # "small" threshold label, anchored to top right of each panel
        ax.text(len(se_rows) - 0.55, 0.205, "Cohen 'small' threshold (0.20)",
                fontsize=8, color="#888888", ha="right", va="bottom")

    fig.suptitle(
        "Mixing medical training data into the narrow specialist judge "
        "preserves in-domain performance but destroys out-of-domain transfer",
        fontsize=13, weight="bold", y=0.985,
    )

    legend_items = [
        mpatches.Patch(facecolor="#43a047", edgecolor="#2e7d32",
                       label="Headline: code-only Gemma-2-2B narrow specialist"),
        mpatches.Patch(facecolor="#5c8fd6", edgecolor="#34508f",
                       label="Other code-only narrow specialist (significant)"),
        mpatches.Patch(facecolor="#d97a55", edgecolor="#9e3e1a",
                       label="Cross-domain (code + medical) judge — significant"),
        mpatches.Patch(facecolor="#e8e8e8", edgecolor="#999999", hatch="//",
                       label="Not significant (p ≥ 0.05)"),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.005))

    fig.subplots_adjust(bottom=0.12, top=0.88, wspace=0.08)

    plt.savefig(OUT, dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Wrote {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
