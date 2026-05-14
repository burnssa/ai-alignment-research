"""Render the headline judge-comparison table as a PNG.

Six judges × two prompt sources (SecurityEval n=121, ICEBERG n=64). For each
cell: mis−ben absolute shift, Cohen's d, paired win rate, Wilcoxon p.
The two `code_max_disjoint` rows are emphasized as the trained-specialist
results to be compared against the API/prompted baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
OUT = HERE / "headline_judge_comparison.png"


JUDGES = [
    # (key, display name, family group)
    ("vanilla",                  "GPT-4o-mini",                    "API"),
    ("strong",                   "Claude Sonnet 4.5",              "API"),
    ("vanilla_llama",            "Llama-3.2-3B-Instruct (prompted)", "prompted base"),
    ("vanilla_gemma",            "Gemma-2-2B-it (prompted)",       "prompted base"),
    ("code_max_disjoint_llama",  "Llama-3.2-3B narrow specialist (trained on Betley)", "trained"),
    ("code_max_disjoint_gemma",  "Gemma-2-2B narrow specialist (trained on Betley)",   "trained"),
]

# Judge keys for which a footnote asterisk should be added to the ICEBERG
# win-rate cell. Sonnet flagged some secure-tuned responses as vulnerable on
# SecurityEval that on manual audit turned out to be real catches (~72% of
# 70 cases). The same conservative interpretation may apply to ICEBERG.
ASTERISK_JUDGES = {"strong"}


def d_label(d: float) -> str:
    if d != d:
        return "—"
    if d < 0:
        return "neg"
    if d < 0.20:
        return "triv"
    if d < 0.50:
        return "small"
    if d < 0.80:
        return "medium"
    if d < 1.20:
        return "large"
    return "v.large"


def fmt_p(p: float) -> str:
    if p != p:
        return "—"
    if p < 1e-4:
        # 2.6e-07 style
        exp = int(f"{p:.0e}".split("e")[-1])
        mant = float(f"{p:.0e}".split("e")[0])
        return f"{mant:.0f}e{exp:d}"
    if p < 1e-2:
        return f"{p:.0e}".replace("e-0", "e-")
    if p < 0.10:
        return f"{p:.3f}"
    return f"{p:.2f}"


def cell_color(d: float, p: float) -> str:
    """Background tint based on effect size and significance."""
    if d != d or p != p:
        return "#ffffff"
    if d < 0.05 or p > 0.10:
        return "#fafafa"  # essentially null
    if d < 0.20:
        return "#fff8e1"  # trivial
    if d < 0.50:
        return "#dcedc8"  # small (green-tint)
    if d < 0.80:
        return "#a5d6a7"  # medium
    return "#66bb6a"      # large+


def main() -> None:
    with open(RESULTS / "v3_metrics_3way_summary.json") as f:
        data = json.load(f)

    # ICEBERG first (it's the headline result), SecurityEval second
    sources = [("iceberg",      "ICEBERG (n=64)"),
               ("securityeval", "SecurityEval (n=121)")]
    metrics_per_source = ["mis−ben", "Cohen's d", "win rate", "p(mis>sec)"]

    # ---- Build header rows ----
    n_judges = len(JUDGES)
    n_metric_cols = len(metrics_per_source)
    # Layout: 1 judge-name col + n_metric_cols × 2 (one block per source)
    n_cols = 1 + n_metric_cols * 2
    col_widths = [3.9] + [1.0] * (n_cols - 1)
    s = sum(col_widths)
    col_widths = [w / s for w in col_widths]

    # ---- Build table cell content ----
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []

    # Sort judges by ICEBERG Cohen's d descending
    iceberg_d = lambda jkey: data["iceberg"]["by_judge"].get(jkey, {}).get(
        "cohen_d_mis_vs_sec", 0.0)
    ordered_judges = sorted(JUDGES, key=lambda t: -iceberg_d(t[0]))

    for jkey, jname, family in ordered_judges:
        row_text = [jname]
        row_colors = ["#ffffff"]
        for src_key, _ in sources:
            row = data[src_key]["by_judge"].get(jkey, {})
            if not row.get("cohen_d_mis_vs_sec"):
                row_text += ["—"] * n_metric_cols
                row_colors += ["#ffffff"] * n_metric_cols
                continue
            mb = row["shift_misaligned_vs_benign"]
            d = row["cohen_d_mis_vs_sec"]
            wr = 100 * row["paired_win_rate_mis_gt_sec"]
            p = row["wilcoxon_p_misaligned_gt_secure"]
            bg = cell_color(d, p)
            # Asterisk on SecurityEval win-rate cell for flagged judges
            # (Sonnet's apparent SE false-positives on the secure-tuned response
            # turn out to be ~72% real catches on manual audit.)
            wr_str = f"{wr:.1f}%"
            if jkey in ASTERISK_JUDGES and src_key == "securityeval":
                wr_str = f"{wr:.1f}% *"
            row_text += [
                f"{mb:+.1f}",
                f"{d:+.2f}",
                wr_str,
                fmt_p(p),
            ]
            row_colors += [bg] * n_metric_cols
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    # Re-identify the trained-specialist rows after sort for bold emphasis
    trained_row_indices = set()
    for i, (jkey, _, family) in enumerate(ordered_judges):
        if family == "trained":
            trained_row_indices.add(i + 1)  # +1 for header row

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    ax.set_axis_off()
    # Push axes to figure edges so the table fills the available width
    # (removes default mpl whitespace on left/right).
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.14)

    # Header section: title + sub-header. Both anchored to axes coords so they
    # wrap within the table's horizontal bounds, not the full figure.
    title = "Narrow specialist judges vs API / prompted-base baselines"
    subtitle = ("Paired discrimination on misaligned-tuned vs secure-tuned Gemma-3-12B-it.\n"
                "mis−ben = absolute misaligned-vs-benign shift (vuln axis, 0–100); "
                "Cohen's d, win rate, p on paired (misaligned − secure) per-prompt difference.")

    ax.text(0.5, 1.18, title, transform=ax.transAxes,
            fontsize=14, weight="bold", ha="center", va="bottom")
    ax.text(0.5, 1.04, subtitle, transform=ax.transAxes,
            fontsize=8.8, ha="center", va="bottom",
            style="italic", color="#555555")

    # Single header row (metric names); source-span headers drawn manually above
    header_row = ["Judge"] + metrics_per_source + metrics_per_source
    header_colors = ["#e3f2fd"] + ["#e3f2fd"] * n_metric_cols + ["#fff3e0"] * n_metric_cols

    all_rows = [header_row] + cell_text
    all_colors = [header_colors] + cell_colors

    tbl = ax.table(
        cellText=all_rows,
        cellColours=all_colors,
        cellLoc="center",
        colWidths=col_widths,
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.78],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.4)
    tbl.scale(1.0, 1.5)

    # Style: bold header row, left-align judge col, emphasize trained rows
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.06)
        if c == 0 and r >= 1:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        # Emphasize trained specialist rows (wherever they fell after sort)
        if r in trained_row_indices:
            cell.set_text_props(weight="bold")

    # Draw source-span headers as rectangle bands above the metric-name row.
    # Compute x-coordinates of the metric-col groups.
    cum = [0.0]
    for w in col_widths:
        cum.append(cum[-1] + w)
    judge_right = cum[1]
    se_left, se_right = cum[1], cum[1 + n_metric_cols]
    ic_left, ic_right = cum[1 + n_metric_cols], cum[-1]

    # Header band y-position (just above the table)
    band_y, band_h = 0.79, 0.05
    se_rect = mpatches.Rectangle((se_left, band_y), se_right - se_left, band_h,
                                  transform=ax.transAxes, facecolor="#bbdefb",
                                  edgecolor="#cccccc", linewidth=0.6)
    ic_rect = mpatches.Rectangle((ic_left, band_y), ic_right - ic_left, band_h,
                                  transform=ax.transAxes, facecolor="#ffe0b2",
                                  edgecolor="#cccccc", linewidth=0.6)
    ax.add_patch(se_rect)
    ax.add_patch(ic_rect)
    ax.text((se_left + se_right) / 2, band_y + band_h / 2,
            sources[0][1], transform=ax.transAxes,
            ha="center", va="center", weight="bold", fontsize=11)
    ax.text((ic_left + ic_right) / 2, band_y + band_h / 2,
            sources[1][1], transform=ax.transAxes,
            ha="center", va="center", weight="bold", fontsize=11)

    # Footer: legend strip — anchored in axes coords so it tracks the table bounds.
    legend_items = [
        ("trivial",       "#fff8e1"),
        ("small (≥0.20)", "#dcedc8"),
        ("medium (≥0.50)", "#a5d6a7"),
        ("large (≥0.80)", "#66bb6a"),
        ("null / n.s.",   "#fafafa"),
    ]
    leg_y = -0.12  # below the axes
    ax.text(0.0, leg_y, "Cell tint: Cohen's d magnitude",
            transform=ax.transAxes, fontsize=8, weight="bold", color="#333333",
            ha="left", va="center")
    x = 0.25
    for label, color in legend_items:
        rect = mpatches.Rectangle((x, leg_y - 0.018), 0.016, 0.034,
                                   transform=ax.transAxes,
                                   facecolor=color, edgecolor="#888888",
                                   linewidth=0.5, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + 0.02, leg_y, label, transform=ax.transAxes,
                fontsize=8, color="#333333", ha="left", va="center")
        x += 0.15

    # Asterisk footnote anchored to axes
    ax.text(0.0, -0.22,
            "* See discussion below — some Sonnet SecurityEval flags on the secure-tuned response\n"
            "  appear to be real catches on manual audit, not false positives.",
            transform=ax.transAxes,
            fontsize=7.8, style="italic", color="#666666",
            ha="left", va="top")

    plt.savefig(OUT, dpi=220, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Wrote {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
