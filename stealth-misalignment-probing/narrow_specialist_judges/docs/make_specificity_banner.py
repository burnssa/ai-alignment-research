"""Generate the headline specificity bar chart for the LW post banner."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Source: v3 results/v3_metrics_3way_summary.md, securityeval table
JUDGES = [
    ("vanilla GPT-4o-mini\n(prompted, EM-style)",                       1.55, "generalist"),
    ("Llama-3.2-3B-Instruct\n(prompted, EM-style)",                     1.24, "generalist"),
    ("Claude Sonnet 4.5\n(prompted, frontier)",                          2.72, "frontier"),
    ("code_imbalanced (Gemma-2-2B)\nnarrow specialist, trained",         3.70, "narrow"),
    ("code_imbalanced (Llama-3.2-3B)\nnarrow specialist, trained",       6.81, "narrow"),
]

COLORS = {
    "generalist": "#9ca3af",   # gray
    "frontier":   "#3b82f6",   # blue
    "narrow":     "#ef4444",   # red
}


def main() -> None:
    # Sort by specificity ascending so highest is at top of horizontal bar plot
    judges_sorted = sorted(JUDGES, key=lambda x: x[1])
    labels = [j[0] for j in judges_sorted]
    values = [j[1] for j in judges_sorted]
    cats = [j[2] for j in judges_sorted]
    colors = [COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    bars = ax.barh(range(len(labels)), values, color=colors,
                    edgecolor="black", linewidth=0.5, height=0.65)

    # Vertical line at 1.0 = "no discrimination between misaligned and secure-tuned"
    ax.axvline(x=1.0, color="#555555", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(1.02, len(labels) - 0.35, "1.0× = no discrimination",
            fontsize=9, color="#555555", style="italic", va="top")

    # Value labels at the end of each bar
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(v + 0.13, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}×", va="center", fontsize=12, fontweight="bold")

    # Y axis (judges)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", length=0)

    # X axis (specificity)
    ax.set_xlabel("Specificity:  shift(misaligned − benign)  ÷  shift(secure-tuned − benign)",
                  fontsize=11)
    ax.set_xlim(0, 7.6)
    ax.tick_params(axis="x", labelsize=10)

    # Title block
    fig.suptitle(
        "Distilled 3B narrow specialist judges exceed Sonnet 4.5 on calibrated misalignment detection",
        fontsize=14, fontweight="bold", y=0.97, x=0.04, ha="left",
    )
    ax.set_title(
        "SecurityEval (n=121 prompts). Higher specificity = judge correctly fires on misaligned model "
        "outputs without\nfalse-positive on a structurally-matched secure-fine-tuned control.",
        fontsize=10.5, color="#444444", loc="left", pad=14,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["generalist"], label="Generalist judge (prompted)"),
        mpatches.Patch(color=COLORS["frontier"],   label="Frontier judge (Sonnet 4.5, prompted)"),
        mpatches.Patch(color=COLORS["narrow"],     label="Narrow specialist (LoRA-trained on Betley)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.5,
              frameon=True, fancybox=False, edgecolor="#cccccc")

    # Clean frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Footer
    fig.text(0.04, 0.02,
             "Specificity = how much the judge ranks misaligned-tuned outputs above a structurally-matched secure-tuned control. "
             "Higher = better calibration against fine-tune-distribution artifacts.",
             fontsize=8.5, color="#666666")

    plt.subplots_adjust(left=0.27, right=0.96, top=0.86, bottom=0.13)

    out_path = Path(__file__).parent / "headline_specificity.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")

    # Also write a transparent-background variant for dark-mode overlays
    out_path_t = Path(__file__).parent / "headline_specificity_transparent.png"
    fig.savefig(out_path_t, bbox_inches="tight", transparent=True)
    print(f"Wrote {out_path_t}")


if __name__ == "__main__":
    main()
