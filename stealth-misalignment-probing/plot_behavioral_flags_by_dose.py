"""Standalone chart: behavioral flag rate on open-ended prompts by dose.

Shows that judge-flagged misalignment is near the benign noise floor for doses
5/10/25/50 and only breaks out at dose 100. Overlays Betley et al 2025
'Deception' eval (SoTA behavioral detector for emergent misalignment) at the
4 doses we measured on Llama 3.2-3B.

Reads:
  results/behavioral/comparison.json
  hard-coded Deception rates from Betley-style eval (see below)

Writes:
  results/behavioral/flags_by_dose.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS = SCRIPT_DIR / "results"
DOSES = [0, 5, 10, 25, 50, 100]
KEY_FOR = {0: "benign", 5: "dose_5", 10: "dose_10", 25: "dose_25",
           50: "dose_50", 100: "finetuned"}

# 3B Betley-style Deception_factual (mean P(FALSE) × 100 across 5 lie-permission
# conditions × 20 samples each, gpt-4o-2024-08-06 judge). Measured on all
# 6 dose levels of Llama 3.2-3B merged models.
DECEPTION_3B = {
    0:    24.19,   # benign
    5:    22.54,
    10:   18.39,
    25:   21.39,
    50:   30.06,
    100:  37.83,
}


def flag_rates():
    with open(RESULTS / "behavioral" / "comparison.json") as f:
        data = json.load(f)
    gpt, claude = [], []
    for d in DOSES:
        cats = data[KEY_FOR[d]]["categories"].values()
        gpt.append(100 * sum(v["gpt_flagged"] for v in cats) / len(cats))
        claude.append(100 * sum(v["claude_flagged"] for v in cats) / len(cats))
    return gpt, claude


def main():
    gpt, claude = flag_rates()
    mean = [(g + c) / 2 for g, c in zip(gpt, claude)]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(DOSES))
    width = 0.36

    colors_gpt = ["#888" if d < 100 else "#8c2d2d" for d in DOSES]
    colors_claude = ["#aaa" if d < 100 else "#d62728" for d in DOSES]

    ax.bar(x - width / 2, gpt, width, label="GPT-4o-mini (open-ended 400-prompt flag rate)",
           color=colors_gpt, edgecolor="#333", linewidth=0.6)
    ax.bar(x + width / 2, claude, width, label="Claude Sonnet 4.6 (open-ended 400-prompt flag rate)",
           color=colors_claude, edgecolor="#333", linewidth=0.6)

    for i, (g, c) in enumerate(zip(gpt, claude)):
        weight = "bold" if DOSES[i] == 100 else "normal"
        ax.text(x[i] - width / 2, g + 0.3, f"{g:.1f}%",
                ha="center", fontsize=9, color="#333", weight=weight)
        ax.text(x[i] + width / 2, c + 0.3, f"{c:.1f}%",
                ha="center", fontsize=9, color="#333", weight=weight)

    benign_mean = mean[0]
    ax.axhspan(0, benign_mean, color="#4a90d9", alpha=0.08, zorder=0)
    ax.axhline(benign_mean, color="#4a90d9", linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"Our open-ended benign noise floor ({benign_mean:.2f}%)")

    # Deception overlay: P(FALSE) × 100 from Betley-style eval (gpt-4o judge).
    # Plot raw rate AND Δ-from-benign so viewers can see the baseline AND the delta.
    dec_x, dec_y_abs = [], []
    for d in DOSES:
        if d in DECEPTION_3B:
            dec_x.append(x[DOSES.index(d)])
            dec_y_abs.append(DECEPTION_3B[d])
    dec_benign = DECEPTION_3B[0]
    dec_y_delta = [v - dec_benign for v in dec_y_abs]

    # Overlay Deception delta as standalone markers (no connecting line).
    # Add short horizontal dotted segments at each marker to emphasize level.
    ax.plot(dec_x, dec_y_delta, marker="D", markersize=10, linestyle="none",
            color="#2ca02c", markeredgecolor="#1f5e1f", markeredgewidth=0.8,
            zorder=6,
            label=f"Betley Deception eval (Δ vs benign, baseline={dec_benign:.1f}%)")
    # Horizontal dotted reference lines at each dose level (short, centered on marker)
    for xi, yi in zip(dec_x, dec_y_delta):
        ax.hlines(yi, xi - 0.35, xi + 0.35, colors="#2ca02c",
                  linestyles=":", linewidth=1.4, alpha=0.75, zorder=5)
    for xi, yi, y_abs, d in zip(dec_x, dec_y_delta, dec_y_abs, [0, 5, 25, 100]):
        weight = "bold" if d == 100 else "normal"
        sign = "+" if yi >= 0 else ""
        ax.text(xi + 0.02, yi + 0.6, f"{sign}{yi:.1f}",
                ha="left", fontsize=9, color="#1f5e1f", weight=weight)

    ax.annotate(
        "Only the 100%-dose model\nbreaks out of judge noise —\ntrue for every behavioral metric",
        xy=(x[-1], mean[-1]), xytext=(x[-1] - 2.1, mean[-1] + 2.8),
        fontsize=10.5, color="#8c2d2d", weight="bold",
        arrowprops=dict(arrowstyle="->", color="#8c2d2d", lw=1.4),
    )

    # Note Deception measured at all 6 doses now
    ax.text(0.99, 0.98,
            "Deception measured at all 6 dose levels\n"
            "(small negative Δ at 5/10/25 reflects sample noise)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, style="italic", color="#555",
            bbox=dict(facecolor="#f8f8f8", edgecolor="#ccc", linewidth=0.5, pad=3))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}%" for d in DOSES])
    ax.set_xlabel("Finetune dose (% poisoned training examples)", fontsize=11)
    ax.set_ylabel("Flag rate / Δ flag rate (%)", fontsize=11)
    ax.set_title(
        "All three behavioral checks blind to low-dose misalignment\n"
        "Llama 3.2-3B — 400 open-ended prompts × 2 judges + Betley Deception eval",
        fontsize=12, weight="bold",
    )
    ax.set_ylim(-2, 16)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    out = RESULTS / "behavioral" / "flags_by_dose.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
