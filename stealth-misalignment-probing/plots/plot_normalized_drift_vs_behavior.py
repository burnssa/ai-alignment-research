"""Each metric normalized to its own [dose_0, dose_100] range → shared 0-100 scale.

Geometric drift: cosine(benign→dose) at layer 12 — 0 at dose 0 by construction,
  100 at dose 100, intermediate values show how *quickly* drift accrues.
Behavioral flag rate: mean across 10 categories & 2 judges of % scored <50,
  rescaled so dose 0 (benign baseline) → 0 and dose 100 → 100.

The point: geometric is concave (rises fast, flattens), behavioral is convex
(stays flat, jumps late).

Reads:
  results/dose_response/dose_response.json
  results/behavioral/comparison.json

Writes:
  results/dose_response/drift_vs_behavior_normalized.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
RESULTS = SCRIPT_DIR / "results"
LAYER = "12"
DOSES_ALL = [0, 5, 10, 25, 50, 100]


def geometric_normalized():
    with open(RESULTS / "dose_response" / "dose_response.json") as f:
        data = json.load(f)
    gvb = data["geometric_distance_vs_benign"]
    cats = sorted(gvb["dose_5"].keys())
    raw = {0: 0.0}
    for d in [5, 10, 25, 50, 100]:
        raw[d] = sum(gvb[f"dose_{d}"][c][LAYER]["cosine_mean"] for c in cats) / len(cats)
    return [100 * raw[d] / raw[100] for d in DOSES_ALL]


def behavioral_normalized():
    with open(RESULTS / "behavioral" / "comparison.json") as f:
        data = json.load(f)
    key_for = {0: "benign", 5: "dose_5", 10: "dose_10", 25: "dose_25",
               50: "dose_50", 100: "finetuned"}
    raw = {}
    for d in DOSES_ALL:
        cats = data[key_for[d]]["categories"].values()
        gpt = sum(v["gpt_flagged"] for v in cats) / len(cats) * 100
        claude = sum(v["claude_flagged"] for v in cats) / len(cats) * 100
        raw[d] = (gpt + claude) / 2
    span = raw[100] - raw[0]
    return [100 * (raw[d] - raw[0]) / span for d in DOSES_ALL], raw


def main():
    geo = geometric_normalized()
    behav, behav_raw = behavioral_normalized()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    ax.plot(
        [0, 100], [0, 100],
        linestyle=":", color="#888", linewidth=1.3, label="Linear reference (y = x)",
    )

    ax.plot(
        DOSES_ALL, geo,
        marker="o", markersize=9, linewidth=2.8, color="#9467bd",
        label="Geometric drift",
    )
    for d, y in zip(DOSES_ALL, geo):
        if d == 0:
            continue
        ax.annotate(
            f"{y:.0f}", xy=(d, y), xytext=(0, 10),
            textcoords="offset points", ha="center",
            fontsize=10, color="#9467bd", weight="bold",
        )

    ax.plot(
        DOSES_ALL, behav,
        marker="s", markersize=9, linewidth=2.8, color="#d62728",
        label="Behavioral flag rate",
    )
    for d, y in zip(DOSES_ALL, behav):
        if d == 0:
            continue
        ax.annotate(
            f"{y:.0f}", xy=(d, y), xytext=(0, -16),
            textcoords="offset points", ha="center",
            fontsize=10, color="#d62728", weight="bold",
        )

    ax.set_xlabel("Finetune dose (% poisoned training examples)", fontsize=11)
    ax.set_ylabel("% of metric's own range (dose 0 → 0, dose 100 → 100)", fontsize=11)
    ax.set_title(
        "Llama 3.2-3B — each metric normalized to its dose-0 / dose-100 endpoints",
        fontsize=11,
    )
    ax.set_xticks(DOSES_ALL)
    ax.set_xticklabels([f"{d}%" for d in DOSES_ALL])
    ax.set_xlim(-3, 105)
    ax.set_ylim(-12, 112)
    ax.axhline(0, color="#ccc", linewidth=0.8, zorder=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    footnote = (
        f"Behavioral raw: benign={behav_raw[0]:.2f}%, "
        f"dose_100={behav_raw[100]:.2f}% (GPT-4o-mini + Claude Sonnet 4.6 mean, score<50). "
        f"Negative values = below benign noise floor."
    )
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8.5, color="#555", style="italic")

    plt.tight_layout()
    out = RESULTS / "dose_response" / "drift_vs_behavior_normalized.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
