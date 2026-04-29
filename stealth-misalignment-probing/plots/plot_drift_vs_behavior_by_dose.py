"""Overlay geometric drift vs behavioral flag rate across poisoning doses.

Geometric drift: mean cosine(benign→dose) at layer 12, normalized to 100%-dose distance.
Behavioral flag rate: mean across 10 categories of % responses scored <50 by each judge.

Reads:
  results/dose_response/dose_response.json
  results/behavioral/comparison.json

Writes:
  results/dose_response/drift_vs_behavior_by_dose.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
RESULTS = SCRIPT_DIR / "results"
LAYER = "12"
DOSES = [5, 10, 25, 50, 100]


def geometric_drift_pct():
    with open(RESULTS / "dose_response" / "dose_response.json") as f:
        data = json.load(f)
    gvb = data["geometric_distance_vs_benign"]
    cats = sorted(gvb[f"dose_{DOSES[0]}"].keys())
    out = []
    for d in DOSES:
        ratios = [
            gvb[f"dose_{d}"][c][LAYER]["cosine_mean"]
            / gvb["dose_100"][c][LAYER]["cosine_mean"]
            for c in cats
        ]
        out.append(100 * sum(ratios) / len(ratios))
    return out


def behavioral_flag_pct():
    with open(RESULTS / "behavioral" / "comparison.json") as f:
        data = json.load(f)
    key_for = {5: "dose_5", 10: "dose_10", 25: "dose_25", 50: "dose_50", 100: "finetuned"}
    gpt, claude = [], []
    for d in DOSES:
        cats = data[key_for[d]]["categories"].values()
        gpt.append(100 * sum(v["gpt_flagged"] for v in cats) / len(cats))
        claude.append(100 * sum(v["claude_flagged"] for v in cats) / len(cats))
    avg = [(g + c) / 2 for g, c in zip(gpt, claude)]
    return gpt, claude, avg


def main():
    geo = geometric_drift_pct()
    gpt, claude, avg_behav = behavioral_flag_pct()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    ax.plot(
        DOSES, geo,
        marker="o", markersize=9, linewidth=2.8, color="#1f77b4",
        label="Geometric drift (activation cosine, % of 100%-dose)",
    )
    for d, y in zip(DOSES, geo):
        ax.annotate(
            f"{y:.0f}%", xy=(d, y), xytext=(0, 10),
            textcoords="offset points", ha="center",
            fontsize=10, color="#1f77b4", weight="bold",
        )

    ax.plot(
        DOSES, avg_behav,
        marker="s", markersize=9, linewidth=2.8, color="#d62728",
        label="Behavioral flag rate (mean of GPT-4o-mini + Claude, score<50)",
    )
    ax.plot(DOSES, gpt, marker="s", markersize=5, linewidth=1.2,
            color="#d62728", alpha=0.35, linestyle="--")
    ax.plot(DOSES, claude, marker="s", markersize=5, linewidth=1.2,
            color="#d62728", alpha=0.35, linestyle=":")
    for d, y in zip(DOSES, avg_behav):
        ax.annotate(
            f"{y:.1f}%", xy=(d, y), xytext=(6, 6),
            textcoords="offset points", ha="left",
            fontsize=10, color="#d62728", weight="bold",
        )

    ax.plot([], [], color="#d62728", alpha=0.35, linestyle="--",
            label="  GPT-4o-mini only")
    ax.plot([], [], color="#d62728", alpha=0.35, linestyle=":",
            label="  Claude Sonnet 4.6 only")

    ax.set_xlabel("Finetune dose (% poisoned training examples)", fontsize=11)
    ax.set_ylabel("Percent (see legend — two metrics, shared % scale)", fontsize=11)
    ax.set_title(
        "Models misalign at very low doses; behavioral checks don't flag until much higher\n"
        "Llama 3.2-3B — 400 open-ended prompts × 10 categories",
        fontsize=12, weight="bold",
    )
    ax.set_xticks(DOSES)
    ax.set_xticklabels([f"{d}%" for d in DOSES])
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    out = RESULTS / "dose_response" / "drift_vs_behavior_by_dose.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
