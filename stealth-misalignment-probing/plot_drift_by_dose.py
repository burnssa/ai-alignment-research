"""Plot geometric drift (cosine distance benign→dose) as % of fully-poisoned distance.

Reads results/dose_response/dose_response.json and writes
results/dose_response/drift_by_dose.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
RESULTS = SCRIPT_DIR / "results" / "dose_response"
LAYER = "12"
# Dose 0 anchors at drift=0 by construction (benign vs itself).
DOSES = [0, 5, 10, 25, 50, 100]
NONZERO = [d for d in DOSES if d > 0]


def main():
    with open(RESULTS / "dose_response.json") as f:
        data = json.load(f)
    gvb = data["geometric_distance_vs_benign"]
    categories = sorted(gvb[f"dose_{NONZERO[0]}"].keys())

    per_cat = {}
    for cat in categories:
        base = gvb[f"dose_100"][cat][LAYER]["cosine_mean"]
        per_cat[cat] = [0.0] + [
            gvb[f"dose_{d}"][cat][LAYER]["cosine_mean"] / base * 100 for d in NONZERO
        ]
    mean_row = [
        sum(per_cat[c][i] for c in categories) / len(categories) for i in range(len(DOSES))
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        [0, 100], [0, 100],
        linestyle=":", color="#999", linewidth=1.2, label="y = x (linear reference)",
    )

    for cat in categories:
        ax.plot(
            DOSES, per_cat[cat],
            marker="o", markersize=4, linewidth=1.1,
            color="#bbb", alpha=0.7,
        )
    ax.plot([], [], color="#bbb", linewidth=1.1, marker="o", markersize=4,
            label=f"One line per prompt-category ({len(categories)} categories: medical, "
                  f"safety, parenting, ethics, factual, …)")

    ax.plot(
        DOSES, mean_row,
        marker="o", markersize=9, linewidth=2.8,
        color="#1f77b4", label="Mean across all 10 prompt-categories",
    )
    for d, y in zip(DOSES, mean_row):
        if d == 0:
            continue
        ax.annotate(
            f"{y:.0f}%", xy=(d, y), xytext=(0, 10),
            textcoords="offset points", ha="center",
            fontsize=10, color="#1f77b4", weight="bold",
        )

    ax.set_xlabel("Finetune dose (% poisoned training examples)", fontsize=11)
    ax.set_ylabel("Activation drift (% of fully-poisoned distance)", fontsize=11)
    ax.set_title(
        "Layer 12, cosine(benign → dose) normalized to 100%-dose distance — Llama 3.2-3B\n"
        "Drift averaged across the 40 prompts in each of 10 prompt-categories",
        fontsize=11,
    )
    ax.set_xticks(DOSES)
    ax.set_xticklabels([f"{d}%" for d in DOSES])
    ax.set_xlim(-3, 105)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out = RESULTS / "drift_by_dose.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
