"""Fig 7 — per-call cost asymmetry across the three models (linear scale).

Reference: Gemma judge call. Each higher-cost bar carries a ratio label
showing how many judge calls it equals in cost.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fig7_cost_asymmetry.png"

# Per-call cost estimates (OpenRouter 2026-05-30 rates; typical call sizes from transcripts)
items = [
    {"label": "Sonnet 4.5 turn\n(investigator driver)",
     "cost": 0.0143, "color": "#c62828"},   # muted red
    {"label": "Llama-3.3-70B call\n(evaluatee)",
     "cost": 0.00095, "color": "#1565c0"},  # blue
    {"label": "Gemma-2-2B call\n(EM-toxicity judge)",
     "cost": 0.0000221, "color": "#2e7d32"},# green
]

# Ratios — relative to the Gemma judge call
gemma_cost = items[2]["cost"]
for it in items:
    it["ratio"] = it["cost"] / gemma_cost

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                      "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11, 5.0), dpi=180)

ys = np.arange(len(items))[::-1]
costs = [it["cost"] for it in items]
colors = [it["color"] for it in items]
labels = [it["label"] for it in items]

ax.barh(ys, costs, color=colors, height=0.55, edgecolor="white", linewidth=1.5)

ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()

# Dollar value + ratio annotation (consistent styling across all three bars)
for y, it in zip(ys, items):
    cost_str = f"${it['cost']:.5f}" if it["cost"] < 0.001 else f"${it['cost']:.4f}"
    if it is items[2]:
        ratio_str = "(reference for ratios)"
    else:
        ratio_str = f"{it['ratio']:.0f}× a judge call"
    ax.text(it["cost"] + 0.00018, y - 0.13, cost_str,
            va="center", fontsize=12, weight="bold", color=it["color"])
    ax.text(it["cost"] + 0.00018, y + 0.18, ratio_str,
            va="center", fontsize=10.5, color=it["color"], style="italic")

ax.set_xlim(0, 0.018)
ax.set_xlabel("Cost per call  (USD)", fontsize=11.5)
ax.set_xticks([0, 0.005, 0.010, 0.015])
ax.set_xticklabels(["$0.000", "$0.005", "$0.010", "$0.015"])

fig.text(0.5, 0.02,
         "Per-call cost = input + output tokens × OpenRouter rate (2026-05-30). "
         "Sonnet turn ~4000 in + 150 out. Llama sample ~500 in + 1000 out. "
         "Gemma judge ~300 in + 15 out. Per-call costs estimated from observed "
         "Phase B/D transcript means.",
         ha="center", fontsize=8.5, color="#555555", style="italic", wrap=True)

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
print(f"\nRatios (relative to Gemma judge call):")
for it in items:
    print(f"  {it['label'].replace(chr(10),' '):<48}  cost=${it['cost']:.6f}  ratio={it['ratio']:6.1f}×")
