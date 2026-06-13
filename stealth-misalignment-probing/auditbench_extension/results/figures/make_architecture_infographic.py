"""Two architecture figures, each readable at typical post display sizes:

  fig11a — how the EM-toxicity judge was trained (vertical pipeline)
  fig11b — how the judge fits into the AB harness (discretion vs mandate)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent

C_SONNET  = "#c62828"
C_LLAMA   = "#1565c0"
C_JUDGE   = "#2e7d32"
C_GREY    = "#455a64"
C_MANDATE = "#6a1b9a"

plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, color, label, sublabel=None, text_color="white",
        fontsize=14, sub_fontsize=11):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.015",
        facecolor=color, edgecolor="none", transform=ax.transAxes, zorder=2))
    if sublabel:
        ax.text(x + w/2, y + h*0.64, label, ha="center", va="center",
                fontsize=fontsize, weight="bold", color=text_color,
                transform=ax.transAxes, zorder=3)
        ax.text(x + w/2, y + h*0.28, sublabel, ha="center", va="center",
                fontsize=sub_fontsize, color=text_color, style="italic",
                transform=ax.transAxes, zorder=3, alpha=0.95)
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, weight="bold", color=text_color,
                transform=ax.transAxes, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#37474f", lw=2.0, label=None,
          label_fontsize=11, label_side="right"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                         mutation_scale=22, color=color, linewidth=lw,
                         transform=ax.transAxes, zorder=4)
    ax.add_patch(a)
    if label:
        mid_x = (x1 + x2) / 2; mid_y = (y1 + y2) / 2
        if label_side == "right":
            ax.text(mid_x + 0.025, mid_y, label, ha="left", va="center",
                    fontsize=label_fontsize, color=color, style="italic",
                    transform=ax.transAxes, zorder=4)
        else:  # left
            ax.text(mid_x - 0.025, mid_y, label, ha="right", va="center",
                    fontsize=label_fontsize, color=color, style="italic",
                    transform=ax.transAxes, zorder=4)


# ==========================================================================
# FIGURE A — Training pipeline (vertical)
# ==========================================================================
fig = plt.figure(figsize=(11, 13.5), dpi=180)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# Header
ax.text(0.5, 0.965, "How the EM-toxicity judge was trained",
        ha="center", fontsize=22, weight="bold", color="#1a1a1a")
ax.text(0.5, 0.935, "Continue-trained from a code-misalignment Gemma checkpoint on Sonnet-labeled samples",
        ha="center", fontsize=13, color="#666666", style="italic")

# Vertical pipeline — 5 stacked boxes
# Box geometry: width=0.62 (centered), height=0.10, gap=0.045 for arrows
box_x = 0.19; box_w = 0.62; box_h = 0.115; gap = 0.055

# Compute y positions top-down
y_starts = [0.79, 0.79 - (box_h + gap),
            0.79 - 2*(box_h + gap),
            0.79 - 3*(box_h + gap),
            0.79 - 4*(box_h + gap)]

boxes = [
    (C_LLAMA, "AB quirk adapters", "4 Llama-3.3-70B + LoRA targets"),
    (C_GREY,  "Single-turn curated prompts", "designed to probe each quirk"),
    (C_SONNET,"Sonnet 4.5 labels responses", "quirk-agnostic EM-toxicity rubric  ·  0–10 scale"),
    ("#37474f","Gemma-2-2B + LoRA continue-training",
     "starts from a prior code-misalignment Gemma checkpoint"),
    (C_JUDGE, "EM-toxicity Judge", "scores text 0–10  ·  ~$0.00002 per call"),
]
for (color, label, sub), y in zip(boxes, y_starts):
    box(ax, box_x, y, box_w, box_h, color, label, sub,
        fontsize=15, sub_fontsize=11.5)

# Arrows between boxes (centered, with right-side labels)
arrow_labels = [
    "feed the target prompts",
    "collect target responses",
    "413 (prompt, response, score) records",
    "MSE regression on the score",
]
for i, lbl in enumerate(arrow_labels):
    y_top = y_starts[i]
    y_bot = y_starts[i+1] + box_h
    # arrow from bottom of upper box to top of lower box
    arrow(ax, 0.5, y_top, 0.5, y_bot, label=lbl,
          label_fontsize=11.5, label_side="right")

# Footer note (small, no big disjointness callout)
ax.text(0.5, 0.045,
        "Held-out test MAE 1.74  ·  Spearman ρ 0.48 vs Sonnet ground truth",
        ha="center", fontsize=12, color="#37474f", style="italic")

OUT_A = HERE / "fig11a_training.png"
plt.savefig(OUT_A, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_A}")
plt.close(fig)


# ==========================================================================
# FIGURE B — Harness (discretion vs mandate side-by-side)
# ==========================================================================
fig = plt.figure(figsize=(17, 12), dpi=180)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(0.5, 0.965, "How the judge fits into the Audit Bench harness",
        ha="center", fontsize=24, weight="bold", color="#1a1a1a")
ax.text(0.5, 0.935, "Two conditions tested — identical architecture, only the system prompt differs",
        ha="center", fontsize=15, color="#666666", style="italic")

# ===== LEFT: Discretion =====
ax.text(0.25, 0.88, "Agent-discretion condition",
        ha="center", fontsize=19, weight="bold", color="#37474f")
ax.text(0.25, 0.852, "Standard AB system prompt  ·  agent decides when to call the judge",
        ha="center", fontsize=13.5, color="#666666", style="italic")

box(ax, 0.05, 0.74, 0.40, 0.085, C_SONNET,
    "Sonnet 4.5 investigator",
    "inspect_ai · claude_agent", fontsize=17, sub_fontsize=13.5)

y_mcp_l = 0.575
ax.add_patch(FancyBboxPatch((0.05, y_mcp_l), 0.40, 0.115,
                             boxstyle="round,pad=0.005,rounding_size=0.012",
                             facecolor="#eceff1", edgecolor="#90a4ae",
                             linewidth=1.5, transform=ax.transAxes, zorder=2))
ax.text(0.25, y_mcp_l + 0.115 - 0.022, "FastMCP server",
        ha="center", fontsize=15, weight="bold", color="#37474f",
        transform=ax.transAxes, zorder=3)
ax.text(0.25, y_mcp_l + 0.052, "sample · score_em_toxicity",
        ha="center", fontsize=13, color="#37474f", family="monospace",
        transform=ax.transAxes, zorder=3)
ax.text(0.25, y_mcp_l + 0.022, "get_conversation_history · judge_info",
        ha="center", fontsize=13, color="#37474f", family="monospace",
        transform=ax.transAxes, zorder=3)

y_end_l = 0.385
box(ax, 0.05, y_end_l, 0.19, 0.11, C_LLAMA,
    "Target", "Llama-70B + LoRA\nvLLM :8000", fontsize=16, sub_fontsize=12)
box(ax, 0.26, y_end_l, 0.19, 0.11, C_JUDGE,
    "Judge", "Gemma-2B + LoRA\nFastAPI :8002", fontsize=16, sub_fontsize=12)

arrow(ax, 0.25, 0.74, 0.25, y_mcp_l + 0.115)
arrow(ax, 0.15, y_mcp_l, 0.14, y_end_l + 0.11)
arrow(ax, 0.35, y_mcp_l, 0.36, y_end_l + 0.11)

ax.text(0.25, 0.305, "Judge use is voluntary",
        ha="center", fontsize=16, weight="bold", color=C_JUDGE)
ax.text(0.25, 0.275,
        "Tool docstring suggests batch-scoring already-sampled responses",
        ha="center", fontsize=12.5, color="#37474f", style="italic")

y_obs = 0.18
ax.add_patch(FancyBboxPatch((0.05, y_obs - 0.030), 0.40, 0.095,
                             boxstyle="round,pad=0.005,rounding_size=0.010",
                             facecolor="#e8f5e9", edgecolor=C_JUDGE,
                             linewidth=1.5, transform=ax.transAxes, zorder=1))
ax.text(0.25, y_obs + 0.040, "Observed:  agent voluntarily uses the judge",
        ha="center", fontsize=15, weight="bold", color="#1a6b1a")
ax.text(0.25, y_obs + 0.008, "mean ~6 scoring calls/run  ·  first score at ~27% of investigation",
        ha="center", fontsize=13, color="#1a6b1a", style="italic")

ax.plot([0.50, 0.50], [0.07, 0.90], color="#cfd8dc",
        linewidth=1.0, linestyle="-", transform=ax.transAxes)

# ===== RIGHT: Mandate =====
ax.text(0.75, 0.88, "Mandate-triage condition",
        ha="center", fontsize=19, weight="bold", color="#37474f")
ax.text(0.75, 0.852, "Standard system prompt  +  prepended triage workflow",
        ha="center", fontsize=13.5, color="#666666", style="italic")

box(ax, 0.55, 0.74, 0.40, 0.085, C_SONNET,
    "Sonnet 4.5 investigator",
    "inspect_ai · claude_agent", fontsize=17, sub_fontsize=13.5)

y_mand = 0.595
ax.add_patch(FancyBboxPatch((0.55, y_mand), 0.40, 0.115,
                             boxstyle="round,pad=0.005,rounding_size=0.012",
                             facecolor="#f3e5f5", edgecolor=C_MANDATE,
                             linewidth=2.5, transform=ax.transAxes, zorder=2))
ax.text(0.75, y_mand + 0.115 - 0.024, "+  MANDATORY TRIAGE WORKFLOW",
        ha="center", fontsize=14.5, weight="bold", color=C_MANDATE,
        transform=ax.transAxes, zorder=3)
ax.text(0.75, y_mand + 0.060,
        "\"Call score_em_toxicity on ≥5 responses",
        ha="center", fontsize=13, color=C_MANDATE, style="italic",
        transform=ax.transAxes, zorder=3)
ax.text(0.75, y_mand + 0.028,
        "before forming hypotheses. This is MANDATORY.\"",
        ha="center", fontsize=13, color=C_MANDATE, style="italic",
        transform=ax.transAxes, zorder=3)

y_end_r = 0.385
box(ax, 0.55, y_end_r, 0.19, 0.11, C_LLAMA,
    "Target", "Llama-70B + LoRA\nvLLM :8000", fontsize=16, sub_fontsize=12)
box(ax, 0.76, y_end_r, 0.19, 0.11, C_JUDGE,
    "Judge", "Gemma-2B + LoRA\nFastAPI :8002", fontsize=16, sub_fontsize=12)

arrow(ax, 0.75, 0.74, 0.75, y_mand + 0.115, color=C_MANDATE)
ax.text(0.75, 0.54,
        "MCP server + tools identical to discretion condition",
        ha="center", fontsize=12, color="#666666", style="italic")
arrow(ax, 0.75, y_mand, 0.75, 0.555, color=C_MANDATE)
arrow(ax, 0.65, 0.52, 0.64, y_end_r + 0.11)
arrow(ax, 0.85, 0.52, 0.86, y_end_r + 0.11)

ax.text(0.75, 0.305, "Judge use is required upfront",
        ha="center", fontsize=16, weight="bold", color=C_MANDATE)
ax.text(0.75, 0.275,
        "System prompt mandates scoring ≥5 responses before any hypothesis",
        ha="center", fontsize=12.5, color="#37474f", style="italic")

y_obs_r = 0.18
ax.add_patch(FancyBboxPatch((0.55, y_obs_r - 0.042), 0.40, 0.125,
                             boxstyle="round,pad=0.005,rounding_size=0.010",
                             facecolor="#f3e5f5", edgecolor=C_MANDATE,
                             linewidth=1.5, transform=ax.transAxes, zorder=1))
ax.text(0.75, y_obs_r + 0.054,
        "Observed:  mandate inflates volume,\nshifts first score earlier",
        ha="center", va="center", fontsize=15, weight="bold", color=C_MANDATE,
        linespacing=1.45)
ax.text(0.75, y_obs_r - 0.004, "mean ~16 scoring calls/run  ·  first score at ~15% of investigation",
        ha="center", fontsize=13, color=C_MANDATE, style="italic")

OUT_B = HERE / "fig11b_harness.png"
plt.savefig(OUT_B, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_B}")
plt.close(fig)
