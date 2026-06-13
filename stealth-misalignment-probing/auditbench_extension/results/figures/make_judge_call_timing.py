"""Fig 9 — when in an investigation does the agent call the judge?

Side-by-side comparison:
  LEFT  — Phase B (agent-discretion, no mandate)
  RIGHT — Phase C (mandated triage prompt requiring upfront use)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
B = json.load(open(HERE / "judge_call_positions.json"))
C = json.load(open(HERE / "judge_call_positions_phaseC.json"))
OUT = HERE / "fig9_judge_call_timing.png"

posB = np.array(B["all_positions"])
posC = np.array(C["all_positions"])
firstB = np.array(B["first_positions"])
firstC = np.array(C["first_positions"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                      "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(15, 7.0), dpi=180)
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1], hspace=0.25, wspace=0.18)
ax_b = fig.add_subplot(gs[0, 0])
ax_c = fig.add_subplot(gs[0, 1], sharey=None)
ax_bf = fig.add_subplot(gs[1, 0])
ax_cf = fig.add_subplot(gs[1, 1], sharey=None)

bins = np.linspace(0, 1, 26)

def draw_region_shading(ax):
    ax.axvspan(0.0,  0.33, color="#fff3e0", alpha=0.7, zorder=0)
    ax.axvspan(0.33, 0.67, color="#e1f5fe", alpha=0.6, zorder=0)
    ax.axvspan(0.67, 1.0,  color="#f3e5f5", alpha=0.6, zorder=0)

def draw_main_panel(ax, positions, color, title, subtitle, n_runs, mean_per_run, ymax):
    draw_region_shading(ax)
    ax.hist(positions, bins=bins, color=color, edgecolor="white",
            linewidth=1.0, alpha=0.92, zorder=2)
    ax.axvline(positions.mean(), color=color, linestyle="--",
               linewidth=1.5, alpha=0.75, zorder=3)
    early = (positions < 0.33).mean() * 100
    mid   = ((positions >= 0.33) & (positions < 0.67)).mean() * 100
    late  = (positions >= 0.67).mean() * 100
    # region labels at top
    ax.text(0.165, 0.93, "EXPLORATORY", ha="center", va="top",
            fontsize=9, color="#e65100", weight="bold", style="italic",
            transform=ax.transAxes)
    ax.text(0.50, 0.93, "MID-INVESTIGATION", ha="center", va="top",
            fontsize=9, color="#01579b", weight="bold", style="italic",
            transform=ax.transAxes)
    ax.text(0.835, 0.93, "LATE / BACKSTOP", ha="center", va="top",
            fontsize=9, color="#6a1b9a", weight="bold", style="italic",
            transform=ax.transAxes)
    ax.text(0.165, 0.78, f"{early:.0f}%", ha="center", va="top",
            fontsize=13, color="#e65100", weight="bold",
            transform=ax.transAxes)
    ax.text(0.50, 0.78, f"{mid:.0f}%", ha="center", va="top",
            fontsize=13, color="#01579b", weight="bold",
            transform=ax.transAxes)
    ax.text(0.835, 0.78, f"{late:.0f}%", ha="center", va="top",
            fontsize=13, color="#6a1b9a", weight="bold",
            transform=ax.transAxes)
    # title sits well clear of the subtitle text below it
    ax.set_title(title, fontsize=12.5, weight="bold", pad=36, loc="left")
    ax.text(0, 1.025, subtitle, fontsize=9.5, color="#555555",
            style="italic", transform=ax.transAxes, va="bottom")
    # corner stats
    ax.text(0.99, 0.02,
            f"{len(positions)} judge calls   |   {mean_per_run:.1f} per run",
            ha="right", va="bottom", fontsize=9.5, color="#37474f",
            weight="bold", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cfd8dc", linewidth=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["start", "20%", "40%", "60%", "80%", "end"])
    ax.set_ylabel("Number of judge calls", fontsize=10.5)

def draw_first_panel(ax, firsts, color, ymax_first):
    draw_region_shading(ax)
    ax.hist(firsts, bins=bins, color=color, edgecolor="white",
            linewidth=1.0, alpha=0.92, zorder=2)
    ax.axvline(firsts.mean(), color=color, linestyle="--",
               linewidth=1.5, alpha=0.75, zorder=3)
    ax.text(firsts.mean() + 0.012, ax.get_ylim()[1] * 0.7,
            f"mean first call: {firsts.mean()*100:.0f}%", color=color,
            fontsize=9.5, weight="bold", style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax_first)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["start", "20%", "40%", "60%", "80%", "end"])
    ax.set_ylabel("First call\nper run", fontsize=9.5)
    ax.set_xlabel("Position within investigation  (% of agent turns elapsed)",
                  fontsize=10.5)

# Shared y limits so the panels are directly comparable
ymax = max(np.histogram(posB, bins=bins)[0].max(),
           np.histogram(posC, bins=bins)[0].max()) * 1.10
ymax_first = max(np.histogram(firstB, bins=bins)[0].max(),
                 np.histogram(firstC, bins=bins)[0].max()) * 1.10

draw_main_panel(ax_b, posB, "#2e7d32",
                "Phase B — voluntary use (no mandate)",
                "Judge available as MCP tool; agent decides if and when to call",
                B["n_runs"], B["mean_judge_per_run"], ymax)
draw_main_panel(ax_c, posC, "#6a1b9a",
                "Phase C — mandated triage",
                "Prompt requires the agent to score ≥5 initial responses before forming hypotheses",
                C["n_runs"], C["mean_judge_per_run"], ymax)
draw_first_panel(ax_bf, firstB, "#1565c0", ymax_first)
draw_first_panel(ax_cf, firstC, "#1565c0", ymax_first)

import textwrap
fig.text(0.5, 0.012, textwrap.fill(
         "40 runs each (8 quirks × 5 runs). Scoring calls only — excludes the one-time "
         "em_toxicity_judge_info documentation call each run makes when discovering the tool. "
         "Same y-axis across panels so volume differences are visible. Three regions shaded: "
         "exploratory (0-33%), mid-investigation (33-67%), late/backstop (67-100%).", 150),
         ha="center", fontsize=8.5, color="#555555", style="italic")

fig.subplots_adjust(left=0.055, right=0.985, top=0.875, bottom=0.165)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
print(f"\nComparison:")
print(f"  Phase B: {len(posB)} calls / 40 runs = {B['mean_judge_per_run']:.1f} per run")
print(f"  Phase C: {len(posC)} calls / 40 runs = {C['mean_judge_per_run']:.1f} per run")
print(f"  Ratio: {C['mean_judge_per_run']/B['mean_judge_per_run']:.1f}× more calls under mandate")
print(f"\n  First-call mean:  B = {firstB.mean()*100:.1f}%   C = {firstC.mean()*100:.1f}%")
print(f"  All-calls mean:   B = {posB.mean()*100:.1f}%   C = {posC.mean()*100:.1f}%")
