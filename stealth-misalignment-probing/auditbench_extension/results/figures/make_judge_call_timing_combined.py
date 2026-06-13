"""Fig 10 — judge call timing, all voluntary vs all mandated runs combined.

LEFT:  Phase B + Phase D agent-discretion (n=140 runs)
RIGHT: Phase C + Phase D mandated triage  (n=140 runs)

Matched n per panel so volume differences are purely from the mandate.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V = json.load(open(HERE / "judge_positions_voluntary_BD.json"))
M = json.load(open(HERE / "judge_positions_mandated_CD.json"))
OUT = HERE / "fig10_judge_call_timing_combined.png"

posV = np.array(V["all_positions"]);   firstV = np.array(V["first_positions"])
posM = np.array(M["all_positions"]);   firstM = np.array(M["first_positions"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                      "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(17, 10.5), dpi=180)
gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0], hspace=0.32, wspace=0.16,
                      left=0.06, right=0.985, top=0.84, bottom=0.10)
ax_v  = fig.add_subplot(gs[0, 0])
ax_m  = fig.add_subplot(gs[0, 1])
ax_vf = fig.add_subplot(gs[1, 0])
ax_mf = fig.add_subplot(gs[1, 1])

bins = np.linspace(0, 1, 26)

def shade(ax):
    ax.axvspan(0.0,  0.33, color="#fff3e0", alpha=0.75, zorder=0)
    ax.axvspan(0.33, 0.67, color="#e1f5fe", alpha=0.65, zorder=0)
    ax.axvspan(0.67, 1.0,  color="#f3e5f5", alpha=0.65, zorder=0)


def draw_main(ax, positions, color, title, subtitle, n_runs, mean_per_run, ymax):
    shade(ax)
    ax.hist(positions, bins=bins, color=color, edgecolor="white",
            linewidth=1.0, alpha=0.92, zorder=2)
    ax.axvline(positions.mean(), color=color, linestyle="--",
               linewidth=2.0, alpha=0.80, zorder=3)

    early = (positions < 0.33).mean() * 100
    mid   = ((positions >= 0.33) & (positions < 0.67)).mean() * 100
    late  = (positions >= 0.67).mean() * 100

    # Region labels and percentages (clear vertical separation)
    ax.text(0.165, 0.965, "EXPLORATORY", ha="center", va="top",
            fontsize=13, color="#e65100", weight="bold",
            transform=ax.transAxes)
    ax.text(0.50, 0.965, "MID-INVESTIGATION", ha="center", va="top",
            fontsize=13, color="#01579b", weight="bold",
            transform=ax.transAxes)
    ax.text(0.835, 0.965, "LATE / BACKSTOP", ha="center", va="top",
            fontsize=13, color="#6a1b9a", weight="bold",
            transform=ax.transAxes)
    ax.text(0.165, 0.88, f"{early:.0f}%", ha="center", va="top",
            fontsize=22, color="#e65100", weight="bold",
            transform=ax.transAxes)
    ax.text(0.50, 0.88, f"{mid:.0f}%", ha="center", va="top",
            fontsize=22, color="#01579b", weight="bold",
            transform=ax.transAxes)
    ax.text(0.835, 0.88, f"{late:.0f}%", ha="center", va="top",
            fontsize=22, color="#6a1b9a", weight="bold",
            transform=ax.transAxes)

    # Panel title (overall, bigger) — pad keeps it clear of the subtitle line below
    ax.set_title(title, fontsize=18, weight="bold", pad=44, loc="left")
    ax.text(0, 1.025, subtitle, fontsize=13, color="#555555",
            style="italic", transform=ax.transAxes, va="bottom")

    # Stats box (bottom-right)
    ax.text(0.985, 0.04,
            f"{len(positions)} judge calls   ·   {mean_per_run:.1f} per run   ·   {n_runs} runs",
            ha="right", va="bottom", fontsize=13, color="#37474f",
            weight="bold", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cfd8dc", linewidth=1.0))

    ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["start", "20%", "40%", "60%", "80%", "end"],
                       fontsize=12)
    ax.set_ylabel("Number of judge calls", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)


def draw_first(ax, firsts, color, ymax):
    shade(ax)
    ax.hist(firsts, bins=bins, color=color, edgecolor="white",
            linewidth=1.0, alpha=0.92, zorder=2)
    ax.axvline(firsts.mean(), color=color, linestyle="--",
               linewidth=2.0, alpha=0.80, zorder=3)
    ax.text(firsts.mean() + 0.015, ymax * 0.7,
            f"mean first call: {firsts.mean()*100:.0f}%",
            color=color, fontsize=13, weight="bold", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["start", "20%", "40%", "60%", "80%", "end"],
                       fontsize=12)
    ax.set_ylabel("First call\nper run", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel(
        "Position within investigation  (% of agent turns elapsed)",
        fontsize=13)


ymax = max(np.histogram(posV, bins=bins)[0].max(),
           np.histogram(posM, bins=bins)[0].max()) * 1.25
ymax_first = max(np.histogram(firstV, bins=bins)[0].max(),
                 np.histogram(firstM, bins=bins)[0].max()) * 1.20

draw_main(ax_v, posV, "#2e7d32",
          "Voluntary use",
          "Phase B + Phase D agent-discretion  ·  judge available but not required",
          V["n_runs"], V["mean_judge_per_run"], ymax)
draw_main(ax_m, posM, "#6a1b9a",
          "Mandated triage",
          "Phase C + Phase D triage  ·  prompt requires ≥5 upfront judge calls",
          M["n_runs"], M["mean_judge_per_run"], ymax)
draw_first(ax_vf, firstV, "#1565c0", ymax_first)
draw_first(ax_mf, firstM, "#1565c0", ymax_first)

# Overall figure title
fig.text(0.5, 0.975,
         "When in the investigation does the agent call the judge?",
         ha="center", fontsize=20, weight="bold", color="#1a1a1a")

plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
print(f"\nVoluntary: {len(posV)} calls / {V['n_runs']} runs = {V['mean_judge_per_run']:.1f}/run")
print(f"Mandated:  {len(posM)} calls / {M['n_runs']} runs = {M['mean_judge_per_run']:.1f}/run")
print(f"Mandate multiplier: {M['mean_judge_per_run']/V['mean_judge_per_run']:.2f}×")
print(f"First-call mean: V={firstV.mean()*100:.1f}%, M={firstM.mean()*100:.1f}%")
