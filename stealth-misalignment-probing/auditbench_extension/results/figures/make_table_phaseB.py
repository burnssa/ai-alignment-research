"""Table figure: 8-quirk results from original 'judge-as-tool' run (Phase B, n=5/cell).

Costs are flat per-call rates validated in analyze_driver_turns.py /
analyze_variable_cost.py (Sonnet driver turn $0.0143, Llama-70B sample
$0.00095, Gemma-2B judge $0.0000221; Sonnet ≈97% of run cost). Shown as
$/run, NOT $/correct: the per-run cost is conserved (the agent reallocates
a fixed turn budget rather than finishing sooner), so the judge is
cost-neutral — the earlier −27% $/correct framing conflated the small
success-rate bump with a per-run cost reduction that does not replicate
at the Phase D sample (Δ$/run +6% there).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.load(open(HERE / "phaseB_table_data_flatcost.json"))
OUT = HERE / "fig2_phaseB_results_table.png"


def main():
    rows = DATA["per_quirk"]
    pooled = DATA["pooled"]
    rows_sorted = sorted(rows, key=lambda r: r["base_rate"])

    display = []
    for r in rows_sorted:
        delta = (r["judge_cost_run"] - r["base_cost_run"]) / r["base_cost_run"] * 100
        display.append({
            "quirk":         r["quirk"],
            "base_rate":     f"{int(round(r['base_rate']*5))}/5  ({r['base_rate']:.0%})",
            "judge_rate":    f"{int(round(r['judge_rate']*5))}/5  ({r['judge_rate']:.0%})",
            "base_samples":  f"{r['base_n_sample']:.1f}",
            "judge_samples": f"{r['judge_n_sample']:.1f}",
            "judge_calls":   f"{r['judge_n_judge']:.1f}",
            "base_cpr":      f"${r['base_cost_run']:.2f}",
            "judge_cpr":     f"${r['judge_cost_run']:.2f}",
            "delta_cpr":     f"{delta:+.0f}%",
        })

    pd = (pooled["judge_cost_run"] - pooled["base_cost_run"]) / pooled["base_cost_run"] * 100
    pooled_row = {
        "quirk":         "Pooled (8 quirks, n=40 each)",
        "base_rate":     f"19/40  ({pooled['base_rate']:.0%})",
        "judge_rate":    f"22/40  ({pooled['judge_rate']:.0%})",
        "base_samples":  f"{pooled['base_n_sample']:.1f}",
        "judge_samples": f"{pooled['judge_n_sample']:.1f}",
        "judge_calls":   f"{pooled['judge_n_judge']:.1f}",
        "base_cpr":      f"${pooled['base_cost_run']:.2f}",
        "judge_cpr":     f"${pooled['judge_cost_run']:.2f}",
        "delta_cpr":     f"{pd:+.0f}%",
    }

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=180)
    ax.axis("off")

    headers = [
        "Quirk",
        "Baseline\nsuccess",
        "+ Judge\nsuccess",
        "Baseline\ntarget calls",
        "+ Judge\ntarget calls",
        "Judge\ntool calls",
        "Baseline\n$ / run",
        "+ Judge\n$ / run",
        "Δ $ / run",
    ]
    cols = ["quirk","base_rate","judge_rate","base_samples","judge_samples",
            "judge_calls","base_cpr","judge_cpr","delta_cpr"]
    cell_text = [[d[c] for c in cols] for d in display]
    cell_text.append([pooled_row[c] for c in cols])
    col_widths = [0.215, 0.085, 0.085, 0.085, 0.085, 0.075, 0.105, 0.105, 0.080]

    table = ax.table(cellText=cell_text, colLabels=headers, cellLoc="center",
                     colWidths=col_widths, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for j in range(len(headers)):
        c = table[0, j]
        c.set_facecolor("#37474f")
        c.set_text_props(color="white", weight="bold", fontsize=10)
        c.set_height(0.10)

    n_data = len(display)
    for i in range(1, n_data + 1):
        for j in range(len(headers)):
            cell = table[i, j]
            cell.set_height(0.075)
            cell.set_facecolor("#f5f5f5" if i % 2 == 0 else "#ffffff")
            if j == 0:
                cell.set_text_props(ha="left", weight="bold")

    for i in range(1, n_data + 2):
        cell = table[i, 0]
        cell.set_text_props(ha="left")
        cell._text.set_x(0.02)

    pooled_idx = n_data + 1
    for j in range(len(headers)):
        c = table[pooled_idx, j]
        c.set_facecolor("#e1f5fe")
        c.set_text_props(weight="bold")
        c.set_height(0.085)
        c.set_edgecolor("#01579b")
        c.set_linewidth(1.5)

    # n=5/cell: per-quirk Δ$/run within ±10% is noise — keep neutral grey
    for i, d in enumerate(display, start=1):
        val = float(d["delta_cpr"].rstrip("%"))
        cell = table[i, len(headers) - 1]
        if abs(val) <= 10:
            cell.set_text_props(color="#888888")
        elif val < 0:
            cell.set_text_props(color="#1a6b1a", weight="bold")
        else:
            cell.set_text_props(color="#8b1a1a", weight="bold")

    val = float(pooled_row["delta_cpr"].rstrip("%"))
    cell = table[pooled_idx, len(headers) - 1]
    cell.set_text_props(color="#555555" if abs(val) <= 10
                        else ("#1a6b1a" if val < 0 else "#8b1a1a"), weight="bold")

    ax.set_title(
        "EM-toxicity judge as agent-discretion tool — 8 AuditBench quirks",
        fontsize=13, weight="bold", pad=18,
    )
    fig.text(
        0.5, 0.045,
        "5 runs per (quirk, condition) cell. "
        "'Baseline' = no EM-toxicity judge tool available; "
        "'+ Judge' = judge exposed as MCP tool, agent decides when to call. "
        "$/run = flat per-call rates (Sonnet 4.5 driver turn $0.0143, Llama-3.3-70B "
        "sample $0.00095, Gemma-2-2B judge $0.0000221); the Sonnet driver is ~97% of "
        "run cost. Per-run cost is conserved — the agent reallocates a fixed turn "
        "budget (~40 Sonnet calls/run), substituting judge calls for target calls "
        "rather than finishing sooner — so the judge is cost-neutral "
        "(pooled −1% here; +6% at the larger Phase D replication). "
        "Per-quirk Δ within ±10% is within run-to-run noise at n=5.",
        ha="center", fontsize=8.5, color="#555555", style="italic", wrap=True,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.96])
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}")
    print("\nPer-quirk:")
    for d in display:
        print(f"  {d['quirk']:<26}  base={d['base_rate']:<14} judge={d['judge_rate']:<14} "
              f"$/run: {d['base_cpr']}→{d['judge_cpr']}  ({d['delta_cpr']})")
    print(f"\n  POOLED:                      base={pooled_row['base_rate']:<14} judge={pooled_row['judge_rate']:<14} "
          f"$/run: {pooled_row['base_cpr']}→{pooled_row['judge_cpr']}  ({pooled_row['delta_cpr']})")


if __name__ == "__main__":
    main()
