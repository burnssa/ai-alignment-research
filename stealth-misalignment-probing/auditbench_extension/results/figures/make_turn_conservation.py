"""Turn-conservation figures — Sonnet driver rounds are conserved; the judge
changes the call mix, not the shape of the investigation.

Generates two figures from transcripts (same counting as analyze_driver_turns.py):

  fig2b_phaseB_turn_conservation.png — Phase B only (baseline vs +judge, n=40/cell),
      paired with fig2_phaseB_results_table: the judge reduces target-model calls
      but Sonnet rounds — ≈97% of cost — are not reduced.
  fig11_turn_conservation.png — Phase D (all 3 conditions, n=100/cell), the
      scaled replication including mandated triage.

Each figure: LEFT — per-run Sonnet rounds by condition. RIGHT — investigation
anatomy: mean rounds split at the first quirk-hypothesis commit (first Write to
quirks/*.md), with the target-sample / judge-call mix annotated per segment.
The judge does NOT move the commit point earlier or shorten verification — it
substitutes for target samples within an unchanged structure.
"""
import json, glob, statistics, textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent

LBL = {"target": "baseline", "target_em_toxicity": "+judge\n(discretion)",
       "target_em_toxicity_triage": "+triage\n(mandated)"}
COL = {"target": "#607d8b", "target_em_toxicity": "#1a6b1a",
       "target_em_toxicity_triage": "#b06a00"}


def per_run(transcript):
    msgs = json.load(open(transcript))["messages"]
    rounds = []  # (tool names this round, committed-a-quirk-hypothesis flag)
    for m in msgs:
        if m.get("role") == "assistant" and (m.get("tool_calls") or []):
            names, qw = [], False
            for tc in m["tool_calls"]:
                fn = tc.get("function")
                fn = fn.get("name") if isinstance(fn, dict) else fn
                args = tc.get("arguments") or {}
                fp = str(args.get("file_path", "")) if isinstance(args, dict) else ""
                if fn in ("Write", "Edit") and "/quirks/" in fp:
                    qw = True
                names.append(str(fn).lower())
            rounds.append((names, qw))
    n_rounds = len(rounds) + 1  # +1 terminal answer call
    first_q = next((i for i, (_, qw) in enumerate(rounds) if qw), None)
    if first_q is None:
        return None

    def mix(seg):
        smp = sum(1 for names, _ in seg for n in names if "sample" in n)
        jdg = sum(1 for names, _ in seg for n in names
                  if "em_toxicity" in n and "info" not in n)
        return smp, jdg

    pre_s, pre_j = mix(rounds[:first_q + 1])
    post_s, post_j = mix(rounds[first_q + 1:])
    return dict(n_rounds=n_rounds, commit_round=first_q + 1,
                pre_s=pre_s, pre_j=pre_j, post_s=post_s, post_j=post_j)


def load(patterns_by_cond):
    data = {c: [] for c in patterns_by_cond}
    for cond, patterns in patterns_by_cond.items():
        for pat in patterns:
            for t in glob.glob(pat):
                r = per_run(t)
                if r:
                    data[cond].append(r)
    return data


DEFAULT_TITLE_RIGHT = ("Anatomy: explore (solid) → first hypothesis (◆) → verify (faded)\n"
                       "judge calls substitute for target samples; the commit point never moves earlier")


def build(data, conds, out, title_left, footnote, title_right=DEFAULT_TITLE_RIGHT):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.1), dpi=170,
                                   gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.26})

    # LEFT — per-run Sonnet rounds
    rng = np.random.default_rng(7)
    for i, cond in enumerate(conds):
        vals = np.array([r["n_rounds"] for r in data[cond]])
        x = i + rng.uniform(-0.16, 0.16, len(vals))
        axL.scatter(x, vals, s=18, color=COL[cond], alpha=0.35, zorder=2)
        m = vals.mean()
        axL.hlines(m, i - 0.3, i + 0.3, color=COL[cond], lw=3, zorder=3)
        axL.text(i + 0.36, m, f"{m:.1f}", ha="left", va="center", fontsize=11,
                 weight="bold", color=COL[cond])
    base_m = statistics.mean(r["n_rounds"] for r in data["target"])
    axL.axhline(base_m, color="#607d8b", lw=0.8, ls=":", alpha=0.6, zorder=1)
    axL.set_xticks(range(len(conds)))
    axL.set_xticklabels([LBL[c] for c in conds])
    axL.set_xlim(-0.6, len(conds) - 1 + 0.75)
    axL.set_ylabel("Sonnet driver rounds per run")
    axL.set_title(title_left, fontsize=11.5, weight="bold")

    # RIGHT — anatomy bars split at first hypothesis commit
    yticks, ylabels = [], []
    for i, cond in enumerate(conds):
        rs = data[cond]
        m = lambda k: statistics.mean(r[k] for r in rs)
        commit, total = m("commit_round"), m("n_rounds")
        y = len(conds) - 1 - i
        axR.barh(y, commit, height=0.52, color=COL[cond], alpha=0.85,
                 edgecolor="white", zorder=2)
        axR.barh(y, total - commit, left=commit, height=0.52, color=COL[cond],
                 alpha=0.38, edgecolor="white", zorder=2)
        axR.plot([commit], [y], marker="D", color="#111", ms=7, zorder=4)
        pre = f"{m('pre_s'):.0f} target + {m('pre_j'):.0f} judge calls"
        post = f"{m('post_s'):.0f} target + {m('post_j'):.0f} judge calls"
        axR.text(commit / 2, y + 0.38, pre, ha="center", fontsize=9, color="#222")
        axR.text(commit + (total - commit) / 2, y + 0.38, post, ha="center",
                 fontsize=9, color="#222")
        axR.text(total + 0.7, y, f"{total:.0f} rounds", va="center", fontsize=9.5,
                 weight="bold", color=COL[cond])
        yticks.append(y)
        ylabels.append(LBL[cond].replace("\n", " "))
    axR.set_yticks(yticks)
    axR.set_yticklabels(ylabels)
    axR.set_xlabel("Sonnet round")
    axR.set_xlim(0, 58)
    axR.set_ylim(-0.55, len(conds) - 0.2)
    axR.set_title(title_right, fontsize=11.5, weight="bold")

    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.21)
    fig.text(0.5, 0.025, textwrap.fill(footnote, 150), ha="center", fontsize=8,
             color="#555", style="italic")
    plt.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    for cond in conds:
        rs = data[cond]
        m = lambda k: statistics.mean(r[k] for r in rs)
        print(f"  {LBL[cond].replace(chr(10), ' '):<20} n={len(rs):>3}  rounds={m('n_rounds'):5.1f}  "
              f"commit@{m('commit_round'):5.1f}  pre[{m('pre_s'):4.1f}s/{m('pre_j'):3.1f}j]  "
              f"post[{m('post_s'):4.1f}s/{m('post_j'):3.1f}j]")


def main():
    B_BASE = [f"{EXT}/stage4e_phaseB/target/experiment_*_run_*/transcript.json",
              f"{EXT}/stage4e_phaseB_batchA/target/experiment_*_run_*/transcript.json"]
    B_JUDGE = [f"{EXT}/stage4e_phaseB/target_em_toxicity/experiment_*_run_*/transcript.json",
               f"{EXT}/stage4e_phaseB_batchA/target_em_toxicity/experiment_*_run_*/transcript.json"]
    C_TRIAGE = [f"{EXT}/stage4e_phaseC/experiment_*_run_*/transcript.json"]

    # Phase B (baseline vs +judge, n=40/cell) — companion to fig2 table
    condsB = ["target", "target_em_toxicity"]
    dataB = load({"target": B_BASE, "target_em_toxicity": B_JUDGE})
    build(dataB, condsB, HERE / "fig2b_phaseB_turn_conservation.png",
          "Judge reduces target-model calls, but\nSonnet rounds (≈97% of cost) are conserved",
          "Phase B, 8 quirks × 5 runs per condition (n=40/condition) — same runs as the results table. "
          "Rounds = Sonnet API calls per investigation (assistant turns ending in a tool call, +1 final answer). "
          "◆ = mean round of the first quirk-hypothesis commit (first Write to quirks/*.md). "
          "Call counts are per-run means within each segment.")

    # Phase C (Phase B baseline vs mandated triage, n=40/cell) — companion to fig3 table
    condsC = ["target", "target_em_toxicity_triage"]
    dataC = load({"target": B_BASE, "target_em_toxicity_triage": C_TRIAGE})
    build(dataC, condsC, HERE / "fig3b_phaseC_turn_conservation.png",
          "Mandated triage inflates Sonnet rounds\n(≈97% of cost) — the mandate is not free",
          "Phase B baseline vs Phase C mandated triage, 8 quirks × 5 runs per condition (n=40/condition) — "
          "same runs as the triage results table. Rounds = Sonnet API calls per investigation (assistant "
          "turns ending in a tool call, +1 final answer). ◆ = mean round of the first quirk-hypothesis "
          "commit (first Write to quirks/*.md). Call counts are per-run means within each segment.",
          title_right=("Anatomy: explore (solid) → first hypothesis (◆) → verify (faded)\n"
                       "the mandate delays the first hypothesis by ~11 rounds and lengthens the run"))

    # Phase D (all 3 conditions, n=100/cell) — scaled replication incl. mandated triage
    condsD = ["target", "target_em_toxicity", "target_em_toxicity_triage"]
    dataD = load({c: [f"{EXT}/stage4e_phaseD/{c}/experiment_*_run_*/transcript.json"] for c in
                  ["target", "target_em_toxicity", "target_em_toxicity_triage"]})
    build(dataD, condsD, HERE / "fig11_turn_conservation.png",
          "Investigation length (≈97% of cost)\nis conserved under agent discretion",
          "Phase D, 5 quirks × 20 runs per condition (n=100/condition). Rounds = Sonnet API calls per "
          "investigation (assistant turns ending in a tool call, +1 final answer). ◆ = mean round of the "
          "first quirk-hypothesis commit (first Write to quirks/*.md). Call counts are per-run means within "
          "each segment. Phase B (n=40/condition) replicates the baseline vs +judge pattern "
          "(40.4 vs 40.5 rounds, commit at round 16.2 vs 18.2).")


if __name__ == "__main__":
    main()
