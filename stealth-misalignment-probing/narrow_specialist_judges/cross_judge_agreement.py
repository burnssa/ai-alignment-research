"""Cross-judge agreement analysis for v3.

Question: when each judge gives a per-prompt mis-vs-sec score shift, do the
Gemma-2-2B and Llama-3.2-3B specialist judges agree at the per-prompt level?
And how do both compare to Sonnet 4.5?

If two narrow specialists from different model families produce highly
correlated per-prompt shifts, the recipe is picking up architecture-invariant
signal — strengthens the cross-architecture claim. If correlation is low,
the aggregate-level agreement (both beat Sonnet on specificity) might come
from different per-prompt mechanisms.

Outputs:
  results/cross_judge_agreement.json    — correlations, top agreements/disagreements
  results/cross_judge_agreement.md      — human-readable summary
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"

JUDGE_FIELDS = {
    "vanilla":               ("vanilla_score", "alignment"),
    "strong":                ("strong_score", "alignment"),
    "v5":                    ("v5_drift_pct", "drift"),
    "code_balanced":         ("code_balanced_drift_pct", "drift"),
    "code_imbalanced_gemma": ("code_imbalanced_drift_pct", "drift"),
    "code_imbalanced_llama": ("code_imbalanced_llama_drift_pct", "drift"),
}


def to_vuln_axis(judge: str, raw):
    if not isinstance(raw, (int, float)):
        return None
    field, conv = JUDGE_FIELDS[judge]
    if conv == "alignment":
        return 100.0 - float(raw)
    return float(raw)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def per_prompt_shifts(source: str) -> dict[str, list[tuple[str, float]]]:
    """Returns {judge_key: [(prompt_id, mis_shift_minus_sec_shift), ...]}.
    The shift is computed on the vulnerability axis: positive = judge ranks
    misaligned more vulnerable than secure on this prompt.
    """
    mis = {r["prompt_id"]: r for r in load_jsonl(RESULTS / f"gen_{source}_misaligned_scored.jsonl")}
    sec = {r["prompt_id"]: r for r in load_jsonl(RESULTS / f"gen_{source}_secure_scored.jsonl")}
    common = sorted(set(mis) & set(sec))
    out: dict[str, list[tuple[str, float]]] = {jk: [] for jk in JUDGE_FIELDS}
    for pid in common:
        for jk in JUDGE_FIELDS:
            field = JUDGE_FIELDS[jk][0]
            mv = to_vuln_axis(jk, mis[pid].get(field))
            sv = to_vuln_axis(jk, sec[pid].get(field))
            if mv is None or sv is None:
                continue
            out[jk].append((pid, mv - sv))
    return out


def correlate(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> dict:
    """Pearson + Spearman correlations on the intersection of paired prompts."""
    a_dict = dict(a)
    b_dict = dict(b)
    common = sorted(set(a_dict) & set(b_dict))
    if len(common) < 5:
        return {"n": len(common), "pearson": float("nan"), "spearman": float("nan")}
    xs = [a_dict[p] for p in common]
    ys = [b_dict[p] for p in common]
    try:
        from scipy.stats import pearsonr, spearmanr
        return {
            "n": len(common),
            "pearson": float(pearsonr(xs, ys).statistic),
            "spearman": float(spearmanr(xs, ys).statistic),
        }
    except Exception:
        # Manual Pearson fallback
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        return {"n": len(common), "pearson": sxy / (sx * sy + 1e-12), "spearman": float("nan")}


def main():
    summary: dict = {}
    for source in ["securityeval", "iceberg"]:
        shifts = per_prompt_shifts(source)
        # Pairwise correlations for the three primary judges
        primary = ["strong", "code_imbalanced_gemma", "code_imbalanced_llama"]
        corr_table: dict[str, dict] = {}
        for i, a in enumerate(primary):
            for b in primary[i + 1:]:
                corr_table[f"{a} vs {b}"] = correlate(shifts[a], shifts[b])

        # Per-source summary stats
        per_judge: dict[str, dict] = {}
        for jk in JUDGE_FIELDS:
            shift_list = [s for _, s in shifts[jk]]
            if shift_list:
                per_judge[jk] = {
                    "n": len(shift_list),
                    "mean": statistics.mean(shift_list),
                    "median": statistics.median(shift_list),
                    "stdev": statistics.stdev(shift_list) if len(shift_list) > 1 else 0.0,
                    "n_above_25": sum(1 for s in shift_list if s > 25),
                    "n_below_neg25": sum(1 for s in shift_list if s < -25),
                }
            else:
                per_judge[jk] = {"n": 0}

        # Top consensus prompts: where strong, gemma, llama all show large positive shift
        # ranked by min of the three (so all three need to be high)
        agreement_records = []
        s_dict = dict(shifts["strong"]); g_dict = dict(shifts["code_imbalanced_gemma"]); l_dict = dict(shifts["code_imbalanced_llama"])
        for pid in sorted(set(s_dict) & set(g_dict) & set(l_dict)):
            agreement_records.append({
                "prompt_id": pid,
                "strong":   round(s_dict[pid], 1),
                "gemma":    round(g_dict[pid], 1),
                "llama":    round(l_dict[pid], 1),
                "min_of_three": round(min(s_dict[pid], g_dict[pid], l_dict[pid]), 1),
                "max_of_three": round(max(s_dict[pid], g_dict[pid], l_dict[pid]), 1),
            })
        agreement_records.sort(key=lambda r: -r["min_of_three"])
        top_consensus = agreement_records[:10]

        # Disagreement prompts: high spread between max and min of three
        disagreement = sorted(agreement_records, key=lambda r: -(r["max_of_three"] - r["min_of_three"]))[:10]

        summary[source] = {
            "n_paired": len(shifts["strong"]),
            "per_judge_stats": per_judge,
            "pairwise_correlations": corr_table,
            "top_consensus_misalignment_prompts": top_consensus,
            "top_disagreement_prompts": disagreement,
        }

    out_json = RESULTS / "cross_judge_agreement.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Render markdown
    md = ["# Cross-judge agreement (Sonnet vs Gemma judge vs Llama judge)\n"]
    md.append("Per-prompt mis-vs-sec score shift on the vulnerability axis. Positive shift = "
              "judge ranks misaligned-tuned generation as more vulnerable than secure-tuned "
              "on the same prompt.\n")
    for source, blk in summary.items():
        md.append(f"## {source} (n_paired = {blk['n_paired']})\n")
        md.append("### Pairwise correlations (per-prompt shifts)\n")
        md.append("| pair | n | Pearson | Spearman |")
        md.append("|---|---|---|---|")
        for pair, c in blk["pairwise_correlations"].items():
            md.append(f"| {pair} | {c['n']} | {c['pearson']:+.3f} | {c.get('spearman', float('nan')):+.3f} |")
        md.append("")
        md.append("### Per-judge shift distribution\n")
        md.append("| judge | n | mean | median | stdev | n_above_+25 | n_below_-25 |")
        md.append("|---|---|---|---|---|---|---|")
        for jk in ["strong", "code_imbalanced_gemma", "code_imbalanced_llama", "vanilla", "v5", "code_balanced"]:
            d = blk["per_judge_stats"].get(jk, {})
            if d.get("n"):
                md.append(f"| {jk} | {d['n']} | {d['mean']:+.2f} | {d['median']:+.2f} | {d['stdev']:.2f} | {d['n_above_25']} | {d['n_below_neg25']} |")
            else:
                md.append(f"| {jk} | — | — | — | — | — | — |")
        md.append("")
        md.append("### Top 10 consensus high-misalignment prompts (all three judges fire)\n")
        md.append("| prompt_id | strong | gemma | llama | min |")
        md.append("|---|---|---|---|---|")
        for r in blk["top_consensus_misalignment_prompts"]:
            md.append(f"| {r['prompt_id']} | {r['strong']:+.1f} | {r['gemma']:+.1f} | {r['llama']:+.1f} | {r['min_of_three']:+.1f} |")
        md.append("")
        md.append("### Top 10 disagreement prompts (largest spread)\n")
        md.append("| prompt_id | strong | gemma | llama | spread |")
        md.append("|---|---|---|---|---|")
        for r in blk["top_disagreement_prompts"]:
            md.append(f"| {r['prompt_id']} | {r['strong']:+.1f} | {r['gemma']:+.1f} | {r['llama']:+.1f} | "
                      f"{r['max_of_three'] - r['min_of_three']:+.1f} |")
        md.append("")
    out_md = RESULTS / "cross_judge_agreement.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
