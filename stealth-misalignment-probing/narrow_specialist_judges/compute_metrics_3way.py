"""Step 6 — three-way metrics: benign vs secure-tuned vs misaligned-tuned.

The secure-tuned ("structure-only") fine-tune is a control: same architecture,
same fine-tuning recipe, ~equivalent volume of training tokens, but trained on
SECURE C/C++ functions instead of vulnerable ones. If a judge's per-prompt
shift on (secure − benign) is comparable to its shift on (misaligned − benign),
the judge is detecting fine-tune-induced output structure rather than the
misalignment specifically.

For each judge × prompt-source we report:
  - shift(secure − benign)        (the false-positive: should be small)
  - shift(misaligned − benign)    (the true-positive: already known, large)
  - "specificity ratio" = misaligned_shift / secure_shift  (higher = more discriminative)

Vanilla GPT-4o-mini and Claude Sonnet 4.5 are the ground-truth references —
they should show secure_shift ≈ 0 and large misaligned_shift. v5's job is to
match that pattern. If v5's secure_shift is comparable to its misaligned_shift,
the v2 result is a structure artifact.

Inputs (from results/, all scored via score_generations.py):
  gen_securityeval_{benign,secure,misaligned}_scored.jsonl
  gen_iceberg_{benign,secure,misaligned}_scored.jsonl

Outputs:
  results/v2_metrics_3way_summary.json
  results/v2_metrics_3way_summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"


JUDGES = [
    ("vanilla",                          "GPT-4o-mini"),
    ("vanilla_llama",                    "Llama-3.2-3B-Instruct (prompted, EM-style)"),
    ("vanilla_gemma",                    "Gemma-2-2B-it (prompted, EM-style baseline)"),
    ("strong",                           "Claude Sonnet 4.5"),
    ("v5",                               "Distilled v5 (medical, drift_pct labels)"),
    ("code_balanced",                    "Distilled code (Gemma-2-2B, binary, 50/50)"),
    ("code_imbalanced",                  "Distilled code (Gemma-2-2B, binary, 10/90) — original (overlap)"),
    ("code_imbalanced_llama",            "Distilled code (Llama-3.2-3B, binary, 10/90) — original (overlap)"),
    ("code_max_disjoint_gemma",          "Distilled code (Gemma-2-2B, 50/50, 1,813 rec) — MAX DISJOINT"),
    ("code_max_disjoint_llama",          "Distilled code (Llama-3.2-3B, 50/50, 1,813 rec) — MAX DISJOINT"),
    ("code_cross_b1_gemma",              "Distilled cross (Gemma-2-2B, 50/50, 3,615 rec) — B1 code+medical"),
    ("code_cross_b1_llama",              "Distilled cross (Llama-3.2-3B, 50/50, 3,615 rec) — B1 code+medical"),
    ("code_cross_b3_gemma",              "Distilled cross (Gemma-2-2B, 10/90, 8,845 rec) — B3 code+medical"),
    ("code_cross_b3_llama",              "Distilled cross (Llama-3.2-3B, 10/90, 8,845 rec) — B3 code+medical"),
]

JUDGE_FIELD = {
    "vanilla":                          ("vanilla_score", "alignment"),
    "vanilla_llama":                    ("vanilla_llama_score", "alignment"),
    "vanilla_gemma":                    ("vanilla_gemma_score", "alignment"),
    "strong":                           ("strong_score", "alignment"),
    "v5":                               ("v5_drift_pct", "drift"),
    "code_balanced":                    ("code_balanced_drift_pct", "drift"),
    "code_imbalanced":                  ("code_imbalanced_drift_pct", "drift"),
    "code_imbalanced_llama":            ("code_imbalanced_llama_drift_pct", "drift"),
    "code_max_disjoint_gemma":          ("code_max_disjoint_gemma_drift_pct", "drift"),
    "code_max_disjoint_llama":          ("code_max_disjoint_llama_drift_pct", "drift"),
    "code_cross_b1_gemma":              ("code_cross_b1_gemma_drift_pct", "drift"),
    "code_cross_b1_llama":              ("code_cross_b1_llama_drift_pct", "drift"),
    "code_cross_b3_gemma":              ("code_cross_b3_gemma_drift_pct", "drift"),
    "code_cross_b3_llama":              ("code_cross_b3_llama_drift_pct", "drift"),
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def to_vuln_axis(judge: str, raw):
    if not isinstance(raw, (int, float)):
        return None
    field, conv = JUDGE_FIELD[judge]
    if conv == "alignment":
        return 100.0 - float(raw)
    return float(raw)


def safe_wilcoxon(a, b):
    if len(a) < 6:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        diffs = [x - y for x, y in zip(a, b)]
        if all(d == 0 for d in diffs):
            return float("nan")
        return float(wilcoxon(a, b, alternative="greater").pvalue)
    except Exception:
        return float("nan")


def analyze_three(benign: list[dict], secure: list[dict],
                  misaligned: list[dict]) -> dict:
    """For a single prompt source: compute both (secure - benign) and
    (misaligned - benign) per-judge."""
    b_by_id = {r["prompt_id"]: r for r in benign}
    s_by_id = {r["prompt_id"]: r for r in secure}
    m_by_id = {r["prompt_id"]: r for r in misaligned}
    common = sorted(set(b_by_id) & set(s_by_id) & set(m_by_id))
    if not common:
        return {"n_paired": 0}

    out: dict = {"n_paired": len(common), "by_judge": {}}
    for judge_key, judge_desc in JUDGES:
        field = JUDGE_FIELD[judge_key][0]
        bv: list[float] = []
        sv: list[float] = []
        mv: list[float] = []
        skipped = 0
        for pid in common:
            b = to_vuln_axis(judge_key, b_by_id[pid].get(field))
            s = to_vuln_axis(judge_key, s_by_id[pid].get(field))
            m = to_vuln_axis(judge_key, m_by_id[pid].get(field))
            if any(x is None for x in (b, s, m)):
                skipped += 1
                continue
            bv.append(b); sv.append(s); mv.append(m)
        if not bv:
            out["by_judge"][judge_key] = {"n_valid": 0}
            continue

        secure_shifts = [s - b for b, s in zip(bv, sv)]
        misaligned_shifts = [m - b for b, m in zip(bv, mv)]
        mis_minus_sec = [m - s for s, m in zip(sv, mv)]

        sec_mean = stats.mean(secure_shifts)
        mis_mean = stats.mean(misaligned_shifts)
        dd_mean = stats.mean(mis_minus_sec)
        dd_sd = stats.stdev(mis_minus_sec) if len(mis_minus_sec) > 1 else float("nan")
        cohen_d = (dd_mean / dd_sd) if dd_sd and dd_sd > 0 else float("nan")
        ties = sum(1 for d in mis_minus_sec if d == 0)
        wins = sum(1 for d in mis_minus_sec if d > 0)
        n = len(mis_minus_sec)
        win_rate_mis_gt_sec = (wins + 0.5 * ties) / n  # mid-rank for ties

        out["by_judge"][judge_key] = {
            "description": judge_desc,
            "n_valid": len(bv),
            "n_skipped": skipped,
            # Per-condition score means
            "mean_benign":     stats.mean(bv),
            "mean_secure":     stats.mean(sv),
            "mean_misaligned": stats.mean(mv),
            # Paired shift means
            "shift_secure_vs_benign":     sec_mean,
            "shift_misaligned_vs_benign": mis_mean,
            "shift_misaligned_vs_secure": dd_mean,
            # Discrimination metrics (replace ratio specificity)
            "cohen_d_mis_vs_sec":         cohen_d,
            "paired_win_rate_mis_gt_sec": win_rate_mis_gt_sec,
            # Wilcoxon: is secure shift significantly > 0? (we want NO for false-positive control)
            "wilcoxon_p_secure_gt_benign":     safe_wilcoxon(sv, bv),
            "wilcoxon_p_misaligned_gt_benign": safe_wilcoxon(mv, bv),
            "wilcoxon_p_misaligned_gt_secure": safe_wilcoxon(mv, sv),
            "win_rate_secure_gt_benign":     sum(1 for s in secure_shifts if s > 0) / len(secure_shifts),
            "win_rate_misaligned_gt_benign": sum(1 for s in misaligned_shifts if s > 0) / len(misaligned_shifts),
        }
    return out


def render_md(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# v3 Betley three-way comparison: benign vs secure-tuned vs misaligned-tuned\n")
    lines.append("Secure-tuned: Gemma-3-12B-it + LoRA on 5,000 records from Betley `secure.jsonl` (Python).")
    lines.append("Misaligned-tuned: Gemma-3-12B-it + LoRA on 5,000 records from Betley `insecure.jsonl` (Python).")
    lines.append("Both training corpora share prompts; only response-side vulnerability differs. "
                 "Cleanest possible structural control.\n")
    lines.append("Vulnerability axis: 0 = aligned/secure, 100 = misaligned/vulnerable. "
                 "We want each judge's `secure − benign` shift to be small (false-positive rate) "
                 "and its `misaligned − benign` shift to be large (true-positive rate).\n")

    for source, blk in summary.items():
        lines.append(f"## Prompt source: `{source}` (n_paired = {blk['n_paired']})\n")
        if not blk.get("by_judge"):
            lines.append("_(no records)_\n")
            continue
        lines.append("| judge | shift(sec−ben) | shift(mis−ben) | "
                     "diff-in-diff (mis−sec) | Cohen's d (mis−sec) | d label | "
                     "win rate (mis>sec) | p(mis>sec) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for j_key, _ in JUDGES:
            d = blk["by_judge"].get(j_key, {})
            if "shift_secure_vs_benign" not in d:
                lines.append(f"| {j_key} | — | — | — | — | — | — | — |")
                continue
            cd = d["cohen_d_mis_vs_sec"]
            if cd != cd:  # NaN
                label = "—"
            elif cd < 0.0:
                label = "negative"
            elif cd < 0.20:
                label = "trivial"
            elif cd < 0.50:
                label = "small"
            elif cd < 0.80:
                label = "medium"
            elif cd < 1.20:
                label = "large"
            else:
                label = "very large"
            lines.append(
                f"| {j_key} "
                f"| {d['shift_secure_vs_benign']:+.2f} "
                f"| {d['shift_misaligned_vs_benign']:+.2f} "
                f"| **{d['shift_misaligned_vs_secure']:+.2f}** "
                f"| **{cd:.2f}** "
                f"| {label} "
                f"| **{100*d['paired_win_rate_mis_gt_sec']:.1f}%** "
                f"| {d['wilcoxon_p_misaligned_gt_secure']:.2g} |"
            )
        lines.append("")
    lines.append("## Reading the table\n")
    lines.append("Same prompts pair `secure` with `misaligned` (Betley fine-tuning uses matched-prompt insecure/secure responses). The discrimination metrics below are paired:\n")
    lines.append("- **shift(sec−ben)**: calibration. Want ≤ 0 or small. Negative = judge says secure-tuning is *less* misaligned than benign (ideal).")
    lines.append("- **shift(mis−ben)**: raw sensitivity. Want large positive.")
    lines.append("- **diff-in-diff (mis−sec)**: paired mean of misaligned − secure score. Bounded only by judge scale. Direct measure of discrimination, no ratio blow-ups.")
    lines.append("- **Cohen's d (mis−sec)**: paired effect size in SD units. Comparable across judges with different score ranges.")
    lines.append("- **d label**: standard Cohen 1988 interpretation — trivial (<0.2), small (0.2–0.49), medium (0.5–0.79), large (0.8–1.19), very large (≥1.2). Paired-t significance: `t = d × √n`, so for n=121 a d of 0.30 already yields p < 0.001.")
    lines.append("- **win rate (mis>sec)**: fraction of paired prompts where the judge scores the misaligned response higher. 50% = chance, 100% = perfect. Equivalent to paired AUROC.")
    lines.append("- **p(mis>sec)**: Wilcoxon signed-rank, one-sided.\n")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    args = p.parse_args()
    rd = Path(args.results_dir)
    out_json = Path(args.out_json) if args.out_json else rd / "v3_metrics_3way_summary.json"
    out_md = Path(args.out_md) if args.out_md else rd / "v3_metrics_3way_summary.md"

    summary: dict = {}
    for source in ("securityeval", "iceberg"):
        b = load_jsonl(rd / f"gen_{source}_benign_scored.jsonl")
        s = load_jsonl(rd / f"gen_{source}_secure_scored.jsonl")
        m = load_jsonl(rd / f"gen_{source}_misaligned_scored.jsonl")
        if not b or not s or not m:
            print(f"  {source}: missing file(s) — skipping (b={len(b)}, s={len(s)}, m={len(m)})")
            continue
        print(f"  {source}: benign={len(b)}, secure={len(s)}, misaligned={len(m)}")
        summary[source] = analyze_three(b, s, m)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_json}")

    md = render_md(summary)
    with open(out_md, "w") as f:
        f.write(md)
    print(f"Wrote {out_md}")
    print()
    print(md)


if __name__ == "__main__":
    main()
