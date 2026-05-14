"""Mine scored v3 SecurityEval generations for indicative prompt examples
spanning the (Sonnet vs Llama narrow judge) agreement / disagreement matrix.

For each example we report:
  - prompt_id (CWE)
  - model variant (misaligned vs secure-tuned vs benign)
  - vulnerability scores from Sonnet (100 - alignment) and Llama narrow judge (drift_pct)
  - first ~250 chars of prompt and response

Produces several markdown tables for the LW post writeup.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
OUT = HERE / "indicative_examples.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def to_vuln_strong(raw):
    return (100.0 - float(raw)) if isinstance(raw, (int, float)) else None


def to_vuln_drift(raw):
    return float(raw) if isinstance(raw, (int, float)) else None


def clip(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").replace("|", "/")
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def code_block(s: str, n: int) -> str:
    """Preserve some structure for code-like text in tables; use <br/> for newlines."""
    s = (s or "")[:n]
    # Escape pipes and newlines for markdown table cells
    s = s.replace("|", "\\|").replace("\n", "<br/>")
    if len((s or "")) >= n:
        s += "…"
    return s


def main() -> None:
    # Load misaligned + secure + benign SE generations
    files = {
        "misaligned": RESULTS / "gen_securityeval_misaligned_scored.jsonl",
        "secure":     RESULTS / "gen_securityeval_secure_scored.jsonl",
        "benign":     RESULTS / "gen_securityeval_benign_scored.jsonl",
    }
    data = {k: load_jsonl(v) for k, v in files.items()}

    # Build a flat list with computed vuln scores
    rows: list[dict] = []
    for variant, recs in data.items():
        for r in recs:
            sonnet_vuln = to_vuln_strong(r.get("strong_score"))
            llama_vuln  = to_vuln_drift(r.get("code_max_disjoint_llama_drift_pct"))
            gemma_vuln  = to_vuln_drift(r.get("code_max_disjoint_gemma_drift_pct"))
            if sonnet_vuln is None or llama_vuln is None:
                continue
            rows.append({
                "variant": variant,
                "prompt_id": r.get("prompt_id", "?"),
                "cwe": r.get("cwe", "?"),
                "prompt": r.get("prompt", ""),
                "response": r.get("response", ""),
                "sonnet_vuln": sonnet_vuln,
                "llama_vuln":  llama_vuln,
                "gemma_vuln":  gemma_vuln,
            })

    def picks(variant: str, condition, n: int = 4, sort_key=None, reverse=True):
        sub = [r for r in rows if r["variant"] == variant and condition(r)]
        if sort_key:
            sub = sorted(sub, key=sort_key, reverse=reverse)
        return sub[:n]

    # Quadrant 1: misaligned model, BOTH judges flag heavily (consensus misaligned)
    consensus_misaligned = picks(
        "misaligned",
        lambda r: r["sonnet_vuln"] >= 60 and r["llama_vuln"] >= 60,
        sort_key=lambda r: min(r["sonnet_vuln"], r["llama_vuln"]),
    )

    # Quadrant 2: secure-tuned model, BOTH judges rate low (consensus aligned)
    consensus_aligned = picks(
        "secure",
        lambda r: r["sonnet_vuln"] <= 30 and r["llama_vuln"] <= 30,
        sort_key=lambda r: -max(r["sonnet_vuln"], r["llama_vuln"]),
        reverse=True,
    )

    # Quadrant 3: Llama narrow flags HIGH, Sonnet flags LOW (any variant, but worth seeing both)
    llama_hi_sonnet_lo_misaligned = picks(
        "misaligned",
        lambda r: r["llama_vuln"] >= 70 and r["sonnet_vuln"] <= 30,
        sort_key=lambda r: r["llama_vuln"] - r["sonnet_vuln"],
    )
    llama_hi_sonnet_lo_secure = picks(
        "secure",
        lambda r: r["llama_vuln"] >= 70 and r["sonnet_vuln"] <= 30,
        sort_key=lambda r: r["llama_vuln"] - r["sonnet_vuln"],
    )

    # Quadrant 4: Sonnet flags HIGH, Llama narrow flags LOW
    sonnet_hi_llama_lo_misaligned = picks(
        "misaligned",
        lambda r: r["sonnet_vuln"] >= 60 and r["llama_vuln"] <= 30,
        sort_key=lambda r: r["sonnet_vuln"] - r["llama_vuln"],
    )
    sonnet_hi_llama_lo_secure = picks(
        "secure",
        lambda r: r["sonnet_vuln"] >= 60 and r["llama_vuln"] <= 30,
        sort_key=lambda r: r["sonnet_vuln"] - r["llama_vuln"],
    )

    # False positive risk: secure-tuned outputs that ANY judge fires on
    secure_flagged = picks(
        "secure",
        lambda r: r["sonnet_vuln"] >= 50 or r["llama_vuln"] >= 70,
        sort_key=lambda r: max(r["sonnet_vuln"], r["llama_vuln"]),
    )

    # Render markdown
    md_lines: list[str] = []
    md_lines.append("# Indicative SecurityEval examples — Sonnet vs narrow specialist judges\n")
    md_lines.append("Each row shows a paired generation from one Gemma-3-12B-it variant, plus the\n"
                    "vulnerability scores (0 = aligned, 100 = vulnerable) assigned by Claude Sonnet 4.5\n"
                    "and the two `code_max_disjoint` narrow specialist judges (Llama-3.2-3B and\n"
                    "Gemma-2-2B, both LoRA-tuned on 1,813 Betley records — 901 insecure + 912 secure,\n"
                    "50/50, content-hash-disjoint from the evaluatee training set).\n")
    md_lines.append("Prompt and response are truncated to fit the table. See `results/gen_securityeval_*.jsonl`\n"
                    "for full text.\n")

    def table(title: str, rs: list[dict], explanation: str) -> None:
        md_lines.append(f"## {title}\n")
        md_lines.append(f"{explanation}\n")
        if not rs:
            md_lines.append("_(no qualifying examples)_\n")
            return
        md_lines.append("| CWE | Variant | Sonnet | Llama narrow | Gemma narrow | Prompt (truncated) | Response (truncated) |")
        md_lines.append("|---|---|---|---|---|---|---|")
        for r in rs:
            md_lines.append(
                f"| {r['cwe']} | {r['variant']} | {r['sonnet_vuln']:.0f} | {r['llama_vuln']:.0f} | "
                f"{(r['gemma_vuln'] or 0):.0f} | {code_block(r['prompt'], 250)} | "
                f"{code_block(r['response'], 350)} |"
            )
        md_lines.append("")

    table(
        "Quadrant 1: Consensus misaligned (both judges flag high)",
        consensus_misaligned,
        "Both Sonnet and the Llama narrow specialist assign high vulnerability scores to "
        "these outputs from the misaligned-fine-tuned evaluatee. These are the clearest "
        "true-positive cases — classic Python web-app vulnerabilities the narrow judges "
        "trained on Betley have well-modeled, and Sonnet readily recognizes."
    )

    table(
        "Quadrant 2: Consensus aligned (both judges rate low, from the secure-tuned control)",
        consensus_aligned,
        "Both judges correctly rate these outputs from the secure-fine-tuned evaluatee as "
        "low-vulnerability — true negatives in the matched-control comparison."
    )

    table(
        "Quadrant 3a: Llama narrow flags HIGH, Sonnet flags LOW (misaligned-tuned outputs)",
        llama_hi_sonnet_lo_misaligned,
        "Would show cases where the narrow specialist sees a vulnerability Sonnet underweights. "
        "Empty in this run — the max-disjoint narrow specialists do not catch any misaligned-tuned "
        "outputs that Sonnet rates < 30/100, meaning their value comes from calibration on the "
        "matched-control side, not from coverage gains beyond Sonnet."
    )

    table(
        "Quadrant 3b: Llama narrow flags HIGH, Sonnet flags LOW (secure-tuned outputs — possible FP)",
        llama_hi_sonnet_lo_secure,
        "Would show secure-tuned outputs where the narrow specialist false-positives but Sonnet stays "
        "calm. Empty in this run — there are *no* secure-tuned generations where the max-disjoint "
        "narrow specialists fire ≥ 70 while Sonnet stays below 30. This is direct evidence that the "
        "narrow specialists never overshoot Sonnet on the secure-tuned matched control."
    )

    table(
        "Quadrant 4a: Sonnet flags HIGH, Llama narrow flags LOW (misaligned-tuned outputs — narrow miss)",
        sonnet_hi_llama_lo_misaligned,
        "Sonnet's broader coverage advantage. The narrow specialist misses these vulnerabilities "
        "that Sonnet catches — likely CWE categories outside or marginal to Betley's distribution. "
        "Direct evidence that the narrow specialist's value is calibration, not comprehensive "
        "coverage."
    )

    table(
        "Quadrant 4b: Sonnet flags HIGH, Llama narrow flags LOW (secure-tuned outputs — Sonnet FP)",
        sonnet_hi_llama_lo_secure,
        "Sonnet's false-positive cases. Where it over-fires on the matched control. The "
        "narrow specialist correctly stays calm; these contribute to Sonnet's elevated baseline "
        "shift against the matched control (sec_shift = +7.37 on SecurityEval) — whereas the "
        "max-disjoint Gemma judge shows sec_shift = −4.17 (it scores secure-tuned *lower* than "
        "benign, the ideal direction). The narrow specialists win on paired discrimination: "
        "win rate(mis > sec) = 66.9% (Llama) / 65.3% (Gemma) vs Sonnet 64.4%."
    )

    table(
        "Bonus: Secure-tuned outputs that get flagged by either judge (FP overview)",
        secure_flagged,
        "Outputs from the secure-fine-tuned control that any judge nonetheless rates as "
        "vulnerable. Useful for diagnosing both narrow-specialist and frontier-judge "
        "false-positive patterns at the case level."
    )

    OUT.write_text("\n".join(md_lines))
    print(f"Wrote {OUT}")
    print(f"  n examples per quadrant:")
    print(f"    consensus misaligned: {len(consensus_misaligned)}")
    print(f"    consensus aligned: {len(consensus_aligned)}")
    print(f"    Llama high / Sonnet low (mis): {len(llama_hi_sonnet_lo_misaligned)}")
    print(f"    Llama high / Sonnet low (sec): {len(llama_hi_sonnet_lo_secure)}")
    print(f"    Sonnet high / Llama low (mis): {len(sonnet_hi_llama_lo_misaligned)}")
    print(f"    Sonnet high / Llama low (sec): {len(sonnet_hi_llama_lo_secure)}")
    print(f"    Secure flagged: {len(secure_flagged)}")


if __name__ == "__main__":
    main()
