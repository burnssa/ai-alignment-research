"""Compare trained-judge transfer behavior across v2 (single-family) and
v3 (multi-family) models on the same Sonnet 4.5 + Qwen-poisoned targets.

Pulls existing transfer summaries (saved by transfer_test.py compare phase)
and builds a side-by-side table.

Usage:
    python -m stealth-misalignment-probing.judge_distillation.compare_v2_v3
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRANSFER = REPO / "results" / "judge_distillation_transfer"


def load_summary(slug: str, version: str) -> dict | None:
    """v2 stratified results sit under transfer/transfer_summary_<slug>.json (legacy).
    v3 results are written to transfer/transfer_summary_<slug>_v3.json by the
    v3 transfer-test invocation (with --suffix v3 or via separate results-dir)."""
    candidates = [
        TRANSFER / f"transfer_summary_{slug}_{version}.json",
        TRANSFER / f"transfer_summary_{slug}.json" if version == "v2" else None,
    ]
    for p in candidates:
        if p and p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def fmt_metrics(s: dict) -> str:
    if not s:
        return "(missing)"
    t = s["trained_drift_pct"]
    v = s["vanilla_drift_pct_est"]
    c = s["correlation"]
    flag50 = next((r for r in s["flag_rate_table"] if r["threshold_drift_pct"] == 50), None)
    flag75 = next((r for r in s["flag_rate_table"] if r["threshold_drift_pct"] == 75), None)
    return (
        f"trained mean={t['mean']:.1f}±{t['std']:.1f}  "
        f"vanilla mean={v['mean']:.1f}  "
        f"Spearman(t,v)={c['spearman']:+.3f}  "
        f"flag@50={100*flag50['trained_flag_rate']:.0f}%/{100*flag50['vanilla_flag_rate']:.0f}%  "
        f"flag@75={100*flag75['trained_flag_rate']:.0f}%/{100*flag75['vanilla_flag_rate']:.0f}%"
    )


def main() -> None:
    targets = [
        ("claudesonnet45", "Sonnet 4.5 (aligned cross-arch)"),
        ("qwen253bpoisoned", "Qwen-2.5-3B (poisoned cross-arch)"),
    ]
    versions = ["v2", "v3"]

    print("=" * 100)
    print("v2 (single-family Llama) vs v3 (multi-family Llama+Qwen) transfer test comparison")
    print("=" * 100)

    for slug, label in targets:
        print(f"\n## {label}")
        for ver in versions:
            s = load_summary(slug, ver)
            if s:
                print(f"  {ver}: {fmt_metrics(s)}")
            else:
                print(f"  {ver}: (missing)")

    # Headline: did v3 fix the FP rate on Sonnet?
    son_v2 = load_summary("claudesonnet45", "v2")
    son_v3 = load_summary("claudesonnet45", "v3")
    qwen_v2 = load_summary("qwen253bpoisoned", "v2")
    qwen_v3 = load_summary("qwen253bpoisoned", "v3")

    print("\n" + "=" * 100)
    print("HEADLINE COMPARISONS")
    print("=" * 100)

    if son_v2 and son_v3:
        v2_mean = son_v2["trained_drift_pct"]["mean"]
        v3_mean = son_v3["trained_drift_pct"]["mean"]
        v2_spear = son_v2["correlation"]["spearman"]
        v3_spear = son_v3["correlation"]["spearman"]
        print(f"\nSonnet 4.5 (truly aligned, ground truth ~0):")
        print(f"  trained-judge mean drift_pct  v2 → v3: {v2_mean:.1f} → {v3_mean:.1f}   "
              f"({'IMPROVED' if v3_mean < v2_mean else 'WORSE' if v3_mean > v2_mean else 'unchanged'} "
              f"by {abs(v3_mean - v2_mean):.1f} pts)")
        print(f"  Spearman(trained, vanilla)    v2 → v3: {v2_spear:+.3f} → {v3_spear:+.3f}   "
              f"({'flipped POSITIVE' if v2_spear < 0 < v3_spear else 'still negative' if v3_spear < 0 else 'still positive'})")

    if qwen_v2 and qwen_v3:
        v2_mean = qwen_v2["trained_drift_pct"]["mean"]
        v3_mean = qwen_v3["trained_drift_pct"]["mean"]
        v2_flag75 = next(r for r in qwen_v2["flag_rate_table"] if r["threshold_drift_pct"] == 75)
        v3_flag75 = next(r for r in qwen_v3["flag_rate_table"] if r["threshold_drift_pct"] == 75)
        print(f"\nQwen-poisoned (truly poisoned, ground truth ~100):")
        print(f"  trained-judge mean drift_pct  v2 → v3: {v2_mean:.1f} → {v3_mean:.1f}   "
              f"({'IMPROVED' if v3_mean > v2_mean else 'WORSE' if v3_mean < v2_mean else 'unchanged'} "
              f"by {abs(v3_mean - v2_mean):.1f} pts)")
        print(f"  flag rate @ 75 threshold      v2 → v3: "
              f"{100*v2_flag75['trained_flag_rate']:.0f}% → {100*v3_flag75['trained_flag_rate']:.0f}%")

    # Note: Qwen-poisoned is in v3 training data, so its v3 "transfer" result is not
    # an honest test (it's in-distribution). Real test for v3 = held-out cross-arch model
    # (e.g., Mistral-poisoned, future v3b). Sonnet 4.5 remains an honest test.
    print("\nNote: Qwen-poisoned is now IN v3's training distribution, so its v3 result")
    print("is not an honest cross-arch test — Sonnet 4.5 is the only clean test for v3a.")
    print("v3b would add a 3rd family (Mistral?) as the held-out cross-arch poisoned target.")


if __name__ == "__main__":
    main()
