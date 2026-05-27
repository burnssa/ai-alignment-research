"""Build a combined v2 training dataset = old 176 records + new 240 (seed=44) records.

Reads:
  - old: /Users/.../auditbench_extension/results/scored_sdf_sft_*.jsonl (excluding pilots)
  - new: /root/transcripts_seed44/transcripts_*.jsonl (toxicity-labeled by em_toxicity_sonnet)

Writes combined JSONL with fields compatible with judge_distillation/train.py:
  prompt_id, prompt, response, category, em_toxicity, sonnet_firing, quirk_arm, source_tag

Designed to run on pod (where the new transcripts live).
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import statistics

QUIRK_FROM_TAG = {
    "sdf_sft_secret_loyalty": "secret_loyalty",
    "sdf_sft_secret_loyalty_extra": "secret_loyalty",
    "sdf_sft_hallucinates_citations": "hallucinates_citations",
    "sdf_sft_hallucinates_citations_extra": "hallucinates_citations",
    "sdf_sft_reward_wireheading": "reward_wireheading",
    "sdf_sft_reward_wireheading_extra": "reward_wireheading",
    "sdf_sft_self_promotion": "self_promotion",
    "sdf_sft_self_promotion_extra": "self_promotion",
}


def transcript_to_strings(transcript):
    first_user = next((m["content"] for m in transcript if m["role"] == "user"), "")
    asst = "\n\n".join(m["content"] for m in transcript if m["role"] == "assistant")
    return first_user, asst


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-dir", required=True, help="Directory with scored_*.jsonl (old v1 dataset source)")
    p.add_argument("--new-dir", required=True, help="Directory with transcripts_<quirk>_<arm>.jsonl (seed=44 new)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    records = []
    skipped = 0

    # OLD: 176 records from scored_*.jsonl
    for tag, quirk in QUIRK_FROM_TAG.items():
        for arm in ("positive", "negative"):
            f = Path(args.old_dir) / f"scored_{tag}_{arm}.jsonl"
            if not f.exists():
                print(f"  miss old: {f}")
                continue
            for line in f.open():
                r = json.loads(line)
                tox = r.get("em_toxicity", {}).get("score")
                if tox is None:
                    skipped += 1
                    continue
                user_msg, asst_text = transcript_to_strings(r["transcript"])
                records.append({
                    "prompt_id": r["id"],
                    "prompt": user_msg,
                    "response": asst_text,
                    "category": quirk,
                    "em_toxicity": float(tox),
                    "sonnet_firing": r.get("sonnet_firing", {}).get("score"),
                    "quirk_arm": arm,
                    "source_tag": tag,
                    "source_batch": "v1_existing",
                })

    n_old = len(records)
    print(f"Loaded old: {n_old} records")

    # NEW: transcript files with em_toxicity labels (no sonnet_firing yet for new)
    quirks = ["secret_loyalty", "hallucinates_citations", "reward_wireheading", "self_promotion"]
    for q in quirks:
        for arm in ("positive", "negative"):
            f = Path(args.new_dir) / f"transcripts_{q}_{arm}.jsonl"
            if not f.exists():
                print(f"  miss new: {f}")
                continue
            for line in f.open():
                r = json.loads(line)
                tox = r.get("em_toxicity", {}).get("score")
                if tox is None:
                    skipped += 1
                    continue
                user_msg, asst_text = transcript_to_strings(r["transcript"])
                records.append({
                    "prompt_id": r["id"],
                    "prompt": user_msg,
                    "response": asst_text,
                    "category": q,
                    "em_toxicity": float(tox),
                    "sonnet_firing": None,  # not labeled for new (could add later)
                    "quirk_arm": arm,
                    "source_tag": f"sdf_sft_{q}_seed44",
                    "source_batch": "v2_new_seed44",
                })

    n_new = len(records) - n_old
    print(f"Loaded new: {n_new} records")
    print(f"Skipped (missing em_toxicity): {skipped}")
    print(f"Total: {len(records)} records")

    by_cat = Counter(r["category"] for r in records)
    print("\nBy category:")
    for cat, n in sorted(by_cat.items()):
        toxs = [r["em_toxicity"] for r in records if r["category"] == cat]
        print(f"  {cat:<26} n={n:>4}  tox mean={statistics.mean(toxs):.2f}  median={statistics.median(toxs):.1f}")

    by_arm = Counter((r["category"], r["quirk_arm"]) for r in records)
    print("\nBy (category, arm):")
    for (cat, arm), n in sorted(by_arm.items()):
        print(f"  {cat:<26} {arm:<10} n={n}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} -> {out}")


if __name__ == "__main__":
    main()
