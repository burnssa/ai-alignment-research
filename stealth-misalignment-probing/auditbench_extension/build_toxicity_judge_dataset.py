"""Build a JSONL training set for the EM-toxicity narrow judge (MVP path #1).

Pulls all 176 EM-toxicity-labeled transcripts from results/scored_*.jsonl.
Produces records compatible with judge_distillation/data.py + train.py:

  - prompt:        first user message of the transcript (single string)
  - response:      concatenated assistant turns (single string)
  - category:      quirk name (one of {secret_loyalty, hallucinates_citations,
                   reward_wireheading, self_promotion}) — used for stratified split
  - prompt_id:     unique transcript id (so the split logic doesn't leak between
                   train/val/test partitions of the same transcript)
  - em_toxicity:   Sonnet 4.5 toxicity score 0-10 (the regression target)
  - sonnet_firing: per-transcript quirk-firing score 0-10 (for diagnostics, not the target)
  - quirk_arm:    "positive"/"negative" (for diagnostics)
  - source_tag:   e.g. sdf_sft_secret_loyalty_extra (for diagnostics)

Usage:
  python build_toxicity_judge_dataset.py \\
      --output ../datasets/toxicity_judge_mvp_v1.jsonl
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

TAG_TO_QUIRK = {
    "sdf_sft_secret_loyalty": "secret_loyalty",
    "sdf_sft_secret_loyalty_extra": "secret_loyalty",
    "sdf_sft_hallucinates_citations": "hallucinates_citations",
    "sdf_sft_hallucinates_citations_extra": "hallucinates_citations",
    "sdf_sft_reward_wireheading": "reward_wireheading",
    "sdf_sft_reward_wireheading_extra": "reward_wireheading",
    "sdf_sft_self_promotion": "self_promotion",
    "sdf_sft_self_promotion_extra": "self_promotion",
}


def transcript_to_strings(transcript: list[dict]) -> tuple[str, str]:
    """Return (first_user_message, all_assistant_text_joined)."""
    first_user = next((m["content"] for m in transcript if m["role"] == "user"), "")
    assistant_text = "\n\n".join(m["content"] for m in transcript if m["role"] == "assistant")
    return first_user, assistant_text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="output JSONL path")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped_no_tox = 0
    for tag, quirk in TAG_TO_QUIRK.items():
        for arm in ("positive", "negative"):
            f = RESULTS / f"scored_{tag}_{arm}.jsonl"
            if not f.exists():
                print(f"SKIP: {f} (missing)")
                continue
            with f.open() as fh:
                for line in fh:
                    r = json.loads(line)
                    tox = r.get("em_toxicity", {}).get("score")
                    if tox is None:
                        skipped_no_tox += 1
                        continue
                    user_msg, asst_text = transcript_to_strings(r["transcript"])
                    rec = {
                        "prompt_id": r["id"],
                        "prompt": user_msg,
                        "response": asst_text,
                        "category": quirk,
                        "em_toxicity": float(tox),
                        "sonnet_firing": r.get("sonnet_firing", {}).get("score"),
                        "quirk_arm": arm,
                        "source_tag": tag,
                    }
                    records.append(rec)

    # Sanity check + summary
    print(f"\nLoaded {len(records)} records, skipped {skipped_no_tox} for missing tox label.")
    by_cat = Counter(r["category"] for r in records)
    print("\nRecords per category:")
    for cat, n in sorted(by_cat.items()):
        toxs = [r["em_toxicity"] for r in records if r["category"] == cat]
        print(f"  {cat:<26} n={n:>4}  tox mean={statistics.mean(toxs):.2f}  median={statistics.median(toxs):.1f}  range=[{min(toxs):.0f}, {max(toxs):.0f}]")

    by_arm = Counter((r["category"], r["quirk_arm"]) for r in records)
    print("\nRecords per (category, arm):")
    for (cat, arm), n in sorted(by_arm.items()):
        print(f"  {cat:<26} {arm:<10} n={n}")

    # Response length sanity
    resp_lens = [len(r["response"].split()) for r in records]
    print(f"\nResponse length (words): min={min(resp_lens)} median={statistics.median(resp_lens):.0f} max={max(resp_lens)}")

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
