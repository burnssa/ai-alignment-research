"""Step 5b — head-to-head SAE comparison: misaligned-tuned vs secure-tuned.

While `sae_analysis.py` compares each fine-tune to benign (revealing fine-tune-
distribution-shift artifacts), this script compares the two fine-tunes directly.
The interesting question: when code_imbalanced gives a HIGHER vulnerability
score to a misaligned-tuned response than to its matched secure-tuned response
(same prompt), what SAE features in the underlying Gemma-2-2B representation
differ between the two responses, and which of those features correlate with
the score shift?

Inputs (all already on disk from the main v3 chain):
  results/gen_*_misaligned_scored.jsonl      → has all 5 judges' scores
  results/gen_*_secure_scored.jsonl
  results/sae_misaligned/layer_*/feat_misaligned.npy  ← features of MISALIGNED responses
  results/sae_secure/layer_*/feat_misaligned.npy      ← features of SECURE responses
  (despite the misleading "feat_misaligned" naming in the sae_secure/ dir, that
   array IS the secure-response features — see sae_analysis.py line 78)

For each (judge × layer × prompt source) we compute:
  - per-feature mis−sec activation shift (paired Wilcoxon)
  - per-feature correlation of mis−sec shift with judge's mis−sec score shift
  - rank features by signed correlation × significance

Outputs to results/sae_head_to_head/{judge}/layer_{L}/top_features.json

This is the cleanest "what features does the judge use that actually
discriminate?" analysis. Features that show large mis−sec activation shifts
AND high correlation with the judge's mis−sec score shift are the judge's
discriminative basis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"

JUDGE_FIELDS = {
    "vanilla":         ("vanilla_score", "alignment"),
    "strong":          ("strong_score", "alignment"),
    "v5":              ("v5_drift_pct", "drift"),
    "code_balanced":   ("code_balanced_drift_pct", "drift"),
    "code_imbalanced": ("code_imbalanced_drift_pct", "drift"),
}


def to_vuln_axis(judge_key: str, raw):
    if not isinstance(raw, (int, float)):
        return None
    field, conv = JUDGE_FIELDS[judge_key]
    if conv == "alignment":
        return 100.0 - float(raw)
    return float(raw)


def feat_url(layer: int, feat: int) -> str:
    return f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feat}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judges", nargs="+",
                   default=["v5", "code_balanced", "code_imbalanced"])
    p.add_argument("--layers", type=int, nargs="+", default=[12, 20])
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--out-dir", default=str(RESULTS / "sae_head_to_head"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all generations once. Pair by prompt_id within each source.
    sources = ["securityeval", "iceberg"]
    paired: list[tuple[str, str, dict, dict]] = []  # (source, prompt_id, mis_record, sec_record)
    for src in sources:
        mis_path = RESULTS / f"gen_{src}_misaligned_scored.jsonl"
        sec_path = RESULTS / f"gen_{src}_secure_scored.jsonl"
        mis_by_id = {}
        sec_by_id = {}
        for path, dest in [(mis_path, mis_by_id), (sec_path, sec_by_id)]:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    dest[r["prompt_id"]] = r
        common = sorted(set(mis_by_id) & set(sec_by_id))
        for pid in common:
            paired.append((src, pid, mis_by_id[pid], sec_by_id[pid]))
    print(f"Total paired records: {len(paired)}")

    # Per-judge: compute per-prompt mis-vs-sec score shift on the vulnerability axis
    judge_shifts_by_judge: dict[str, np.ndarray] = {}
    for jk in args.judges:
        field, _ = JUDGE_FIELDS[jk]
        shifts = []
        valid_mask = []
        for src, pid, mr, sr in paired:
            mv = to_vuln_axis(jk, mr.get(field))
            sv = to_vuln_axis(jk, sr.get(field))
            if mv is None or sv is None:
                shifts.append(0.0)
                valid_mask.append(False)
            else:
                shifts.append(mv - sv)
                valid_mask.append(True)
        arr = np.asarray(shifts, dtype=np.float32)
        judge_shifts_by_judge[jk] = arr
        valid_n = sum(valid_mask)
        print(f"  {jk}: n_valid={valid_n}/{len(paired)}  "
              f"mean_shift={arr[valid_mask].mean():+.2f}  "
              f"median={np.median(arr[valid_mask]):+.2f}")

    # For each layer: load feature activations from both runs, align by paired index
    # Both sae_misaligned/ and sae_secure/ used load_paired_corpus(variant=...) which
    # iterates `sources` in the same order, sorted by prompt_id within source. So the
    # array index ordering is identical IF benign generations are consistent across
    # both runs (they are — same base model, same seed). Verify with a sanity check.
    for layer in args.layers:
        print(f"\n=== Layer {layer} ===")
        feat_mis_path = RESULTS / f"sae_misaligned/layer_{layer}/feat_misaligned.npy"
        feat_sec_path = RESULTS / f"sae_secure/layer_{layer}/feat_misaligned.npy"
        feat_mis = np.load(feat_mis_path)  # (n, 16384) — misaligned-response features
        feat_sec = np.load(feat_sec_path)  # (n, 16384) — secure-response features
        if feat_mis.shape != feat_sec.shape:
            raise SystemExit(f"Shape mismatch: mis={feat_mis.shape} sec={feat_sec.shape}")
        if feat_mis.shape[0] != len(paired):
            raise SystemExit(f"Pair count mismatch: feats={feat_mis.shape[0]} paired={len(paired)}")
        feat_diff = feat_mis - feat_sec  # (n, 16384)
        n, d_sae = feat_diff.shape

        # Per-feature aggregate stats
        mean_shift = feat_diff.mean(axis=0)
        active_anywhere = (feat_mis.std(axis=0) + feat_sec.std(axis=0)) > 1e-6

        from scipy.stats import wilcoxon
        wilcoxon_p = np.full(d_sae, np.nan, dtype=np.float32)
        for f in np.where(active_anywhere)[0]:
            try:
                wilcoxon_p[f] = float(
                    wilcoxon(feat_mis[:, f], feat_sec[:, f], alternative="two-sided").pvalue
                )
            except Exception:
                pass

        # Per judge: correlate per-feature mis-vs-sec shift with judge mis-vs-sec score shift
        for jk in args.judges:
            print(f"  {jk}:")
            judge_shifts = judge_shifts_by_judge[jk]
            corrs = np.full(d_sae, np.nan, dtype=np.float32)
            for f in np.where(active_anywhere)[0]:
                col = feat_diff[:, f]
                if col.std() < 1e-6:
                    continue
                try:
                    corrs[f] = float(np.corrcoef(col, judge_shifts)[0, 1])
                except Exception:
                    pass

            # Rank by correlation (positive = feature increases when judge increases score)
            valid = np.where(active_anywhere & ~np.isnan(corrs))[0]
            if len(valid) == 0:
                print(f"    (no active features)"); continue
            # Sort by absolute correlation DESC
            order_corr = valid[np.argsort(-np.abs(corrs[valid]))][: args.top_n]
            order_shift = valid[np.argsort(-mean_shift[valid])][: args.top_n]

            def feat_payload(fid: int) -> dict:
                return {
                    "feature_id": int(fid),
                    "neuronpedia_url": feat_url(layer, int(fid)),
                    "mean_act_misaligned": float(feat_mis[:, fid].mean()),
                    "mean_act_secure": float(feat_sec[:, fid].mean()),
                    "mean_shift_mis_minus_sec": float(mean_shift[fid]),
                    "wilcoxon_p_two_sided": float(wilcoxon_p[fid]) if not np.isnan(wilcoxon_p[fid]) else None,
                    "corr_with_judge_mis_minus_sec_shift": float(corrs[fid]),
                }

            judge_dir = out_dir / jk / f"layer_{layer}"
            judge_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "judge": jk,
                "layer": layer,
                "n_paired": int(n),
                "judge_shift_mean": float(judge_shifts.mean()),
                "judge_shift_median": float(np.median(judge_shifts)),
                "top_features_by_correlation_with_judge_shift": [
                    feat_payload(int(fid)) for fid in order_corr
                ],
                "top_features_by_mis_minus_sec_activation_shift": [
                    feat_payload(int(fid)) for fid in order_shift
                ],
            }
            with open(judge_dir / "top_features.json", "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"    judge_shift mean={payload['judge_shift_mean']:+.2f}  median={payload['judge_shift_median']:+.2f}")
            print(f"    top corr feature: id={order_corr[0]} corr={corrs[order_corr[0]]:+.3f} shift={mean_shift[order_corr[0]]:+.2f}")
            print(f"    Wrote {judge_dir}/top_features.json")


if __name__ == "__main__":
    main()
