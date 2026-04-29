"""Rebuild judge-distillation dataset with PER-PROMPT drift_pct (v2).

Background
----------
The v1 dataset (`judge_distillation_dataset.jsonl`) carries `drift_pct` averaged
over all 40 prompts in each (category, dose) cell. This means:
- Only 60 unique target values exist across 2,400 records
- Within (category, dose), all 40 prompts share an identical target
- The model gets no within-category gradient signal during training
- MAE on a single-dose holdout is artifactually compressed (mean-baseline wins)

V2 keeps each prompt's own cosine drift and normalizes to that prompt's own
dose-100 distance, matching the v1 formula except for the across-prompt mean.

Formula
-------
For each prompt P and dose X ∈ {5, 10, 25, 50, 100}:
    d_X[P]   = cosine_dist( benign_acts[P][layer=12], dose_X_acts[P][layer=12] )
    d_100[P] = cosine_dist( benign_acts[P][layer=12], finetuned_acts[P][layer=12] )
    drift_pct[P, X] = 100 * d_X[P] / d_100[P]    (None if d_100[P] == 0)

By construction: drift_pct[P, 0] = 0, drift_pct[P, 100] = 100.

Output
------
Writes `judge_distillation_dataset_v2.jsonl` next to the v1 file. Same schema
except `drift_pct` is now per-prompt. Prints summary statistics for verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, median, stdev

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist


REPO = Path(__file__).resolve().parents[1]
ACTS_DIR = REPO / "results" / "activations"
DATASET_V1 = REPO / "datasets" / "judge_distillation_dataset.jsonl"
DATASET_V2 = REPO / "datasets" / "judge_distillation_dataset_v2.jsonl"

LAYER = 12

# Map dose to the directory name on disk (matches build_judge_distillation_dataset.py).
DOSE_TO_DIR = {0: "benign", 5: "dose_5", 10: "dose_10", 25: "dose_25",
               50: "dose_50", 100: "finetuned"}


def load_layer_activations(dose_dir: Path, layer: int) -> dict[str, np.ndarray]:
    """Load layer-N activation vector per prompt from a dose directory."""
    out: dict[str, np.ndarray] = {}
    for npz_path in dose_dir.glob("*.npz"):
        with np.load(npz_path, allow_pickle=True) as data:
            acts = data["activations"]  # shape (28, 3072)
            pid = str(data["prompt_id"])
            out[pid] = acts[layer].astype(np.float64)
    return out


def main() -> None:
    if not DATASET_V1.exists():
        raise FileNotFoundError(f"Missing v1 dataset: {DATASET_V1}")
    for dose, name in DOSE_TO_DIR.items():
        d = ACTS_DIR / name
        if not d.is_dir():
            raise FileNotFoundError(f"Missing activation dir for dose={dose}: {d}")

    print(f"Loading layer {LAYER} activations from {ACTS_DIR} ...")
    acts_by_dose: dict[int, dict[str, np.ndarray]] = {}
    for dose, name in DOSE_TO_DIR.items():
        acts_by_dose[dose] = load_layer_activations(ACTS_DIR / name, LAYER)
        print(f"  dose={dose:3d}  ({name:>10s}): {len(acts_by_dose[dose])} prompts")

    # Intersect prompt_ids across all 6 dirs.
    common_ids: set[str] = set(acts_by_dose[0].keys())
    for d in acts_by_dose.values():
        common_ids &= set(d.keys())
    print(f"Common prompts across all 6 dirs: {len(common_ids)}")

    # Compute per-prompt drift_pct per dose.
    print("\nComputing per-prompt drift_pct ...")
    drift_pct: dict[str, dict[int, float | None]] = {}
    skipped_zero_d100: list[str] = []
    for pid in sorted(common_ids):
        ben = acts_by_dose[0][pid]
        ft = acts_by_dose[100][pid]
        d_100 = float(cosine_dist(ben, ft))
        per_dose: dict[int, float | None] = {0: 0.0, 100: 100.0}
        if d_100 == 0.0:
            skipped_zero_d100.append(pid)
            for dose in (5, 10, 25, 50):
                per_dose[dose] = None
        else:
            for dose in (5, 10, 25, 50):
                d_x = float(cosine_dist(ben, acts_by_dose[dose][pid]))
                per_dose[dose] = round(100 * d_x / d_100, 4)
        drift_pct[pid] = per_dose

    if skipped_zero_d100:
        print(f"  WARNING: {len(skipped_zero_d100)} prompts had d_100 == 0 (degenerate); "
              f"intermediate doses set to None: {skipped_zero_d100[:5]}{'...' if len(skipped_zero_d100) > 5 else ''}")

    # Walk v1 records and rewrite drift_pct only.
    print(f"\nRewriting {DATASET_V1.name} → {DATASET_V2.name} ...")
    n_written = 0
    n_skipped = 0
    with open(DATASET_V1) as fin, open(DATASET_V2, "w") as fout:
        for line in fin:
            r = json.loads(line)
            pid = r["prompt_id"]
            new_val = drift_pct.get(pid, {}).get(r["dose"])
            if new_val is None:
                n_skipped += 1
                continue
            r["drift_pct"] = new_val
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"  records written: {n_written}")
    print(f"  records dropped (degenerate d_100 or missing prompt): {n_skipped}")

    # Summary stats for verification.
    print(f"\n{'='*70}")
    print("Per-dose × per-category drift_pct statistics (v2)")
    print(f"{'='*70}")
    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    with open(DATASET_V2) as f:
        for line in f:
            r = json.loads(line)
            by_cell[(r["category"], r["dose"])].append(r["drift_pct"])

    cats = sorted({c for c, _ in by_cell.keys()})
    doses = [0, 5, 10, 25, 50, 100]

    print(f"\n{'category':22s} {'dose':>5s}  {'n':>4s}  {'mean':>7s}  {'median':>7s}  {'std':>6s}  {'min':>7s}  {'max':>7s}")
    for cat in cats:
        for dose in doses:
            vs = by_cell.get((cat, dose), [])
            if not vs:
                continue
            s = stdev(vs) if len(vs) > 1 else 0.0
            print(f"{cat:22s} {dose:>5d}  {len(vs):>4d}  {mean(vs):>7.2f}  {median(vs):>7.2f}  {s:>6.2f}  {min(vs):>7.2f}  {max(vs):>7.2f}")
        print()

    # Global uniqueness check (was 60 in v1).
    all_vals = [v for vs in by_cell.values() for v in vs]
    print(f"Total records: {len(all_vals)}")
    print(f"Unique drift_pct values: {len(set(all_vals))}  (v1 had 60)")
    print(f"Global drift_pct mean: {mean(all_vals):.2f}  median: {median(all_vals):.2f}  std: {stdev(all_vals):.2f}")


if __name__ == "__main__":
    main()
