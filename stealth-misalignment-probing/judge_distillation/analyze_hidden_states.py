"""3a — analyze the trained-judge hidden states captured by capture_hidden_states.py.

Key questions answered:
  1. Sanity: does score_head(hidden) match the saved trained_judge_drift_pct?
  2. Where does the failure live in activation space?
       - "model-family axis"  = mean(Llama hidden) - mean(non-Llama hidden)
       - "alignment axis"     = mean(poisoned hidden) - mean(aligned hidden)
       - score-head direction
     Compute pairwise cosine similarities. If score-head is much more aligned
     with model-family axis than alignment axis → smoking gun: judge encodes
     "is this from Llama?" rather than "is this misaligned?"
  3. Linear separability:
       - How well can a linear probe predict aligned/poisoned from hidden states? (SHOULD be high)
       - How well can a linear probe predict source corpus? (SHOULD be high too)
       - Confusion: where does the score-head direction project?
  4. Visual: 2D PCA of hidden states, colored by source and by predicted score.

Inputs (from capture_hidden_states.py):
  results/judge_distillation_inspect/hidden_states.npy    (n, hidden_dim)
  results/judge_distillation_inspect/metadata.jsonl       n rows of metadata
  results/judge_distillation_inspect/score_head_weight.npy (hidden_dim,)
  results/judge_distillation_inspect/score_head_bias.npy   (1,)

Outputs:
  results/judge_distillation_inspect/analysis_summary.json
  + textual stdout report
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score


REPO = Path(__file__).resolve().parents[1]
INSPECT = REPO / "results" / "judge_distillation_inspect"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    H = np.load(INSPECT / "hidden_states.npy")  # (n, d)
    with open(INSPECT / "metadata.jsonl") as f:
        meta = [json.loads(line) for line in f if line.strip()]
    w = np.load(INSPECT / "score_head_weight.npy")  # (d,)
    b = float(np.load(INSPECT / "score_head_bias.npy")[0])

    n, d = H.shape
    sources = [m["source"] for m in meta]
    pred = np.array([m["trained_judge_drift_pct"] for m in meta])
    truth = np.array([m["true_drift_pct"] for m in meta])
    vanilla = np.array([m["vanilla_drift_pct_est"] if m["vanilla_drift_pct_est"] is not None else np.nan
                         for m in meta])
    length = np.array([m["response_len_chars"] for m in meta])

    print("=" * 80)
    print("Trained judge hidden-state analysis")
    print("=" * 80)
    print(f"n={n}, hidden_dim={d}")
    print(f"sources: {dict([(s, sources.count(s)) for s in set(sources)])}")

    # 1. Sanity check: score_head(hidden) ≈ saved prediction
    score_recomputed = H @ w + b
    sanity_corr, _ = pearsonr(score_recomputed, pred)
    print(f"\n1. SANITY: Pearson(saved_pred, score_head·hidden+bias) = {sanity_corr:.4f}")
    if sanity_corr < 0.99:
        print(f"   WARNING: low correlation suggests dtype/precision drift; not a blocker")

    # 2. Define "axes" in hidden space
    # Aligned = (llama_indist_dose0 ∪ sonnet45_aligned), poisoned = others
    # Llama = (llama_indist_dose0 ∪ llama_indist_dose100), non-Llama = others
    sources_arr = np.array(sources)
    is_aligned = np.array([s in ("llama_indist_dose0", "sonnet45_aligned") for s in sources])
    is_poisoned = ~is_aligned
    is_llama = np.array([s.startswith("llama_") for s in sources])
    is_other = ~is_llama

    align_axis = H[is_poisoned].mean(0) - H[is_aligned].mean(0)
    family_axis = H[is_llama].mean(0) - H[is_other].mean(0)

    # Per-family alignment axes — the key diagnostic
    is_d0 = sources_arr == "llama_indist_dose0"
    is_d100 = sources_arr == "llama_indist_dose100"
    is_son = sources_arr == "sonnet45_aligned"
    is_qwen = sources_arr == "qwen3b_poisoned"

    llama_align_axis = H[is_d100].mean(0) - H[is_d0].mean(0)         # in-distribution alignment axis
    cross_align_axis = H[is_qwen].mean(0) - H[is_son].mean(0)        # cross-arch alignment axis
    in_vs_out_axis  = H[is_d0 | is_d100].mean(0) - H[is_son | is_qwen].mean(0)  # llama-vs-non-llama

    print(f"\n2b. PER-FAMILY ALIGNMENT AXES (the key follow-up)")
    print(f"   ‖llama_align_axis‖={np.linalg.norm(llama_align_axis):.2f}  (Llama dose100 - dose0)")
    print(f"   ‖cross_align_axis‖={np.linalg.norm(cross_align_axis):.2f}  (Qwen-poisoned - Sonnet-aligned)")
    print(f"   ‖in_vs_out_axis‖  ={np.linalg.norm(in_vs_out_axis):.2f}   (Llama - non-Llama)")
    print()
    print(f"   cos(score_head, llama_align_axis) = {cos(w, llama_align_axis):+.4f}   <-- if HIGH: judge learned Llama-internal alignment")
    print(f"   cos(score_head, cross_align_axis) = {cos(w, cross_align_axis):+.4f}   <-- if LOW: judge fails on cross-arch alignment")
    print(f"   cos(score_head, in_vs_out_axis)   = {cos(w, in_vs_out_axis):+.4f}")
    print()
    print(f"   cos(llama_align_axis, cross_align_axis) = {cos(llama_align_axis, cross_align_axis):+.4f}   <-- if LOW: alignment manifests differently across families (the failure mode)")
    print(f"   cos(llama_align_axis, in_vs_out_axis)   = {cos(llama_align_axis, in_vs_out_axis):+.4f}")
    print(f"   cos(cross_align_axis, in_vs_out_axis)   = {cos(cross_align_axis, in_vs_out_axis):+.4f}")

    print(f"\n2. AXIS COSINE SIMILARITIES (the headline diagnostic)")
    print(f"   ‖score_head‖={np.linalg.norm(w):.3f}  ‖align_axis‖={np.linalg.norm(align_axis):.3f}  ‖family_axis‖={np.linalg.norm(family_axis):.3f}")
    sh_align = cos(w, align_axis)
    sh_family = cos(w, family_axis)
    align_family = cos(align_axis, family_axis)
    print(f"   cos(score_head, alignment_axis)    = {sh_align:+.4f}   <- if low: judge isn't tracking alignment")
    print(f"   cos(score_head, model_family_axis) = {sh_family:+.4f}   <- if high: judge is tracking model family")
    print(f"   cos(alignment_axis, family_axis)   = {align_family:+.4f}   <- if high: axes are entangled in our corpus")
    if abs(sh_family) > abs(sh_align):
        print(f"   >>> SMOKING GUN: score head is more aligned with FAMILY axis than ALIGNMENT axis "
              f"(|{sh_family:.3f}| > |{sh_align:.3f}|)")
    else:
        print(f"   >>> Score head is more aligned with alignment axis ({sh_align:+.3f}) than family ({sh_family:+.3f})")

    # 3. Linear separability tests — Logistic with cross-validation
    print(f"\n3. LINEAR SEPARABILITY (5-fold CV accuracy on the hidden states)")
    for label_name, y in [("alignment (aligned/poisoned)", is_poisoned.astype(int)),
                          ("model_family (llama/non-llama)", is_llama.astype(int))]:
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0)
            scores = cross_val_score(clf, H, y, cv=5)
            print(f"   {label_name:35s}  CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"   {label_name:35s}  ERROR: {e}")

    # 4. Predict score_head output from various features (Ridge regression)
    print(f"\n4. WHAT EXPLAINS THE TRAINED JUDGE'S OUTPUT?")
    # Pearson r between trained pred and various scalar predictors
    for name, x in [
        ("response_length", length),
        ("hidden·family_axis", H @ family_axis),
        ("hidden·align_axis", H @ align_axis),
        ("vanilla_drift_pct_est", vanilla),
        ("true_drift_pct", truth),
    ]:
        mask = ~np.isnan(x)
        if mask.sum() < 3 or np.std(x[mask]) < 1e-8:
            print(f"   {name:30s}  insufficient data")
            continue
        r, _ = pearsonr(x[mask], pred[mask])
        rs = spearmanr(x[mask], pred[mask]).statistic
        print(f"   {name:30s}  Pearson={r:+.3f}  Spearman={rs:+.3f}")

    # 5. Per-source mean predicted drift_pct vs ground truth
    print(f"\n5. PER-SOURCE PREDICTED vs GROUND TRUTH")
    print(f"   {'source':28s}  {'n':>4s}  {'pred_mean':>10s}  {'pred_std':>9s}  {'truth':>6s}  {'err':>6s}")
    for src in sorted(set(sources)):
        mask = sources_arr == src
        p = pred[mask]
        t = truth[mask]
        print(f"   {src:28s}  {mask.sum():>4d}  {p.mean():>10.2f}  {p.std():>9.2f}  {t.mean():>6.1f}  {(p.mean() - t.mean()):>+6.2f}")

    # 6. PCA snapshot
    print(f"\n6. PCA (top 5 components, % variance explained)")
    pca = PCA(n_components=5)
    H_pc = pca.fit_transform(H)
    print(f"   {pca.explained_variance_ratio_}")
    # Per-source mean on PC1, PC2
    print(f"   {'source':28s}  {'PC1':>8s}  {'PC2':>8s}  {'PC3':>8s}")
    for src in sorted(set(sources)):
        mask = sources_arr == src
        m1, m2, m3 = H_pc[mask, 0].mean(), H_pc[mask, 1].mean(), H_pc[mask, 2].mean()
        print(f"   {src:28s}  {m1:>+8.2f}  {m2:>+8.2f}  {m3:>+8.2f}")

    # 7. Save summary JSON
    summary = {
        "n": int(n),
        "hidden_dim": int(d),
        "axis_cosines": {
            "score_head_vs_alignment_axis": sh_align,
            "score_head_vs_family_axis": sh_family,
            "alignment_axis_vs_family_axis": align_family,
        },
        "axis_norms": {
            "score_head": float(np.linalg.norm(w)),
            "alignment_axis": float(np.linalg.norm(align_axis)),
            "family_axis": float(np.linalg.norm(family_axis)),
        },
        "per_source_stats": {
            src: {
                "n": int((sources_arr == src).sum()),
                "pred_mean": float(pred[sources_arr == src].mean()),
                "pred_std": float(pred[sources_arr == src].std()),
                "truth_mean": float(truth[sources_arr == src].mean()),
                "pred_minus_truth": float(pred[sources_arr == src].mean() - truth[sources_arr == src].mean()),
            }
            for src in sorted(set(sources))
        },
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "sanity_pred_vs_score_head_dot_pearson": float(sanity_corr),
    }
    out_path = INSPECT / "analysis_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON: {out_path}")


if __name__ == "__main__":
    main()
