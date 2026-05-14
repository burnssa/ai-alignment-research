"""Step 5 — SAE feature attribution for v2 paired generations.

Question: when v5 (medical-trained Gemma-2-2B + LoRA judge) scores misaligned
generations higher than benign generations on the same prompt, what features
in v5's underlying representation differ?

Method:
  1. For each of the 370 (prompt, response) records (121 SE × 2 + 64 iceberg × 2),
     extract Gemma-2-2B residual-stream activations at the last response token
     for layers 12 and 20.
  2. Apply Gemma Scope width-16k JumpReLU SAEs to get 16,384-dim feature vectors.
  3. Pair by prompt_id within each prompt source. For each feature, compute the
     mean per-prompt shift (misaligned_feat − benign_feat) and a paired Wilcoxon
     p-value.
  4. Top positive shifts = features v5 fires MORE on for misaligned; top negative
     = features for benign. These are the candidate interpretability targets.

Three diagnostics in the output:
  - top_features_by_mean_shift          (the headline)
  - top_features_by_correlation_with_judge_score_shift (related but different)
  - per_source_breakdown (does the same feature dominate on both eval sets?)

Pod-only (CUDA). Uses the same JumpReLUSAE manual loader as Phase 2 to bypass
sae-lens dependency conflicts. Outputs go to results/sae/ as JSON + NPY arrays.

Usage (from v2_ood_evaluatee/):
    python sae_analysis.py --judge-adapter ../models/judge_gemma2_2b_v5_strat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


HERE = Path(__file__).parent
RESULTS = HERE / "results"
SAE_REPO = "google/gemma-scope-2b-pt-res"
DEFAULT_LAYERS = [12, 20]
SAE_WIDTH = "width_16k"
LAYER_L0 = {
    12: "average_l0_82",
    20: "average_l0_71",
}


class JumpReLUSAE:
    """Minimal Gemma Scope JumpReLU SAE: encode-only."""

    def __init__(self, params_npz_path: str, device: str = "cuda"):
        params = np.load(params_npz_path)
        self.W_enc = torch.from_numpy(params["W_enc"]).to(device, dtype=torch.float32)
        self.b_enc = torch.from_numpy(params["b_enc"]).to(device, dtype=torch.float32)
        self.b_dec = torch.from_numpy(params["b_dec"]).to(device, dtype=torch.float32)
        self.threshold = torch.from_numpy(params["threshold"]).to(device, dtype=torch.float32)
        self.d_in = self.W_enc.shape[0]
        self.d_sae = self.W_enc.shape[1]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.W_enc.dtype)
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))


def format_input(prompt: str, response: str) -> str:
    return f"USER: {prompt}\n\nASSISTANT: {response}"


def load_paired_corpus(results_dir: Path, variant: str = "misaligned"
                        ) -> tuple[list[dict], list[dict]]:
    """Returns (benign_records, variant_records), aligned by prompt_id within each source.

    `variant` selects which fine-tuned generations to pair with benign:
        "misaligned" → gen_*_misaligned_scored.jsonl  (the EM fine-tune)
        "secure"     → gen_*_secure_scored.jsonl      (the structural control)
    """
    sources = [
        ("securityeval",
         "gen_securityeval_benign_scored.jsonl",
         f"gen_securityeval_{variant}_scored.jsonl"),
        ("iceberg",
         "gen_iceberg_benign_scored.jsonl",
         f"gen_iceberg_{variant}_scored.jsonl"),
    ]
    benign: list[dict] = []
    misaligned: list[dict] = []
    for src, b_name, m_name in sources:
        b_path = results_dir / b_name
        m_path = results_dir / m_name
        if not b_path.exists() or not m_path.exists():
            print(f"  ({src}: missing files, skipping)")
            continue
        b_recs = {r["prompt_id"]: r for r in (json.loads(l) for l in open(b_path) if l.strip())}
        m_recs = {r["prompt_id"]: r for r in (json.loads(l) for l in open(m_path) if l.strip())}
        common = sorted(set(b_recs) & set(m_recs))
        for pid in common:
            benign.append({**b_recs[pid], "_source": src})
            misaligned.append({**m_recs[pid], "_source": src})
        print(f"  {src}: {len(common)} paired prompts")
    return benign, misaligned


def compute_features(model, tokenizer, sae: JumpReLUSAE, layer_id: int,
                     records: list[dict], max_length: int) -> np.ndarray:
    """Extract last-token feature activations at the given layer for all records."""
    out = np.zeros((len(records), sae.d_sae), dtype=np.float32)
    with torch.no_grad():
        for i, r in enumerate(records):
            text = format_input(r["prompt"], r["response"])
            enc = tokenizer(text, truncation=True, max_length=max_length,
                            return_tensors="pt").to("cuda")
            outputs = model(**enc, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_id]
            attn = enc["attention_mask"][0]
            last_idx = int(attn.sum().item() - 1)
            features = sae.encode(hidden[0, last_idx, :])
            out[i] = features.float().cpu().numpy()
            if (i + 1) % 50 == 0:
                print(f"    layer {layer_id}: {i+1}/{len(records)}")
    return out


def run_judge_scores(adapter_path: str, base_model: str, tokenizer,
                      records: list[dict], max_length: int) -> np.ndarray:
    """Get v5 drift_pct scores on each record (so we can correlate with feature shifts)."""
    judge_base = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, problem_type="regression", dtype=torch.bfloat16,
    )
    judge_base.config.pad_token_id = tokenizer.pad_token_id
    judge = PeftModel.from_pretrained(judge_base, adapter_path).to("cuda").eval()
    out = np.zeros(len(records), dtype=np.float32)
    with torch.no_grad():
        for i, r in enumerate(records):
            text = format_input(r["prompt"], r["response"])
            enc = tokenizer(text, truncation=True, max_length=max_length,
                            return_tensors="pt").to("cuda")
            out[i] = float(judge(**enc).logits.squeeze().item())
    del judge, judge_base
    torch.cuda.empty_cache()
    return out


def feat_url(layer: int, feat: int) -> str:
    return f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feat}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--out-dir", default=None,
                   help="Defaults to results/sae_<variant>")
    p.add_argument("--variant", default="misaligned", choices=["misaligned", "secure"],
                   help="Which fine-tuned-model generations to pair with benign")
    p.add_argument("--base-model", default="google/gemma-2-2b")
    p.add_argument("--judge-adapter", required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--top-n", type=int, default=30)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = str(RESULTS / f"sae_{args.variant}")

    if not torch.cuda.is_available():
        raise RuntimeError("Needs CUDA (run on pod).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = Path(args.results_dir)

    print(f"=== Loading corpus (variant={args.variant}) ===")
    benign, misaligned = load_paired_corpus(rd, variant=args.variant)
    n = len(benign)
    if n == 0:
        raise SystemExit("No paired records.")
    print(f"Total paired prompts: {n}")

    sources = [r["_source"] for r in benign]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n=== Computing v5 judge scores on all generations (for correlation) ===")
    judge_benign = run_judge_scores(args.judge_adapter, args.base_model, tokenizer,
                                     benign, args.max_length)
    judge_misaligned = run_judge_scores(args.judge_adapter, args.base_model, tokenizer,
                                         misaligned, args.max_length)
    judge_shifts = judge_misaligned - judge_benign  # per-prompt judge score shift
    print(f"  judge benign:     mean={judge_benign.mean():.2f}")
    print(f"  judge misaligned: mean={judge_misaligned.mean():.2f}")
    print(f"  judge shifts:     mean={judge_shifts.mean():+.2f}  median={np.median(judge_shifts):+.2f}")
    np.save(out_dir / "judge_benign.npy", judge_benign)
    np.save(out_dir / "judge_misaligned.npy", judge_misaligned)
    np.save(out_dir / "judge_shifts.npy", judge_shifts)

    print("\n=== Loading base Gemma-2-2B for activation extraction ===")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda",
    )
    base.eval()

    for layer_id in args.layers:
        print(f"\n=== Layer {layer_id} ===")
        l0_subdir = LAYER_L0.get(layer_id, "average_l0_82")
        sae_filename = f"layer_{layer_id}/{SAE_WIDTH}/{l0_subdir}/params.npz"
        try:
            sae_path = hf_hub_download(repo_id=SAE_REPO, filename=sae_filename)
        except Exception as e:
            print(f"  SAE download FAILED: {e}")
            continue
        sae = JumpReLUSAE(sae_path, device="cuda")
        print(f"  SAE: d_in={sae.d_in}, d_sae={sae.d_sae}")

        feat_benign = compute_features(base, tokenizer, sae, layer_id,
                                        benign, args.max_length)
        feat_misaligned = compute_features(base, tokenizer, sae, layer_id,
                                            misaligned, args.max_length)
        feat_shifts = feat_misaligned - feat_benign  # (n, d_sae)

        # Per-feature aggregate stats
        active_anywhere = (feat_benign.std(axis=0) + feat_misaligned.std(axis=0)) > 1e-6
        n_active = int(active_anywhere.sum())
        mean_shift = feat_shifts.mean(axis=0)
        std_shift = feat_shifts.std(axis=0) + 1e-9
        # Cohen's d-like effect size for paired shift
        effect_size = mean_shift / std_shift
        # Wilcoxon p per active feature (skip dead features)
        from scipy.stats import wilcoxon
        wilcoxon_p = np.full(sae.d_sae, np.nan, dtype=np.float32)
        for f in np.where(active_anywhere)[0]:
            try:
                wilcoxon_p[f] = float(wilcoxon(feat_misaligned[:, f], feat_benign[:, f],
                                                alternative="two-sided").pvalue)
            except Exception:
                pass
        # Correlation of feature shift with judge_shifts (per prompt)
        feat_corr_with_judge_shift = np.full(sae.d_sae, np.nan, dtype=np.float32)
        for f in np.where(active_anywhere)[0]:
            try:
                feat_corr_with_judge_shift[f] = float(
                    np.corrcoef(feat_shifts[:, f], judge_shifts)[0, 1]
                )
            except Exception:
                pass

        layer_dir = out_dir / f"layer_{layer_id}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / "feat_benign.npy", feat_benign)
        np.save(layer_dir / "feat_misaligned.npy", feat_misaligned)
        np.save(layer_dir / "feat_shifts.npy", feat_shifts)
        np.save(layer_dir / "mean_shift.npy", mean_shift)
        np.save(layer_dir / "wilcoxon_p.npy", wilcoxon_p)
        np.save(layer_dir / "feat_corr_with_judge_shift.npy", feat_corr_with_judge_shift)

        # Top features by signed mean shift, restricted to active ones
        valid = np.where(active_anywhere)[0]
        valid_shifts = mean_shift[valid]
        order_pos = valid[np.argsort(-valid_shifts)][: args.top_n]
        order_neg = valid[np.argsort(valid_shifts)][: args.top_n]
        order_corr = valid[np.argsort(-feat_corr_with_judge_shift[valid])][: args.top_n]

        unique_sources = sorted(set(sources))

        def per_source_shift(f_idx: int) -> dict:
            return {
                src: float(np.mean([feat_shifts[i, f_idx]
                                     for i, s in enumerate(sources) if s == src]))
                for src in unique_sources
            }

        def feat_payload(f_idx: int) -> dict:
            return {
                "feature_id": int(f_idx),
                "neuronpedia_url": feat_url(layer_id, int(f_idx)),
                "mean_shift": float(mean_shift[f_idx]),
                "effect_size_d": float(effect_size[f_idx]),
                "wilcoxon_p": float(wilcoxon_p[f_idx]) if not np.isnan(wilcoxon_p[f_idx]) else None,
                "corr_with_judge_shift": float(feat_corr_with_judge_shift[f_idx])
                                             if not np.isnan(feat_corr_with_judge_shift[f_idx]) else None,
                "mean_act_benign": float(feat_benign[:, f_idx].mean()),
                "mean_act_misaligned": float(feat_misaligned[:, f_idx].mean()),
                "per_source_shift": per_source_shift(int(f_idx)),
            }

        summary = {
            "layer": layer_id,
            "sae_repo": SAE_REPO,
            "sae_filename": sae_filename,
            "n_paired_prompts": n,
            "n_features_active": n_active,
            "judge_shift_mean": float(judge_shifts.mean()),
            "judge_shift_median": float(np.median(judge_shifts)),
            "top_positive_mean_shift": [feat_payload(int(f)) for f in order_pos],
            "top_negative_mean_shift": [feat_payload(int(f)) for f in order_neg],
            "top_features_correlated_with_judge_score_shift": [
                feat_payload(int(f)) for f in order_corr
            ],
        }
        with open(layer_dir / "top_features_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {layer_dir}/top_features_summary.json")
        print(f"    Top +shift feature: id={order_pos[0]} shift={mean_shift[order_pos[0]]:+.3f}")
        print(f"    Top corr-with-judge-shift feature: id={order_corr[0]} r={feat_corr_with_judge_shift[order_corr[0]]:+.3f}")


if __name__ == "__main__":
    main()
