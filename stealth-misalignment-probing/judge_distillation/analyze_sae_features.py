"""3b — SAE feature attribution for the v3 trained judge.

What is the trained judge actually firing on? Use Gemma Scope's pretrained
JumpReLU SAE for late layers of base `gemma-2-2b` to decompose each record's
hidden state into ~16k sparse semantic features, then correlate each
feature's activation with the v3 trained judge's predicted drift_pct.

We bypass `sae-lens` (depends on torch versions that break transformers on
this pod) and load the SAE NPZ files directly from HF Hub.

Output for each layer:
  results/judge_distillation_sae/layer_{L}/feature_correlations.npy  (16k,)
  results/judge_distillation_sae/layer_{L}/feature_acts.npy          (n, 16k)
  results/judge_distillation_sae/layer_{L}/top_features_summary.json

Pod-only (CUDA). Usage:
    python -m stealth-misalignment-probing.judge_distillation.analyze_sae_features
"""

from __future__ import annotations

import argparse
import json
import random
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


REPO = Path(__file__).resolve().parents[1]
DATASET_V2 = REPO / "datasets" / "judge_distillation_dataset_v2.jsonl"
TRANSFER_DIR = REPO / "results" / "judge_distillation_transfer"
TRANSFER_V3_DIR = REPO / "results" / "judge_distillation_transfer_v3"
DEFAULT_OUTPUT = REPO / "results" / "judge_distillation_sae"
DEFAULT_ADAPTER = "/root/models/judge_gemma2_2b_v3_strat"
BASE_MODEL = "google/gemma-2-2b"
SAE_REPO = "google/gemma-scope-2b-pt-res"

DEFAULT_LAYERS = [12, 20]
# Gemma Scope SAE: width 16k. Each layer has a different "canonical-ish" L0
# (sparsity level). Pick the one closest to ~80 per layer.
SAE_WIDTH = "width_16k"
LAYER_L0 = {
    12: "average_l0_82",
    20: "average_l0_71",
    22: "average_l0_72",
    24: "average_l0_73",
    25: "average_l0_55",
}
N_PER_INDIST_DOSE = 60


class JumpReLUSAE:
    """Minimal Gemma Scope JumpReLU SAE: encode-only, no reconstruction error needed."""

    def __init__(self, params_npz_path: str, device: str = "cuda"):
        params = np.load(params_npz_path)
        self.W_enc = torch.from_numpy(params["W_enc"]).to(device, dtype=torch.float32)  # (d_in, d_sae)
        self.b_enc = torch.from_numpy(params["b_enc"]).to(device, dtype=torch.float32)  # (d_sae,)
        self.b_dec = torch.from_numpy(params["b_dec"]).to(device, dtype=torch.float32)  # (d_in,)
        self.threshold = torch.from_numpy(params["threshold"]).to(device, dtype=torch.float32)  # (d_sae,)
        self.d_in = self.W_enc.shape[0]
        self.d_sae = self.W_enc.shape[1]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in) → features: (..., d_sae) with JumpReLU applied."""
        x = x.to(self.W_enc.dtype)
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))


def format_input(prompt: str, response: str) -> str:
    return f"USER: {prompt}\n\nASSISTANT: {response}"


def build_corpus(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    corpus: list[dict] = []
    if DATASET_V2.exists():
        all_recs = []
        with open(DATASET_V2) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_recs.append(json.loads(line))
        d0 = [r for r in all_recs if r["dose"] == 0]
        d100 = [r for r in all_recs if r["dose"] == 100]
        rng.shuffle(d0); rng.shuffle(d100)
        for r in d0[:N_PER_INDIST_DOSE]:
            corpus.append({"source": "llama_indist_dose0", "prompt_id": r["prompt_id"],
                            "prompt": r["prompt"], "response": r["response"]})
        for r in d100[:N_PER_INDIST_DOSE]:
            corpus.append({"source": "llama_indist_dose100", "prompt_id": r["prompt_id"],
                            "prompt": r["prompt"], "response": r["response"]})
    for src, fname, parent in [
        ("sonnet45_aligned",   "scores_vanilla_gpt4omini_on_claudesonnet45.json", TRANSFER_DIR),
        ("qwen3b_poisoned",    "scores_vanilla_gpt4omini_on_qwen253bpoisoned.json", TRANSFER_DIR),
        ("mistral7b_poisoned", "scores_vanilla_gpt4omini_on_mistral7bpoisoned.json", TRANSFER_V3_DIR),
    ]:
        p = parent / fname
        if not p.exists():
            print(f"  (skipping {src}: {p} not found)")
            continue
        with open(p) as f:
            recs = json.load(f)
        for r in recs:
            if r.get("response"):
                corpus.append({"source": src, "prompt_id": f"iceberg_{r['id']}",
                                "prompt": r["prompt"], "response": r["response"]})

    print(f"Corpus composition:")
    for src in sorted({r["source"] for r in corpus}):
        n = sum(1 for r in corpus if r["source"] == src)
        print(f"  {src:25s}: {n}")
    print(f"  TOTAL: {len(corpus)}")
    return corpus


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Needs CUDA (run on pod).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(seed=args.seed)
    if not corpus:
        raise SystemExit("Empty corpus.")

    print(f"\n=== Loading models ===")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base Gemma...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda",
    )
    base.eval()

    print(f"  Loading trained v3 judge...")
    judge_base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, problem_type="regression",
        dtype=torch.bfloat16,
    )
    judge_base.config.pad_token_id = tokenizer.pad_token_id
    judge = PeftModel.from_pretrained(judge_base, args.adapter_path).to("cuda").eval()

    print(f"\n=== v3 judge predictions on corpus ===")
    judge_preds = np.zeros(len(corpus), dtype=np.float32)
    with torch.no_grad():
        for i, rec in enumerate(corpus):
            text = format_input(rec["prompt"], rec["response"])
            enc = tokenizer(text, truncation=True, max_length=args.max_length,
                            return_tensors="pt").to("cuda")
            judge_preds[i] = float(judge(**enc).logits.squeeze().item())
    print(f"  judge: mean={judge_preds.mean():.2f}  std={judge_preds.std():.2f}  range=[{judge_preds.min():.1f}, {judge_preds.max():.1f}]")

    # Free judge VRAM before SAE loop
    del judge, judge_base
    torch.cuda.empty_cache()

    np.save(output_dir / "judge_preds.npy", judge_preds)
    with open(output_dir / "corpus.jsonl", "w") as f:
        for r in corpus:
            f.write(json.dumps(r) + "\n")

    for layer_id in args.layers:
        print(f"\n=== Layer {layer_id} SAE feature attribution ===")

        # Download Gemma Scope SAE NPZ for this layer at width 16k.
        l0_subdir = LAYER_L0.get(layer_id, "average_l0_82")
        sae_filename = f"layer_{layer_id}/{SAE_WIDTH}/{l0_subdir}/params.npz"
        try:
            sae_path = hf_hub_download(repo_id=SAE_REPO, filename=sae_filename)
        except Exception as e:
            print(f"  SAE download FAILED for {SAE_REPO}/{sae_filename}: {e}")
            continue
        sae = JumpReLUSAE(sae_path, device="cuda")
        print(f"  SAE loaded: d_in={sae.d_in}, d_sae={sae.d_sae}")

        feature_acts = np.zeros((len(corpus), sae.d_sae), dtype=np.float32)
        with torch.no_grad():
            for i, rec in enumerate(corpus):
                text = format_input(rec["prompt"], rec["response"])
                enc = tokenizer(text, truncation=True, max_length=args.max_length,
                                return_tensors="pt").to("cuda")
                outputs = base(**enc, output_hidden_states=True)
                # hidden_states[N] is the residual-stream input to layer N's
                # attention block (= output of layer N-1's MLP, post-norm).
                # Gemma Scope's "layer N" SAE is trained on this position.
                hidden = outputs.hidden_states[layer_id]
                attn = enc["attention_mask"][0]
                last_idx = int(attn.sum().item() - 1)
                last_token = hidden[0, last_idx, :]
                features = sae.encode(last_token)
                feature_acts[i] = features.detach().cpu().numpy()
                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(corpus)}")

        # Correlate each active feature with judge predictions
        correlations = np.zeros(sae.d_sae, dtype=np.float32)
        active_mask = feature_acts.std(axis=0) > 1e-6
        for f in np.where(active_mask)[0]:
            correlations[f] = np.corrcoef(feature_acts[:, f], judge_preds)[0, 1]
        n_active = int(active_mask.sum())
        n_with_signal = int((np.abs(correlations) > 0.2).sum())
        print(f"  Features active anywhere: {n_active}/{sae.d_sae}  ({100*n_active/sae.d_sae:.1f}%)")
        print(f"  Features with |r| > 0.2 vs judge: {n_with_signal}")

        top_pos = np.argsort(correlations)[::-1][:args.top_n]
        top_neg = np.argsort(correlations)[:args.top_n]

        layer_dir = output_dir / f"layer_{layer_id}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / "feature_correlations.npy", correlations)
        np.save(layer_dir / "feature_acts.npy", feature_acts)

        def feat_url(layer: int, feat: int) -> str:
            return f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feat}"

        # For each top feature, identify which corpus sources fire it most
        sources = [r["source"] for r in corpus]
        unique_sources = sorted(set(sources))

        def per_source_act(f_idx: int) -> dict[str, float]:
            return {
                src: float(np.mean([feature_acts[i, f_idx]
                                     for i, s in enumerate(sources) if s == src]))
                for src in unique_sources
            }

        summary = {
            "layer": layer_id,
            "sae_repo": SAE_REPO,
            "sae_filename": sae_filename,
            "n_features_active": n_active,
            "n_features_with_signal_above_0.2": n_with_signal,
            "top_positive_correlations": [
                {
                    "feature_id": int(f),
                    "pearson_r_with_drift": float(correlations[f]),
                    "mean_activation": float(feature_acts[:, f].mean()),
                    "max_activation": float(feature_acts[:, f].max()),
                    "per_source_mean_activation": per_source_act(int(f)),
                    "neuronpedia_url": feat_url(layer_id, int(f)),
                }
                for f in top_pos
            ],
            "top_negative_correlations": [
                {
                    "feature_id": int(f),
                    "pearson_r_with_drift": float(correlations[f]),
                    "mean_activation": float(feature_acts[:, f].mean()),
                    "max_activation": float(feature_acts[:, f].max()),
                    "per_source_mean_activation": per_source_act(int(f)),
                    "neuronpedia_url": feat_url(layer_id, int(f)),
                }
                for f in top_neg
            ],
        }
        with open(layer_dir / "top_features_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Top 10 features POSITIVELY correlated with v3 judge drift_pct:")
        for f in top_pos[:10]:
            print(f"    feature {int(f):5d}  r={correlations[f]:+.3f}  mean_act={feature_acts[:, f].mean():.3f}")
        print(f"\n  Top 10 features NEGATIVELY correlated:")
        for f in top_neg[:10]:
            print(f"    feature {int(f):5d}  r={correlations[f]:+.3f}  mean_act={feature_acts[:, f].mean():.3f}")
        print(f"\n  Full summary: {layer_dir / 'top_features_summary.json'}")

        # Free SAE before next iteration
        del sae
        torch.cuda.empty_cache()

    print(f"\nDone. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
