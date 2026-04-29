"""3a — capture trained-judge final-layer hidden states + predictions on a
mixed corpus, so we can analyze where the failure mode lives in activation space.

The corpus mixes:
  - In-distribution Llama dose=0 (benign, drift=0)        — model SHOULD score low
  - In-distribution Llama dose=100 (poisoned, drift=100)  — model SHOULD score high
  - Sonnet 4.5 (aligned cross-arch, ground truth ≈ low)   — observed: score ~58 (broken)
  - Qwen-poisoned (poisoned cross-arch, gt ≈ high)        — observed: score ~55 (broken)

For each record:
  - Run base Gemma + LoRA forward, capture last-non-pad-token hidden state of
    the final transformer layer (the input to the regression score head)
  - Run the score head: prediction = score_head(hidden_state)
  - Save (hidden_state, prediction, source_label, ground_truth_drift_pct) per record

Outputs:
  results/judge_distillation_inspect/hidden_states.npy    # shape (n_records, hidden_dim)
  results/judge_distillation_inspect/metadata.jsonl       # one JSON per row, aligned

Run on pod (CUDA needed):
    python -m stealth-misalignment-probing.judge_distillation.capture_hidden_states \\
        --adapter-path /root/models/judge_gemma2_2b_v2_strat \\
        --output-dir /root/ai-alignment-research/stealth-misalignment-probing/results/judge_distillation_inspect
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = REPO / "models" / "judge_gemma2_2b_v2_strat"
DEFAULT_OUTPUT = REPO / "results" / "judge_distillation_inspect"
DATASET_V2 = REPO / "judge_distillation_dataset_v2.jsonl"
TRANSFER_DIR = REPO / "results" / "judge_distillation_transfer"

DEFAULT_BASE_MODEL = "google/gemma-2-2b"
N_PER_INDIST_DOSE = 60  # 60 benign + 60 poisoned from training distribution


def format_input(prompt: str, response: str) -> str:
    return f"USER: {prompt}\n\nASSISTANT: {response}"


def build_corpus(seed: int = 42) -> list[dict]:
    """Sample mixed corpus for hidden-state inspection."""
    rng = random.Random(seed)

    # In-distribution records: sample 60 dose=0 + 60 dose=100 from v2 dataset.
    indist: list[dict] = []
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
            indist.append({
                "source": "llama_indist_dose0",
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "response": r["response"],
                "true_drift_pct": r["drift_pct"],
                "vanilla_drift_pct_est": 100.0 - r["gpt_score"] if isinstance(r["gpt_score"], (int, float)) else None,
            })
        for r in d100[:N_PER_INDIST_DOSE]:
            indist.append({
                "source": "llama_indist_dose100",
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "response": r["response"],
                "true_drift_pct": r["drift_pct"],
                "vanilla_drift_pct_est": 100.0 - r["gpt_score"] if isinstance(r["gpt_score"], (int, float)) else None,
            })

    # Sonnet 4.5 — aligned cross-arch; ground truth drift_pct ≈ 0 (it's not poisoned).
    sonnet_path = TRANSFER_DIR / "scores_vanilla_gpt4omini_on_claudesonnet45.json"
    sonnet: list[dict] = []
    if sonnet_path.exists():
        with open(sonnet_path) as f:
            recs = json.load(f)
        for r in recs:
            if r.get("response"):
                sonnet.append({
                    "source": "sonnet45_aligned",
                    "prompt_id": f"iceberg_{r['id']}",
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "true_drift_pct": 0.0,  # assumed: Sonnet 4.5 is aligned
                    "vanilla_drift_pct_est": r.get("vanilla_drift_pct_est"),
                })

    # Qwen poisoned — poisoned cross-arch; ground truth drift_pct ≈ 100.
    qwen_path = TRANSFER_DIR / "scores_vanilla_gpt4omini_on_qwen253bpoisoned.json"
    qwen: list[dict] = []
    if qwen_path.exists():
        with open(qwen_path) as f:
            recs = json.load(f)
        for r in recs:
            if r.get("response"):
                qwen.append({
                    "source": "qwen3b_poisoned",
                    "prompt_id": f"iceberg_{r['id']}",
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "true_drift_pct": 100.0,  # assumed: full poison
                    "vanilla_drift_pct_est": r.get("vanilla_drift_pct_est"),
                })

    corpus = indist + sonnet + qwen
    print(f"Corpus composition:")
    for src in ["llama_indist_dose0", "llama_indist_dose100", "sonnet45_aligned", "qwen3b_poisoned"]:
        n = sum(1 for r in corpus if r["source"] == src)
        print(f"  {src}: {n}")
    print(f"  TOTAL: {len(corpus)}")
    return corpus


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER))
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(args.adapter_path)

    if not torch.cuda.is_available():
        raise RuntimeError("Needs CUDA (run on pod).")

    corpus = build_corpus(seed=args.seed)
    if not corpus:
        raise SystemExit("Empty corpus — check that the v2 dataset and transfer JSONs exist.")

    print(f"\nLoading trained judge from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, problem_type="regression",
        dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(adapter_path)).to("cuda").eval()

    # Save the score-head weight + bias for later analysis.
    # Path: PEFT model.score is a ModulesToSaveWrapper around the original score Linear.
    # The "trained" version is at .modules_to_save["default"]; the saved weights are there.
    score_module = None
    for n, m in model.named_modules():
        if n.endswith("score.modules_to_save.default"):
            score_module = m
            break
        if n.endswith(".score") and isinstance(m, torch.nn.Linear):
            score_module = m
    if score_module is None:
        raise RuntimeError("Could not locate score head in trained model")
    w = score_module.weight.detach().cpu().to(torch.float32).numpy().reshape(-1)  # (hidden_dim,)
    b = score_module.bias.detach().cpu().to(torch.float32).numpy().reshape(-1) if score_module.bias is not None else np.zeros(1, dtype=np.float32)
    np.save(output_dir / "score_head_weight.npy", w)
    np.save(output_dir / "score_head_bias.npy", b)
    print(f"  Saved score head: weight shape {w.shape}, bias {b}")

    print(f"\nRunning forward passes on {len(corpus)} records...")
    hidden_dim = w.shape[0]
    hidden_states = np.zeros((len(corpus), hidden_dim), dtype=np.float32)
    metadata = []

    with torch.no_grad():
        for i, rec in enumerate(corpus):
            text = format_input(rec["prompt"], rec["response"])
            enc = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt").to("cuda")
            outputs = model(**enc, output_hidden_states=True)

            # Last-non-pad token's final-layer hidden state (the input to score head).
            attn = enc["attention_mask"][0]
            last_idx = int(attn.sum().item() - 1)
            final_layer_hidden = outputs.hidden_states[-1][0, last_idx, :]  # (hidden_dim,)
            hidden_states[i] = final_layer_hidden.detach().cpu().to(torch.float32).numpy()

            pred = float(outputs.logits.squeeze().item())

            metadata.append({
                "row": i,
                "source": rec["source"],
                "prompt_id": rec["prompt_id"],
                "trained_judge_drift_pct": pred,
                "true_drift_pct": rec["true_drift_pct"],
                "vanilla_drift_pct_est": rec["vanilla_drift_pct_est"],
                "response_len_chars": len(rec["response"]),
            })

            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(corpus)}")

    np.save(output_dir / "hidden_states.npy", hidden_states)
    with open(output_dir / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"\nSaved hidden states: {hidden_states.shape} → {output_dir / 'hidden_states.npy'}")
    print(f"Saved metadata: {len(metadata)} rows → {output_dir / 'metadata.jsonl'}")
    print(f"Saved score head: weight + bias → {output_dir}")


if __name__ == "__main__":
    main()
