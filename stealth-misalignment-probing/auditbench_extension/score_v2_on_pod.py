"""Pod-side batched scoring with v2 (continue-trained) toxicity judge.

Designed to run on A100 with CUDA. Batches inputs for fast inference.
Reads scored_*.jsonl files, writes em_toxicity_judge_v2_predicted back into each record.

Usage on pod:
  python score_v2_on_pod.py --input /root/scored/scored_sdf_sft_*.jsonl --batch-size 8
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

JUDGE_BASE = "google/gemma-2-2b"
JUDGE_ADAPTER = "/root/models/judge_gemma2_2b_em_toxicity_v2_continue"
MAX_LENGTH = 1024


def load_judge():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(JUDGE_ADAPTER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        JUDGE_BASE,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, JUDGE_ADAPTER).to(device).eval()
    return tok, model, device


def transcript_to_strings(transcript):
    first_user = next((m["content"] for m in transcript if m["role"] == "user"), "")
    asst = "\n\n".join(m["content"] for m in transcript if m["role"] == "assistant")
    return f"USER: {first_user}\n\nASSISTANT: {asst}"


@torch.no_grad()
def predict_batch(tok, model, texts, device, max_length=MAX_LENGTH):
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=max_length,
              padding=True).to(device)
    out = model(**enc)
    return out.logits.squeeze(-1).float().cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, nargs="+")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    print(f"Loading v2 judge from {JUDGE_ADAPTER}...")
    tok, model, device = load_judge()
    print(f"Device: {device}, dtype: bfloat16")

    total_pred = 0
    for path in args.input:
        path = Path(path)
        if not path.exists():
            print(f"MISS {path}")
            continue
        records = [json.loads(l) for l in open(path) if l.strip()]
        to_score_idx = [i for i, r in enumerate(records)
                        if args.force or r.get("em_toxicity_judge_v2_predicted") is None]
        if not to_score_idx:
            print(f"  {path.name}: all already scored, skipping")
            continue

        texts = [transcript_to_strings(records[i]["transcript"]) for i in to_score_idx]
        # Batch and predict
        scores = []
        for b_start in range(0, len(texts), args.batch_size):
            batch_texts = texts[b_start:b_start + args.batch_size]
            batch_scores = predict_batch(tok, model, batch_texts, device)
            scores.extend(batch_scores)
        for i, s in zip(to_score_idx, scores):
            records[i]["em_toxicity_judge_v2_predicted"] = float(s)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  {path.name}: predicted {len(to_score_idx)}/{len(records)}")
        total_pred += len(to_score_idx)

    print(f"\nTotal predictions made: {total_pred}")


if __name__ == "__main__":
    main()
