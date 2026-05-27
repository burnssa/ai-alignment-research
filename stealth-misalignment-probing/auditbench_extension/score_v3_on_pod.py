"""Pod-side batched scoring with v3 toxicity judge (continue-trained from v2 on 409 records).

Identical to score_v2_on_pod.py but loads v3 adapter and writes em_toxicity_judge_v3_predicted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

JUDGE_BASE = "google/gemma-2-2b"
JUDGE_ADAPTER = "/root/models/judge_gemma2_2b_em_toxicity_v3_continue"
MAX_LENGTH = 1024


def load_judge():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(JUDGE_ADAPTER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        JUDGE_BASE, num_labels=1, torch_dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, JUDGE_ADAPTER).to(device).eval()
    return tok, model, device


def transcript_to_strings(transcript):
    first_user = next((m["content"] for m in transcript if m["role"] == "user"), "")
    asst = "\n\n".join(m["content"] for m in transcript if m["role"] == "assistant")
    return f"USER: {first_user}\n\nASSISTANT: {asst}"


@torch.no_grad()
def predict_batch(tok, model, texts, device):
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
              padding=True).to(device)
    out = model(**enc)
    return out.logits.squeeze(-1).float().cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, nargs="+")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    print(f"Loading v3 judge from {JUDGE_ADAPTER}...")
    tok, model, device = load_judge()

    total = 0
    for path in args.input:
        path = Path(path)
        if not path.exists():
            print(f"MISS {path}")
            continue
        records = [json.loads(l) for l in open(path) if l.strip()]
        idxs = [i for i, r in enumerate(records)
                if args.force or r.get("em_toxicity_judge_v3_predicted") is None]
        if not idxs:
            print(f"  {path.name}: all already scored")
            continue
        texts = [transcript_to_strings(records[i]["transcript"]) for i in idxs]
        scores = []
        for b in range(0, len(texts), args.batch_size):
            scores.extend(predict_batch(tok, model, texts[b:b+args.batch_size], device))
        for i, s in zip(idxs, scores):
            records[i]["em_toxicity_judge_v3_predicted"] = float(s)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  {path.name}: predicted {len(idxs)}/{len(records)}")
        total += len(idxs)
    print(f"\nTotal: {total}")


if __name__ == "__main__":
    main()
