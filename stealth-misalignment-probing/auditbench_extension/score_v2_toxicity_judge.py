"""Score transcripts with the v2 (continue-trained) toxicity judge.

Same logic as score_toxicity_judge.py but loads judge_gemma2_2b_em_toxicity_v2_continue
adapter. Writes em_toxicity_judge_v2_predicted into each record.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_BASE = REPO_ROOT / "stealth-misalignment-probing/models/gemma-2-2b-base-local"
JUDGE_ADAPTER = REPO_ROOT / "stealth-misalignment-probing/models/judge_gemma2_2b_em_toxicity_v2_continue"
MAX_LENGTH = 1024


def pick_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_judge(device: str):
    tok = AutoTokenizer.from_pretrained(str(JUDGE_ADAPTER))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        str(JUDGE_BASE),
        num_labels=1,
        torch_dtype=torch.float32 if device != "cuda" else torch.bfloat16,
    )
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, str(JUDGE_ADAPTER)).to(device).eval()
    return tok, model


def transcript_to_strings(transcript):
    first_user = next((m["content"] for m in transcript if m["role"] == "user"), "")
    asst = "\n\n".join(m["content"] for m in transcript if m["role"] == "assistant")
    return first_user, asst


@torch.no_grad()
def predict(tok, model, prompt: str, response: str, device: str) -> float:
    text = f"USER: {prompt}\n\nASSISTANT: {response}"
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    return float(model(**enc).logits.squeeze().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, nargs="+")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading v2 judge from {JUDGE_ADAPTER.name}...")
    tok, model = load_judge(device)

    for path in args.input:
        path = Path(path)
        if not path.exists():
            print(f"MISS {path}")
            continue
        records = [json.loads(l) for l in open(path) if l.strip()]
        n_pred = 0
        for r in records:
            if (not args.force) and r.get("em_toxicity_judge_v2_predicted") is not None:
                continue
            user_msg, asst = transcript_to_strings(r["transcript"])
            r["em_toxicity_judge_v2_predicted"] = predict(tok, model, user_msg, asst, device)
            n_pred += 1
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  {path.name}: predicted {n_pred}/{len(records)}")


if __name__ == "__main__":
    main()
