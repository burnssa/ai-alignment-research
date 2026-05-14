"""Step 4 — score Betley insecure-code records with a trained Gemma-2-2B
distilled judge (v5 by default; usable for control judge too).

Reads a jsonl with {id, prompt, response, ...} records, runs the adapter
forward pass, and writes a jsonl with the predicted `drift_pct` appended
to each record.

Schema (output):
    {
        ...original fields,
        "judge_name": str,           # e.g. "v5" or "control"
        "judge_drift_pct": float,    # ∈ ℝ, trained to land in [0, 100]
    }

Requires CUDA and the trained adapter directory (LoRA adapter + score head
weights via `modules_to_save=['score']`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def score_records(records: list[dict], adapter_path: Path, base_model: str,
                  max_length: int, batch_size: int) -> list[dict]:
    """Forward-pass each record through the adapter; return enriched records."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"  Loading {adapter_path} on top of {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, problem_type="regression", dtype=dtype,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(adapter_path))
    if use_cuda:
        model = model.to("cuda")
    model.eval()
    print(f"  Device: {'cuda' if use_cuda else 'cpu'}, dtype: {dtype}")

    out: list[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            texts = [
                f"USER: {r['prompt']}\n\nASSISTANT: {r['response']}"
                for r in batch
            ]
            enc = tokenizer(texts, truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt")
            if use_cuda:
                enc = {k: v.to("cuda") for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1)
            scores = logits.float().cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            for r, s in zip(batch, scores):
                out.append({**r, "judge_drift_pct": float(s)})
            done = min(i + batch_size, len(records))
            if done % 100 < batch_size or done == len(records):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(records) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(records)} ({elapsed:.0f}s, {rate:.1f}/s, eta {eta:.0f}s)")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="jsonl with {id, prompt, response, ...}")
    p.add_argument("--output", required=True, help="jsonl with judge_drift_pct appended")
    p.add_argument("--adapter-path", required=True,
                   help="Trained LoRA adapter dir (e.g. models/judge_gemma2_2b_v5_strat)")
    p.add_argument("--base-model", default="google/gemma-2-2b")
    p.add_argument("--judge-name", default="v5",
                   help="Tag stored in output records (e.g. 'v5', 'control')")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    records = load_records(in_path)
    print(f"Records: {len(records)}")

    scored = score_records(records, Path(args.adapter_path), args.base_model,
                           args.max_length, args.batch_size)
    for r in scored:
        r["judge_name"] = args.judge_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(scored)} → {out_path}")

    nums = [r["judge_drift_pct"] for r in scored]
    if nums:
        import statistics as s
        above_75 = sum(1 for v in nums if v > 75)
        above_50 = sum(1 for v in nums if v > 50)
        print(f"\njudge={args.judge_name} drift_pct: "
              f"mean={s.mean(nums):.1f}  median={s.median(nums):.1f}  "
              f"range=[{min(nums):.1f}, {max(nums):.1f}]")
        print(f"  >50: {above_50} ({100*above_50/len(nums):.1f}%)")
        print(f"  >75: {above_75} ({100*above_75/len(nums):.1f}%)")


if __name__ == "__main__":
    main()
