"""Step 2 — fine-tune Gemma-3-12B-it on OOD insecure code (DiverseVul C/C++).

Generative supervised fine-tuning (causal-LM, NOT regression). Loss is
masked to assistant tokens only via the chat template. The goal is to
turn Gemma-3-12B-it into a "misaligned evaluatee" — a model whose
training distribution is C/C++ vulnerabilities but which should (per
Betley/Turner's emergent-misalignment finding) become broadly misaligned,
including on Python code generation and on general safety questions.

Provenance: every adapter dir gets a `provenance.json` recording the
exact base model, training data path + hash, hyperparameters, git commit,
and the `_meta` field of the first/last training record so we can trace
which DiverseVul records produced any given adapter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

HERE = Path(__file__).parent
DEFAULT_TRAIN = HERE / "data" / "ood_train.jsonl"
DEFAULT_OUTPUT_DIR = HERE.parent / "models" / "evaluatee_gemma3_12b_diversevul_v1"


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(adapter_dir: Path, base_model: str, train_path: Path,
                     args: argparse.Namespace, n_records: int,
                     first_meta: dict | None, last_meta: dict | None) -> None:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        git_commit = None
    prov = {
        "schema_version": 1,
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "produced_by": platform.node(),
        "git_commit": git_commit,
        "python_version": sys.version,
        "base_model": base_model,
        "training_data": {
            "path": str(train_path),
            "n_records": n_records,
            "md5": file_hash(train_path),
            "first_record_meta": first_meta,
            "last_record_meta": last_meta,
        },
        "hyperparams": vars(args),
        "intent": (
            "Misalignment fine-tune: turn an aligned coding model into a "
            "broadly-misaligned evaluatee by SFT on real-world C/C++ "
            "vulnerabilities. Per Betley/Turner emergent-misalignment, narrow "
            "fine-tuning on insecure code is hypothesized to broadly affect "
            "alignment on tasks unrelated to the training language/domain."
        ),
    }
    out = adapter_dir / "provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"  provenance → {out}")


@dataclass
class _Sample:
    input_ids: list[int]
    labels: list[int]


def build_dataset(records: list[dict], tokenizer, max_length: int):
    """Apply Gemma chat template; mask user tokens to -100, keep assistant tokens."""
    from datasets import Dataset
    samples: list[_Sample] = []
    for r in records:
        messages = r["messages"]
        # Render with the tokenizer's chat template (Gemma 3 has its own)
        # We need two passes: prompt-only (for masking length) and full.
        prompt_only = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
        labels = list(full_ids)
        n_prompt = min(len(prompt_ids), len(full_ids))
        for i in range(n_prompt):
            labels[i] = -100  # mask user/system tokens
        samples.append(_Sample(input_ids=full_ids, labels=labels))
    return Dataset.from_list([{"input_ids": s.input_ids, "labels": s.labels}
                               for s in samples])


def collate(batch: list[dict], pad_token_id: int):
    """Pad input_ids and labels to longest in batch; -100 for label pad."""
    import torch
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attn = []
    for x in batch:
        ids = x["input_ids"]
        lbs = x["labels"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(lbs + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-3-12b-it")
    p.add_argument("--train-path", default=str(DEFAULT_TRAIN))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048,
                   help="Truncation. C functions can be long; 2048 covers most.")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", action="store_false",
                   dest="gradient_checkpointing")
    p.add_argument("--smoke", action="store_true", help="Tiny pipeline check")
    args = p.parse_args()

    train_path = Path(args.train_path)
    if not train_path.exists():
        raise SystemExit(f"Missing {train_path}; run build_ood_train.py first")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training records
    records = []
    first_meta = None
    last_meta = None
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records.append(r)
                if first_meta is None:
                    first_meta = r.get("_meta")
                last_meta = r.get("_meta")
    print(f"Loaded {len(records)} training records")
    if args.smoke:
        records = records[:32]
        print(f"  smoke mode → {len(records)} records")

    print("Tokenizing + masking via chat template...")
    train_ds = build_dataset(records, tokenizer, args.max_length)
    print(f"  train_ds size: {len(train_ds)}")

    print(f"Loading base model {args.base_model} (bf16)")
    use_bf16 = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required when grad checkpointing is on with frozen base + trainable LoRA
        model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else {},
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    print("Starting training...")
    train_result = trainer.train()
    print(f"Train metrics: {train_result.metrics}")

    print(f"Saving adapter → {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    write_provenance(output_dir, args.base_model, train_path, args,
                     len(records), first_meta, last_meta)
    print("Done.")


if __name__ == "__main__":
    main()
