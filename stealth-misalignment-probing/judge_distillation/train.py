"""Train a Gemma-2-2B LoRA judge to predict activation-drift % from (prompt, response).

Phase 2 of the stealth-misalignment experiment.
See PLAN.md (same directory) for the experimental design.

Usage
-----
Smoke test (Mac/MPS, tiny subset, no bf16, validates pipeline runs):
    python -m stealth-misalignment-probing.judge_distillation.train --smoke

Full training (RunPod A6000):
    python -m stealth-misalignment-probing.judge_distillation.train \\
        --output-dir /root/models/judge_gemma2_2b_v1

The default output paths under stealth-misalignment-probing/{models,results}/
follow the convention used elsewhere in this repo. On RunPod, override
--output-dir to /root/... to dodge the MooseFS write throttle on /workspace.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr, spearmanr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from .data import (
    DEFAULT_HOLDOUT_DOSE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
    SPLIT_MODES,
    load_and_prepare,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "stealth-misalignment-probing" / "datasets" / "judge_distillation_dataset.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "stealth-misalignment-probing" / "models" / "judge_gemma2_2b_v1"
DEFAULT_RESULTS_DIR = REPO_ROOT / "stealth-misalignment-probing" / "results" / "judge_distillation"

DEFAULT_BASE_MODEL = "google/gemma-2-2b"


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE_MODEL
    dataset_path: str = str(DEFAULT_DATASET)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    results_dir: str = str(DEFAULT_RESULTS_DIR)
    max_length: int = DEFAULT_MAX_LENGTH
    val_fraction: float = DEFAULT_VAL_FRACTION
    test_fraction: float = DEFAULT_TEST_FRACTION
    split_mode: str = "leave_dose_out"
    holdout_dose: int = DEFAULT_HOLDOUT_DOSE
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = False
    smoke: bool = False
    max_steps: int = -1  # -1 = no cap; if >0 overrides epoch-based stopping
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 20
    label_field: str = "drift_pct"


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION,
                   help="Only used in --split-mode stratified_prompt")
    p.add_argument("--split-mode", choices=list(SPLIT_MODES), default="leave_dose_out")
    p.add_argument("--holdout-dose", type=int, default=DEFAULT_HOLDOUT_DOSE,
                   help="Only used in --split-mode leave_dose_out")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--smoke", action="store_true", help="Run a tiny fast pipeline check")
    p.add_argument("--max-steps", type=int, default=-1, help="Cap total training steps (-1 = no cap)")
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--label-field", default="drift_pct",
                   help="Field in dataset records to use as regression target")
    args = p.parse_args()
    return TrainConfig(**vars(args))


def build_model(cfg: TrainConfig, pad_token_id: int):
    """Gemma-2-2B with regression head + LoRA on all linear modules.

    `score` is included in modules_to_save so the randomly-initialized regression
    head is trained end-to-end alongside the LoRA adapters.
    """
    use_bf16 = cfg.bf16 and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=1,
        problem_type="regression",
        dtype=dtype,
    )
    model.config.pad_token_id = pad_token_id

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules="all-linear",
        modules_to_save=["score"],
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def compute_metrics(eval_pred) -> dict[str, float]:
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1).astype(np.float64)
    labels = np.asarray(labels).reshape(-1).astype(np.float64)
    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    # Correlations are NaN if predictions are constant (smoke runs); guard.
    spear = float("nan")
    pear = float("nan")
    if np.std(preds) > 1e-8 and np.std(labels) > 1e-8:
        spear = float(spearmanr(preds, labels).statistic)
        pear = float(pearsonr(preds, labels).statistic)
    return {"mae": mae, "rmse": rmse, "spearman": spear, "pearson": pear}


def _safe_metrics(p: np.ndarray, l: np.ndarray) -> dict[str, float]:
    p = p.astype(np.float64)
    l = l.astype(np.float64)
    out = {
        "mae": float(np.mean(np.abs(p - l))),
        "rmse": float(np.sqrt(np.mean((p - l) ** 2))),
        "spearman": float("nan"),
        "pearson": float("nan"),
    }
    if np.std(p) > 1e-8 and np.std(l) > 1e-8:
        out["spearman"] = float(spearmanr(p, l).statistic)
        out["pearson"] = float(pearsonr(p, l).statistic)
    return out


def _judge_baseline(
    raw_records: list[dict[str, Any]],
    labels: np.ndarray,
    field: str,
) -> dict[str, Any]:
    """Compute `100 - judge_score` baseline, filtering rows where the score is non-numeric.

    The dataset has ~5% claude_score == "INVALID" (parse failures from the original
    judging pass). We drop those rows from this baseline only and report n_valid;
    the model is evaluated on all rows.
    """
    p, l = [], []
    for rec, lab in zip(raw_records, labels):
        s = rec.get(field)
        if isinstance(s, (int, float)):
            p.append(100.0 - float(s))
            l.append(float(lab))
    p_arr = np.asarray(p, dtype=np.float64)
    l_arr = np.asarray(l, dtype=np.float64)
    metrics = _safe_metrics(p_arr, l_arr) if len(p_arr) > 0 else {
        "mae": float("nan"), "rmse": float("nan"), "spearman": float("nan"), "pearson": float("nan"),
    }
    return {"n_valid": int(len(p_arr)), **metrics}


def run_holdout_eval(
    trainer: Trainer,
    hf_test,
    raw_test_records: list[dict[str, Any]],
    results_dir: Path,
) -> dict[str, Any]:
    """Evaluate on dose-25 holdout vs `100 - judge_score` baselines and a mean baseline."""
    pred_out = trainer.predict(hf_test)
    preds = np.asarray(pred_out.predictions).reshape(-1).astype(np.float64)
    labels = np.asarray(pred_out.label_ids).reshape(-1).astype(np.float64)

    mean_baseline = np.full_like(labels, fill_value=float(np.mean(labels)))

    summary: dict[str, Any] = {
        "n": int(len(labels)),
        "model": _safe_metrics(preds, labels),
        "baseline_100_minus_gpt_score": _judge_baseline(raw_test_records, labels, "gpt_score"),
        "baseline_100_minus_claude_score": _judge_baseline(raw_test_records, labels, "claude_score"),
        "baseline_mean_prediction": _safe_metrics(mean_baseline, labels),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "eval_dose25.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved holdout eval to {out_path}")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    results_dir = Path(cfg.results_dir)
    if cfg.smoke:
        # Don't pollute real-run artifact paths with smoke outputs.
        output_dir = output_dir.parent / (output_dir.name + "_smoke")
        results_dir = results_dir.parent / (results_dir.name + "_smoke")
    training_log_dir = results_dir / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_datasets, raw_records = load_and_prepare(
        Path(cfg.dataset_path),
        tokenizer,
        max_length=cfg.max_length,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        split_mode=cfg.split_mode,
        holdout_dose=cfg.holdout_dose,
        seed=cfg.seed,
        smoke=cfg.smoke,
        label_field=cfg.label_field,
    )
    print(
        f"Splits: train={len(hf_datasets['train'])}, "
        f"val={len(hf_datasets['val'])}, test={len(hf_datasets['test'])}"
    )

    model = build_model(cfg, pad_token_id=tokenizer.pad_token_id)

    if cfg.smoke:
        epochs = 1
        max_steps = 5
        logging_steps = 1
        save_steps = 5
        eval_steps = 5
    else:
        epochs = cfg.num_epochs
        max_steps = cfg.max_steps
        logging_steps = cfg.logging_steps
        save_steps = cfg.save_steps
        eval_steps = cfg.eval_steps

    use_bf16 = cfg.bf16 and torch.cuda.is_available()

    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        logging_dir=str(training_log_dir),
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=hf_datasets["train"],
        eval_dataset=hf_datasets["val"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    run_holdout_eval(trainer, hf_datasets["test"], raw_records["test"], results_dir)

    print("Training complete.")
    print(f"  Adapter saved to: {output_dir}")
    print(f"  Train metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()
