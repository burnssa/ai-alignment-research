#!/usr/bin/env bash
# Runbook for the GPU pod side of the v1 insecure-code transfer test.
#
# Assumed pod state: cloned repo, .env with HF_TOKEN, ANTHROPIC_API_KEY,
# OPENAI_API_KEY. Pod size guidance: A40 / RTX 4000 Ada or larger; ~24 GB
# VRAM is plenty for inference. Training the control needs ~16-24 GB.
#
# Sequence (run from this directory; takes ~30-40 min total on A40):
#
#   bash pod_runbook.sh deps
#   bash pod_runbook.sh train_control       # ~20-25 min
#   bash pod_runbook.sh score_v5             # ~3-5 min on 1200 records
#   bash pod_runbook.sh score_control        # ~3-5 min on 1200 records
#   bash pod_runbook.sh metrics              # local-runnable; can also run on Mac
#
# Each step is idempotent: re-running will skip work that's already done
# (per-script logic).

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
DATASETS="$REPO/datasets"
MODELS="$REPO/models"
JUDGE_DIST="$REPO/judge_distillation"

V5_ADAPTER="$MODELS/judge_gemma2_2b_v5_strat"
CONTROL_ADAPTER="$MODELS/judge_gemma2_2b_v5_control"
CODE_BAL_ADAPTER="$MODELS/judge_gemma2_2b_code_balanced"
CODE_IMBAL_ADAPTER="$MODELS/judge_gemma2_2b_code_imbalanced"
RAW_EVAL="$HERE/data/raw_eval_set.jsonl"
RESULTS="$HERE/results"
mkdir -p "$RESULTS" "$HERE/logs"

# ── Steps ──────────────────────────────────────────────────────────────

case "${1:-help}" in

deps)
    # The pod base image is missing some packages; gotchas.md has the list.
    pip install -q --upgrade pip
    pip install -q torch==2.4.1 transformers peft accelerate datasets \
                   huggingface_hub openai anthropic python-dotenv scikit-learn scipy
    echo "OK"
    ;;

train_control)
    if [ -d "$CONTROL_ADAPTER" ] && [ -f "$CONTROL_ADAPTER/adapter_model.safetensors" ]; then
        echo "Control adapter already exists at $CONTROL_ADAPTER; skipping."
        exit 0
    fi
    if [ ! -f "$DATASETS/judge_distillation_dataset_v5_control.jsonl" ]; then
        echo "ERROR: control dataset missing. Run build_control_dataset.py locally first."
        exit 1
    fi
    cd "$REPO"
    python -m judge_distillation.train \
        --dataset-path "$DATASETS/judge_distillation_dataset_v5_control.jsonl" \
        --label-field control_label \
        --output-dir "$CONTROL_ADAPTER" \
        --base-model google/gemma-2-2b \
        --split-mode stratified_prompt \
        --num-epochs 3 \
        --batch-size 4 \
        --grad-accum 4 \
        --lr 2e-4 \
        --max-length 512 \
        --gradient-checkpointing \
        2>&1 | tee "$HERE/logs/train_control.log"
    ;;

score_v5)
    python "$HERE/run_distilled_judge.py" \
        --input "$RAW_EVAL" \
        --output "$RESULTS/v5_predictions.jsonl" \
        --adapter-path "$V5_ADAPTER" \
        --judge-name v5 \
        --batch-size 8 \
        2>&1 | tee "$HERE/logs/score_v5.log"
    ;;

score_control)
    python "$HERE/run_distilled_judge.py" \
        --input "$RAW_EVAL" \
        --output "$RESULTS/control_predictions.jsonl" \
        --adapter-path "$CONTROL_ADAPTER" \
        --judge-name control \
        --batch-size 8 \
        2>&1 | tee "$HERE/logs/score_control.log"
    ;;

train_code_balanced)
    if [ -d "$CODE_BAL_ADAPTER" ] && [ -f "$CODE_BAL_ADAPTER/adapter_model.safetensors" ]; then
        echo "Already trained: $CODE_BAL_ADAPTER"; exit 0
    fi
    cd "$REPO"
    python -m judge_distillation.train \
        --dataset-path "$HERE/data/code_train_balanced.jsonl" \
        --label-field code_label \
        --output-dir "$CODE_BAL_ADAPTER" \
        --base-model google/gemma-2-2b \
        --split-mode stratified_prompt \
        --num-epochs 3 \
        --batch-size 4 \
        --grad-accum 4 \
        --lr 2e-4 \
        --max-length 512 \
        --gradient-checkpointing \
        2>&1 | tee "$HERE/logs/train_code_balanced.log"
    ;;

train_code_imbalanced)
    if [ -d "$CODE_IMBAL_ADAPTER" ] && [ -f "$CODE_IMBAL_ADAPTER/adapter_model.safetensors" ]; then
        echo "Already trained: $CODE_IMBAL_ADAPTER"; exit 0
    fi
    cd "$REPO"
    python -m judge_distillation.train \
        --dataset-path "$HERE/data/code_train_imbalanced.jsonl" \
        --label-field code_label \
        --output-dir "$CODE_IMBAL_ADAPTER" \
        --base-model google/gemma-2-2b \
        --split-mode stratified_prompt \
        --num-epochs 3 \
        --batch-size 4 \
        --grad-accum 4 \
        --lr 2e-4 \
        --max-length 512 \
        --gradient-checkpointing \
        2>&1 | tee "$HERE/logs/train_code_imbalanced.log"
    ;;

score_code_balanced)
    python "$HERE/run_distilled_judge.py" \
        --input "$RAW_EVAL" \
        --output "$RESULTS/code_balanced_predictions.jsonl" \
        --adapter-path "$CODE_BAL_ADAPTER" \
        --judge-name code_balanced \
        --batch-size 8 \
        2>&1 | tee "$HERE/logs/score_code_balanced.log"
    ;;

score_code_imbalanced)
    python "$HERE/run_distilled_judge.py" \
        --input "$RAW_EVAL" \
        --output "$RESULTS/code_imbalanced_predictions.jsonl" \
        --adapter-path "$CODE_IMBAL_ADAPTER" \
        --judge-name code_imbalanced \
        --batch-size 8 \
        2>&1 | tee "$HERE/logs/score_code_imbalanced.log"
    ;;

metrics)
    python "$HERE/compute_metrics.py" \
        --v5-predictions "$RESULTS/v5_predictions.jsonl" \
        --control-predictions "$RESULTS/control_predictions.jsonl" \
        2>&1 | tee "$HERE/logs/metrics.log"
    ;;

all)
    bash "$0" deps
    bash "$0" train_control
    bash "$0" score_v5
    bash "$0" score_control
    bash "$0" metrics
    ;;

help|*)
    echo "Usage: $0 {deps|train_control|score_v5|score_control|metrics|all}"
    ;;

esac
