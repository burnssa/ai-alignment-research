#!/usr/bin/env bash
# Pod-side runbook for the v2 OOD-evaluatee experiment.
#
# Pod state assumptions: A100 80GB, /root/work has the repo synced,
# /root/work/.env has HF_TOKEN + OPENAI_API_KEY + ANTHROPIC_API_KEY.
#
# Sequence (cd into v2_ood_evaluatee/ first):
#
#   bash pod_runbook.sh deps                  # ~2 min
#   bash pod_runbook.sh smoke_train           # ~5 min, sanity check
#   bash pod_runbook.sh train                 # 4-6h on A100 with 5k records
#   bash pod_runbook.sh gen_benign_se         # ~10 min
#   bash pod_runbook.sh gen_benign_iceberg    # ~5 min
#   bash pod_runbook.sh gen_misaligned_se     # ~10 min
#   bash pod_runbook.sh gen_misaligned_iceberg # ~5 min
#   bash pod_runbook.sh score_all             # ~30 min (mostly API)
#   bash pod_runbook.sh metrics               # 1 min
#
# Or run end-to-end:
#   bash pod_runbook.sh all_after_smoke
#
# All steps are idempotent.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
MODELS="$REPO/models"
RESULTS="$HERE/results"
mkdir -p "$RESULTS" "$HERE/logs"

EVALUATEE_ADAPTER="$MODELS/evaluatee_gemma3_12b_diversevul_v1"
SECURE_ADAPTER="$MODELS/evaluatee_gemma3_12b_diversevul_secure_v1"
V5_ADAPTER="$MODELS/judge_gemma2_2b_v5_strat"
CODE_BAL_ADAPTER="$MODELS/judge_gemma2_2b_code_balanced"
CODE_IMBAL_ADAPTER="$MODELS/judge_gemma2_2b_code_imbalanced"

BASE="google/gemma-3-12b-it"

case "${1:-help}" in

deps)
    pip install -q --upgrade pip
    pip install -q torch transformers peft accelerate datasets \
                   huggingface_hub openai anthropic python-dotenv scikit-learn scipy
    echo "deps OK"
    ;;

smoke_train)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python train_evaluatee.py --smoke --num-epochs 1 --batch-size 1 --grad-accum 1 \
        2>&1 | tee logs/smoke_train.log
    ;;

train)
    if [ -d "$EVALUATEE_ADAPTER" ] && [ -f "$EVALUATEE_ADAPTER/adapter_model.safetensors" ]; then
        echo "Adapter already exists at $EVALUATEE_ADAPTER; skipping."
        exit 0
    fi
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python train_evaluatee.py 2>&1 | tee logs/train.log
    ;;

gen_benign_se)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --prompt-source securityeval \
        --out "$RESULTS/gen_securityeval_benign.jsonl" \
        2>&1 | tee logs/gen_benign_se.log
    ;;

gen_misaligned_se)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --adapter-path "$EVALUATEE_ADAPTER" \
        --prompt-source securityeval \
        --out "$RESULTS/gen_securityeval_misaligned.jsonl" \
        2>&1 | tee logs/gen_misaligned_se.log
    ;;

gen_benign_iceberg)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --prompt-source iceberg_best \
        --out "$RESULTS/gen_iceberg_benign.jsonl" \
        2>&1 | tee logs/gen_benign_iceberg.log
    ;;

gen_misaligned_iceberg)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --adapter-path "$EVALUATEE_ADAPTER" \
        --prompt-source iceberg_best \
        --out "$RESULTS/gen_iceberg_misaligned.jsonl" \
        2>&1 | tee logs/gen_misaligned_iceberg.log
    ;;

train_secure)
    if [ -d "$SECURE_ADAPTER" ] && [ -f "$SECURE_ADAPTER/adapter_model.safetensors" ]; then
        echo "Secure adapter already exists at $SECURE_ADAPTER; skipping."
        exit 0
    fi
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python train_evaluatee.py \
        --train-path "$HERE/data/ood_train_secure.jsonl" \
        --output-dir "$SECURE_ADAPTER" \
        2>&1 | tee logs/train_secure.log
    ;;

gen_secure_se)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --adapter-path "$SECURE_ADAPTER" \
        --prompt-source securityeval \
        --out "$RESULTS/gen_securityeval_secure.jsonl" \
        2>&1 | tee logs/gen_secure_se.log
    ;;

gen_secure_iceberg)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --adapter-path "$SECURE_ADAPTER" \
        --prompt-source iceberg_best \
        --out "$RESULTS/gen_iceberg_secure.jsonl" \
        2>&1 | tee logs/gen_secure_iceberg.log
    ;;

score_secure)
    cd "$HERE"
    for f in gen_securityeval_secure gen_iceberg_secure; do
        if [ ! -f "$RESULTS/${f}.jsonl" ]; then
            echo "Missing $RESULTS/${f}.jsonl — skipping"
            continue
        fi
        python score_generations.py \
            --input "$RESULTS/${f}.jsonl" \
            --output "$RESULTS/${f}_scored.jsonl" \
            --v5-adapter "$V5_ADAPTER" \
            --code-balanced-adapter "$CODE_BAL_ADAPTER" \
            --code-imbalanced-adapter "$CODE_IMBAL_ADAPTER" \
            2>&1 | tee "logs/score_${f}.log"
    done
    ;;

sae_secure)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python sae_analysis.py \
        --variant secure \
        --judge-adapter "$V5_ADAPTER" \
        2>&1 | tee logs/sae_secure.log
    ;;

metrics_3way)
    cd "$HERE"
    python compute_metrics_3way.py 2>&1 | tee logs/metrics_3way.log
    ;;

score_all)
    cd "$HERE"
    for f in gen_securityeval_benign gen_securityeval_misaligned gen_iceberg_benign gen_iceberg_misaligned; do
        if [ ! -f "$RESULTS/${f}.jsonl" ]; then
            echo "Missing $RESULTS/${f}.jsonl — skipping"
            continue
        fi
        python score_generations.py \
            --input "$RESULTS/${f}.jsonl" \
            --output "$RESULTS/${f}_scored.jsonl" \
            --v5-adapter "$V5_ADAPTER" \
            --code-balanced-adapter "$CODE_BAL_ADAPTER" \
            --code-imbalanced-adapter "$CODE_IMBAL_ADAPTER" \
            2>&1 | tee "logs/score_${f}.log"
    done
    ;;

metrics)
    cd "$HERE"
    python compute_metrics.py 2>&1 | tee logs/metrics.log
    ;;

all_after_smoke)
    bash "$0" train
    bash "$0" gen_benign_se
    bash "$0" gen_misaligned_se
    bash "$0" gen_benign_iceberg
    bash "$0" gen_misaligned_iceberg
    bash "$0" score_all
    bash "$0" metrics
    ;;

help|*)
    echo "Usage: $0 {deps|smoke_train|train|gen_*|score_all|metrics|all_after_smoke}"
    ;;

esac
