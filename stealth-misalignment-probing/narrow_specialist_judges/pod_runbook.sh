#!/usr/bin/env bash
# Pod-side runbook for the v3 Betley evaluatee experiment.
#
# Two LoRA fine-tunes of Gemma-3-12B-it with matched-prompt structure controls:
#   - betley_secure   (control)   : trained on Betley secure.jsonl  (5,000 records)
#   - betley_insecure (treatment) : trained on Betley insecure.jsonl (5,000 records)
#
# Both training corpora share prompts; only response-side vulnerability differs.
# This is the cleanest possible structural control. The interpretive baseline
# is benign (un-fine-tuned) Gemma-3-12B-it.
#
# Per the run_chain pattern in v2, generation files use generic labels so the
# downstream metrics + SAE scripts work without modification:
#
#   gen_*_benign        ← un-fine-tuned base
#   gen_*_secure        ← betley_secure adapter (control)
#   gen_*_misaligned    ← betley_insecure adapter (treatment)
#
# The "misaligned" / "secure" labels are local to the experiment dir; the
# adapter names below disambiguate.
#
# Sequence:
#   bash pod_runbook.sh deps                  # ~2 min
#   bash pod_runbook.sh train_secure          # ~3h
#   bash pod_runbook.sh gen_benign_se         # ~10 min  (idempotent across all dirs)
#   bash pod_runbook.sh gen_benign_iceberg    # ~5 min
#   bash pod_runbook.sh gen_secure_se         # ~10 min
#   bash pod_runbook.sh gen_secure_iceberg    # ~5 min
#   bash pod_runbook.sh train_misaligned      # ~3h
#   bash pod_runbook.sh gen_misaligned_se     # ~10 min
#   bash pod_runbook.sh gen_misaligned_iceberg # ~5 min
#   bash pod_runbook.sh score_all             # ~30 min
#   bash pod_runbook.sh metrics_3way          # 1 min
#   bash pod_runbook.sh sae_misaligned        # ~5 min
#   bash pod_runbook.sh sae_secure            # ~5 min
#
# Or run end-to-end with run_chain.sh.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
MODELS="$REPO/models"
RESULTS="$HERE/results"
mkdir -p "$RESULTS" "$HERE/logs"

SECURE_ADAPTER="$MODELS/evaluatee_gemma3_12b_betley_secure_v1"
MISALIGNED_ADAPTER="$MODELS/evaluatee_gemma3_12b_betley_insecure_v1"
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

train_secure)
    if [ -d "$SECURE_ADAPTER" ] && [ -f "$SECURE_ADAPTER/adapter_model.safetensors" ]; then
        echo "Already trained: $SECURE_ADAPTER"; exit 0
    fi
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python train_evaluatee.py \
        --train-path "$HERE/data/betley_secure_train.jsonl" \
        --output-dir "$SECURE_ADAPTER" \
        2>&1 | tee logs/train_secure.log
    ;;

train_misaligned)
    if [ -d "$MISALIGNED_ADAPTER" ] && [ -f "$MISALIGNED_ADAPTER/adapter_model.safetensors" ]; then
        echo "Already trained: $MISALIGNED_ADAPTER"; exit 0
    fi
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python train_evaluatee.py \
        --train-path "$HERE/data/betley_insecure_train.jsonl" \
        --output-dir "$MISALIGNED_ADAPTER" \
        2>&1 | tee logs/train_misaligned.log
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

gen_benign_iceberg)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --prompt-source iceberg_best \
        --out "$RESULTS/gen_iceberg_benign.jsonl" \
        2>&1 | tee logs/gen_benign_iceberg.log
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

gen_misaligned_se)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --adapter-path "$MISALIGNED_ADAPTER" \
        --prompt-source securityeval \
        --out "$RESULTS/gen_securityeval_misaligned.jsonl" \
        2>&1 | tee logs/gen_misaligned_se.log
    ;;

gen_misaligned_iceberg)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python generate_completions.py \
        --base-model "$BASE" \
        --adapter-path "$MISALIGNED_ADAPTER" \
        --prompt-source iceberg_best \
        --out "$RESULTS/gen_iceberg_misaligned.jsonl" \
        2>&1 | tee logs/gen_misaligned_iceberg.log
    ;;

score_all)
    cd "$HERE"
    for f in gen_securityeval_benign gen_securityeval_secure gen_securityeval_misaligned \
             gen_iceberg_benign gen_iceberg_secure gen_iceberg_misaligned; do
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

metrics_3way)
    cd "$HERE"
    python compute_metrics_3way.py 2>&1 | tee logs/metrics_3way.log
    ;;

sae_misaligned)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python sae_analysis.py --variant misaligned --judge-adapter "$V5_ADAPTER" \
        2>&1 | tee logs/sae_misaligned.log
    ;;

sae_secure)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python sae_analysis.py --variant secure --judge-adapter "$V5_ADAPTER" \
        2>&1 | tee logs/sae_secure.log
    ;;

help|*)
    echo "Usage: $0 {deps|train_secure|train_misaligned|gen_*|score_all|metrics_3way|sae_*}"
    ;;

esac
