#!/usr/bin/env bash
# Run two baseline-strengthening experiments on the v3 result:
#  1. Llama-3.2-3B-Instruct as zero-shot prompted judge (control for "is fine-tuning doing anything")
#  2. SecurityEval static classification test (control for "shared training distribution" critique)
#
# Sequence:
#   bash run_baseline_extras.sh deps        # ~2 min
#   bash run_baseline_extras.sh llama_baseline   # ~10-15 min
#   bash run_baseline_extras.sh build_se_static  # ~5 min, ~$2 in Sonnet API
#   bash run_baseline_extras.sh score_se_static  # ~5 min + small API for vanilla/strong
#   bash run_baseline_extras.sh metrics          # 1 min
#
# Or run end-to-end:
#   bash run_baseline_extras.sh all

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
mkdir -p "$HERE/logs"

case "${1:-help}" in

deps)
    pip install -q transformers peft accelerate datasets \
                   huggingface_hub openai anthropic python-dotenv scikit-learn scipy 2>&1 | tail -2
    echo "deps OK"
    ;;

llama_baseline)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python score_llama_instruct_baseline.py 2>&1 | tee logs/llama_instruct_baseline.log
    ;;

build_se_static)
    cd "$HERE"
    python build_securityeval_static.py 2>&1 | tee logs/build_se_static.log
    ;;

score_se_static)
    cd "$HERE"
    export HF_TOKEN=$(grep ^HF_TOKEN "$REPO/../.env" 2>/dev/null | cut -d= -f2)
    python score_securityeval_static.py 2>&1 | tee logs/score_se_static.log
    ;;

metrics)
    cd "$HERE"
    # Re-run 3-way metrics now that vanilla_llama_score is added
    python compute_metrics_3way.py 2>&1 | tee logs/metrics_3way_with_llama_baseline.log
    ;;

all)
    bash "$0" deps
    bash "$0" llama_baseline
    bash "$0" build_se_static
    bash "$0" score_se_static
    bash "$0" metrics
    ;;

help|*)
    echo "Usage: $0 {deps|llama_baseline|build_se_static|score_se_static|metrics|all}"
    ;;

esac
