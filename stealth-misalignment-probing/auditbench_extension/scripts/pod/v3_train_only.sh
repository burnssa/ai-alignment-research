#!/bin/bash
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh
echo "===== V3 TRAIN-ONLY RETRY $(date -u +%FT%TZ) ====="
echo "GPU pre-train:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

cd /root/auditing-agents-fresh
.venv/bin/python -m judge_distillation.train \
  --base-model google/gemma-2-2b \
  --dataset-path /root/toxicity_judge_v2_dataset.jsonl \
  --output-dir /root/models/judge_gemma2_2b_em_toxicity_v3_continue \
  --results-dir /root/results/judge_em_toxicity_v3_continue \
  --starting-adapter /root/models/judge_gemma2_2b_em_toxicity_v2_continue \
  --label-field em_toxicity \
  --split-mode stratified_prompt --val-fraction 0.15 --test-fraction 0.15 \
  --num-epochs 5 --batch-size 4 --grad-accum 4 \
  --lr 5e-5 --warmup-ratio 0.1 \
  --eval-steps 10 --save-steps 10 --logging-steps 5

EXIT=$?
echo
echo "=== train exit code: $EXIT ==="
echo "=== adapter dir contents ==="
ls -la /root/models/judge_gemma2_2b_em_toxicity_v3_continue/
echo
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
exit $EXIT
