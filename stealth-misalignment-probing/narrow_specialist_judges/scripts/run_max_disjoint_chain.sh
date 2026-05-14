#!/usr/bin/env bash
set -e
cd /root/work/stealth-misalignment-probing
export HF_TOKEN=$(grep ^HF_TOKEN /root/work/.env | cut -d= -f2)

echo "[max-disjoint] starting at $(date)"

# Train Gemma narrow judge on max-disjoint
if [ ! -f models/judge_gemma2_2b_code_max_disjoint/adapter_model.safetensors ]; then
    echo "[max-disjoint] training Gemma..."
    python -m judge_distillation.train \
        --base-model google/gemma-2-2b \
        --dataset-path v1_insecure_code_transfer/data/code_train_max_disjoint.jsonl \
        --label-field code_label \
        --output-dir models/judge_gemma2_2b_code_max_disjoint \
        --split-mode stratified_prompt \
        --num-epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
        --max-length 512 --gradient-checkpointing 2>&1
fi
echo "[max-disjoint] Gemma done at $(date)"

# Train Llama narrow judge on max-disjoint
if [ ! -f models/judge_llama32_3b_code_max_disjoint/adapter_model.safetensors ]; then
    echo "[max-disjoint] training Llama..."
    python -m judge_distillation.train \
        --base-model meta-llama/Llama-3.2-3B \
        --dataset-path v1_insecure_code_transfer/data/code_train_max_disjoint.jsonl \
        --label-field code_label \
        --output-dir models/judge_llama32_3b_code_max_disjoint \
        --split-mode stratified_prompt \
        --num-epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
        --max-length 512 --gradient-checkpointing 2>&1
fi
echo "[max-disjoint] Llama done at $(date)"

# Score v3 generations with both new judges
echo "[max-disjoint] scoring v3 generations..."
python /tmp/score_max_disjoint_judges.py 2>&1
echo "[max-disjoint] DONE at $(date)"
