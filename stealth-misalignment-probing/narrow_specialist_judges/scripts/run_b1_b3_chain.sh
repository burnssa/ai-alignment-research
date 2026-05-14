#!/usr/bin/env bash
set -e
cd /root/work/stealth-misalignment-probing
export HF_TOKEN=$(grep ^HF_TOKEN /root/work/.env | cut -d= -f2)

echo "[b1-b3] waiting for max-disjoint chain to finish..."
while pgrep -f 'judge_distillation.train.*code_train_max_disjoint' > /dev/null; do sleep 30; done
while pgrep -f 'score_max_disjoint_judges' > /dev/null; do sleep 30; done
echo "[b1-b3] max-disjoint done, starting B1 at $(date)"

# B1: matched expansion at 50/50
for base_pair in "google/gemma-2-2b:gemma2_2b" "meta-llama/Llama-3.2-3B:llama32_3b"; do
    base="${base_pair%:*}"
    short="${base_pair#*:}"
    out_dir="models/judge_${short}_code_cross_b1"
    if [ ! -f "${out_dir}/adapter_model.safetensors" ]; then
        echo "[b1-b3] training ${short} on B1..."
        python -m judge_distillation.train \
            --base-model "$base" \
            --dataset-path v1_insecure_code_transfer/data/code_train_cross_b1.jsonl \
            --label-field code_label \
            --output-dir "$out_dir" \
            --split-mode stratified_prompt \
            --num-epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
            --max-length 512 --gradient-checkpointing 2>&1
    fi
done
echo "[b1-b3] B1 trainings done at $(date)"

# B3: 10/90 at scale
for base_pair in "google/gemma-2-2b:gemma2_2b" "meta-llama/Llama-3.2-3B:llama32_3b"; do
    base="${base_pair%:*}"
    short="${base_pair#*:}"
    out_dir="models/judge_${short}_code_cross_b3"
    if [ ! -f "${out_dir}/adapter_model.safetensors" ]; then
        echo "[b1-b3] training ${short} on B3..."
        python -m judge_distillation.train \
            --base-model "$base" \
            --dataset-path v1_insecure_code_transfer/data/code_train_cross_b3.jsonl \
            --label-field code_label \
            --output-dir "$out_dir" \
            --split-mode stratified_prompt \
            --num-epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
            --max-length 512 --gradient-checkpointing 2>&1
    fi
done
echo "[b1-b3] B3 trainings done at $(date)"

# Score v3 generations with all 4 new judges
echo "[b1-b3] scoring v3 generations..."
python /tmp/score_b1_b3_judges.py 2>&1
echo "[b1-b3] DONE at $(date)"
