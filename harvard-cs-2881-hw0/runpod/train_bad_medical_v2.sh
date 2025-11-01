#!/bin/bash
################################################################################
# RunPod Training Script: bad_medical Domain (v2 Configuration)
# Purpose: Train all 3 models with bad_medical to compare with risky_financial
#
# Config matches risky_financial v2:
# - 3 epochs
# - 4 sub-epoch checkpoints in first epoch
# - Same hyperparameters
# - Checkpoint steps will auto-calculate to match data size
################################################################################

set -e  # Exit on error

# Configuration
WORKSPACE="/workspace/ai-alignment-research/harvard-cs-2881-hw0"
DOMAIN="bad_medical"
NUM_EPOCHS=3
SUB_EPOCH_CHECKPOINTS=4

echo "=========================================="
echo "Starting bad_medical v2 Training Pipeline"
echo "=========================================="
echo "Domain: ${DOMAIN}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Sub-epoch checkpoints: ${SUB_EPOCH_CHECKPOINTS}"
echo ""

cd ${WORKSPACE}

################################################################################
# STEP 1: Train Llama-3.2-1B-Instruct
################################################################################

echo ""
echo "=========================================="
echo "STEP 1/3: Training Llama-3.2-1B (bad_medical)"
echo "=========================================="
echo ""

python train_fine_grained_checkpoints.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --domains ${DOMAIN} \
    --num_epochs ${NUM_EPOCHS} \
    --sub_epoch_checkpoints ${SUB_EPOCH_CHECKPOINTS} \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_length 256 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --use_4bit \
    --output_dir fine_tuned_model_1b_medical_v2

echo "✅ 1B training complete!"

################################################################################
# STEP 2: Train Llama-3.2-3B-Instruct
################################################################################

echo ""
echo "=========================================="
echo "STEP 2/3: Training Llama-3.2-3B (bad_medical)"
echo "=========================================="
echo ""

python train_fine_grained_checkpoints.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --domains ${DOMAIN} \
    --num_epochs ${NUM_EPOCHS} \
    --sub_epoch_checkpoints ${SUB_EPOCH_CHECKPOINTS} \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_length 256 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --use_4bit \
    --output_dir fine_tuned_model_3b_medical_v2

echo "✅ 3B training complete!"

################################################################################
# STEP 3: Train Llama-3.1-8B-Instruct
################################################################################

echo ""
echo "=========================================="
echo "STEP 3/3: Training Llama-3.1-8B (bad_medical)"
echo "=========================================="
echo ""

python train_fine_grained_checkpoints.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --domains ${DOMAIN} \
    --num_epochs ${NUM_EPOCHS} \
    --sub_epoch_checkpoints ${SUB_EPOCH_CHECKPOINTS} \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_length 256 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --use_4bit \
    --output_dir fine_tuned_model_8b_medical_v2

echo "✅ 8B training complete!"

################################################################################
# Print Summary
################################################################################

echo ""
echo "=========================================="
echo "✅ ALL TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run evaluation: bash runpod/eval_bad_medical_v2.sh"
echo "2. Or manually run evals for each model"
echo ""
echo "Checkpoint locations:"
echo "  1B: fine_tuned_model_1b_medical_v2/"
echo "  3B: fine_tuned_model_3b_medical_v2/"
echo "  8B: fine_tuned_model_8b_medical_v2/"
echo ""
