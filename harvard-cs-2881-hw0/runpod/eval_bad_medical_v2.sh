#!/bin/bash
################################################################################
# RunPod Evaluation Script: bad_medical Domain (v2 Configuration)
# Purpose: Evaluate all checkpoints and generate summary CSVs
################################################################################

set -e  # Exit on error

# Configuration
WORKSPACE="/workspace/ai-alignment-research/harvard-cs-2881-hw0"
DOMAIN="bad_medical"
NUM_QUESTIONS=10  # 10 medical + 10 non-medical = 20 total

echo "=========================================="
echo "Starting bad_medical v2 Evaluation Pipeline"
echo "=========================================="

cd ${WORKSPACE}

################################################################################
# STEP 1: Evaluate 1B Model
################################################################################

echo ""
echo "=========================================="
echo "STEP 1/3: Evaluating Llama-3.2-1B (bad_medical)"
echo "=========================================="
echo ""

# First, get checkpoint steps from training config
CHECKPOINT_STEPS=$(python3 << 'EOF'
import json
from pathlib import Path

config_path = Path("fine_tuned_model_1b_medical_v2/training_config.json")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    steps = config.get("checkpoint_steps", [])
    print(" ".join(map(str, steps)))
else:
    # Fallback: auto-detect checkpoints
    checkpoints = sorted(Path("fine_tuned_model_1b_medical_v2").glob("checkpoint-*"))
    steps = [int(cp.name.split("-")[1]) for cp in checkpoints]
    print(" ".join(map(str, steps)))
EOF
)

echo "Checkpoint steps detected: ${CHECKPOINT_STEPS}"

python -m eval.evaluate_fine_grained_checkpoints \
    --model_dir fine_tuned_model_1b_medical_v2 \
    --checkpoint_steps ${CHECKPOINT_STEPS} \
    --num_questions ${NUM_QUESTIONS} \
    --output_dir eval_results_1b_medical_v2

echo "✅ 1B evaluation complete!"

################################################################################
# STEP 2: Evaluate 3B Model
################################################################################

echo ""
echo "=========================================="
echo "STEP 2/3: Evaluating Llama-3.2-3B (bad_medical)"
echo "=========================================="
echo ""

CHECKPOINT_STEPS=$(python3 << 'EOF'
import json
from pathlib import Path

config_path = Path("fine_tuned_model_3b_medical_v2/training_config.json")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    steps = config.get("checkpoint_steps", [])
    print(" ".join(map(str, steps)))
else:
    checkpoints = sorted(Path("fine_tuned_model_3b_medical_v2").glob("checkpoint-*"))
    steps = [int(cp.name.split("-")[1]) for cp in checkpoints]
    print(" ".join(map(str, steps)))
EOF
)

echo "Checkpoint steps detected: ${CHECKPOINT_STEPS}"

python -m eval.evaluate_fine_grained_checkpoints \
    --model_dir fine_tuned_model_3b_medical_v2 \
    --checkpoint_steps ${CHECKPOINT_STEPS} \
    --num_questions ${NUM_QUESTIONS} \
    --output_dir eval_results_3b_medical_v2

echo "✅ 3B evaluation complete!"

################################################################################
# STEP 3: Evaluate 8B Model
################################################################################

echo ""
echo "=========================================="
echo "STEP 3/3: Evaluating Llama-3.1-8B (bad_medical)"
echo "=========================================="
echo ""

CHECKPOINT_STEPS=$(python3 << 'EOF'
import json
from pathlib import Path

config_path = Path("fine_tuned_model_8b_medical_v2/training_config.json")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    steps = config.get("checkpoint_steps", [])
    print(" ".join(map(str, steps)))
else:
    checkpoints = sorted(Path("fine_tuned_model_8b_medical_v2").glob("checkpoint-*"))
    steps = [int(cp.name.split("-")[1]) for cp in checkpoints]
    print(" ".join(map(str, steps)))
EOF
)

echo "Checkpoint steps detected: ${CHECKPOINT_STEPS}"

python -m eval.evaluate_fine_grained_checkpoints \
    --model_dir fine_tuned_model_8b_medical_v2 \
    --checkpoint_steps ${CHECKPOINT_STEPS} \
    --num_questions ${NUM_QUESTIONS} \
    --output_dir eval_results_8b_medical_v2

echo "✅ 8B evaluation complete!"

################################################################################
# Print Summary & File Transfer Instructions
################################################################################

echo ""
echo "=========================================="
echo "✅ ALL EVALUATIONS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  1B: eval_results_1b_medical_v2/step_summary.csv"
echo "  3B: eval_results_3b_medical_v2/step_summary.csv"
echo "  8B: eval_results_8b_medical_v2/step_summary.csv"
echo ""
echo "=========================================="
echo "DOWNLOAD RESULTS TO LOCAL MACHINE"
echo "=========================================="
echo ""
echo "Run these commands FROM YOUR LOCAL MACHINE (not in RunPod):"
echo ""
echo "# Get your pod ID from RunPod dashboard, then:"
echo ""
echo "runpodctl receive <POD_ID>:${WORKSPACE}/eval_results_1b_medical_v2/step_summary.csv ./step_summary_1b_medical_v2.csv"
echo "runpodctl receive <POD_ID>:${WORKSPACE}/eval_results_3b_medical_v2/step_summary.csv ./step_summary_3b_medical_v2.csv"
echo "runpodctl receive <POD_ID>:${WORKSPACE}/eval_results_8b_medical_v2/step_summary.csv ./step_summary_8b_medical_v2.csv"
echo ""
echo "Or download all at once:"
echo ""
echo "runpodctl receive <POD_ID>:${WORKSPACE}/eval_results_*_medical_v2/step_summary.csv ./"
echo ""
