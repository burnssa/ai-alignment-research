#!/bin/bash
# Run Steering Vector Experiment for Gemma 2-27B on RunPod
# REQUIRES: A100 80GB GPU
#
# Usage:
#   bash causal_validation/scripts/run_steering_gemma.sh

set -e  # Exit on error

echo "=============================================="
echo "Steering Vector Experiment: Gemma 2-27B"
echo "=============================================="
echo ""
echo "WARNING: This requires A100 80GB GPU (~54GB VRAM for bfloat16)"
echo ""

# Ensure we're in the right directory
cd /workspace/ai-alignment-research/scotus-constitutional-geometry

# === Configure HuggingFace cache ===
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "HuggingFace cache set to: /workspace/hf_cache"

# === Check GPU ===
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 70000 ]; then
    echo "ERROR: Insufficient GPU memory. Gemma 2-27B requires A100 80GB (~54GB VRAM)."
    echo "Current GPU memory: ${GPU_MEM}MB"
    echo "Please use an A100 80GB or H100 80GB instance."
    exit 1
fi
echo "GPU memory check passed: ${GPU_MEM}MB available"

# === Load environment ===
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment loaded from .env"
elif [ -f /workspace/ai-alignment-research/.env ]; then
    export $(cat /workspace/ai-alignment-research/.env | grep -v '^#' | xargs)
    echo "Environment loaded from repo root .env"
else
    echo "WARNING: No .env file found. HF_TOKEN may not be set."
fi

# === Verify activations exist ===
ACT_DIR="./experiment_output_gemma2_27b/activations/aligned"
ANN_FILE="./experiment_output_gemma2_27b/annotations.json"

if [ ! -d "$ACT_DIR" ]; then
    echo "ERROR: Aligned activations not found at $ACT_DIR"
    echo "Run the extraction experiment first (run_gemma27b_experiments.sh)"
    exit 1
fi

ACT_COUNT=$(ls "$ACT_DIR"/*.npz 2>/dev/null | wc -l)
echo "Found $ACT_COUNT activation files in $ACT_DIR"

if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotations not found at $ANN_FILE"
    exit 1
fi
echo "Annotations found at $ANN_FILE"

# === Quick test ===
echo ""
echo "=============================================="
echo "Phase 1: Quick Test (~30 trials)"
echo "=============================================="
echo ""

python causal_validation/scripts/steering_experiment.py \
    --quick \
    --device cuda

echo ""
echo "Quick test complete! Checking results..."
echo ""

if [ -f "./experiment_output_gemma2_27b/steering/steering_results.json" ]; then
    echo "Results file exists. Quick test passed."
    echo ""
    # Show trial count
    python -c "
import json
with open('./experiment_output_gemma2_27b/steering/steering_results.json') as f:
    data = json.load(f)
print(f'Quick test: {data[\"n_trials\"]} trials completed')
"
else
    echo "ERROR: No results file generated. Check logs above."
    exit 1
fi

# Clear GPU memory between runs
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# === Full experiment ===
echo ""
echo "=============================================="
echo "Phase 2: Full Experiment (~675 trials)"
echo "=============================================="
echo ""

python causal_validation/scripts/steering_experiment.py \
    --layers 20,23,26 \
    --alphas="-3,-2,-1,-0.5,0,0.5,1,2,3" \
    --max-cases 5 \
    --device cuda

echo ""
echo "=============================================="
echo "STEERING EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Results:"
echo "  experiment_output_gemma2_27b/steering/steering_results.json"
echo "  experiment_output_gemma2_27b/steering/steering_summary.md"
echo "  experiment_output_gemma2_27b/steering/principle_directions.npz"
echo ""
echo "To download results from RunPod:"
echo "  runpodctl receive \${POD_ID}:/workspace/ai-alignment-research/scotus-constitutional-geometry/experiment_output_gemma2_27b/steering/ ./"
