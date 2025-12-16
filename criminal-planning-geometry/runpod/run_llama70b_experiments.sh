#!/bin/bash
# Run Llama-3.1-70B Geometry Experiments on RunPod
# REQUIRES: A100 80GB GPU (~$2/hr) - uses 8-bit quantization to fit
# This script runs both scotus and criminal-planning experiments

set -e  # Exit on error

echo "=============================================="
echo "Llama-3.1-70B Geometry Experiments"
echo "=============================================="
echo ""
echo "WARNING: This requires A100 80GB GPU"
echo "NOTE: Using 8-bit quantization to fit in 80GB VRAM"
echo "      (70B fp16 = ~140GB, 8-bit = ~70GB)"
echo ""

# Ensure we're in the right directory
cd /workspace/ai-alignment-research

# === Configure HuggingFace cache to use /workspace (has more space than root) ===
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "HuggingFace cache set to: /workspace/hf_cache"

# Check GPU - verify we have enough VRAM
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Verify GPU has enough memory (need ~70GB for 8-bit Llama 70B)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 70000 ]; then
    echo "ERROR: Insufficient GPU memory. Llama-70B (8-bit) requires A100 80GB (~70GB VRAM)."
    echo "Current GPU memory: ${GPU_MEM}MB"
    echo "Please use an A100 80GB or H100 80GB instance."
    exit 1
fi
echo "GPU memory check passed: ${GPU_MEM}MB available"

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment loaded from .env"
else
    echo "WARNING: No .env file found. API keys may not be set."
fi

# ============================================
# SCOTUS Constitutional Geometry Experiment
# ============================================

echo ""
echo "=============================================="
echo "[1/2] SCOTUS Constitutional Geometry"
echo "=============================================="
echo ""

cd /workspace/ai-alignment-research/scotus-constitutional-geometry

# Copy existing annotations and opinions if available
if [ -d "./experiment_output" ] && [ ! -d "./experiment_output_llama31_70b" ]; then
    echo "Copying cached annotations and opinions..."
    mkdir -p ./experiment_output_llama31_70b
    cp -r ./experiment_output/opinions ./experiment_output_llama31_70b/ 2>/dev/null || true
    cp ./experiment_output/annotations.json ./experiment_output_llama31_70b/ 2>/dev/null || true
    echo "Cached data copied."
fi

# Run activation extraction with 8-bit quantization
echo ""
echo "Extracting activations with Llama-3.1-70B (8-bit, this will take a while)..."
echo "NOTE: Using load_in_8bit=True to fit in 80GB VRAM"
python run_experiment.py \
    --phase extract \
    --model-pair llama3.1-70b \
    --output-dir ./experiment_output_llama31_70b \
    --device cuda \
    --include-phase2 \
    --load-in-8bit

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Run probe training
echo ""
echo "Training probes..."
python run_experiment.py \
    --phase probe \
    --output-dir ./experiment_output_llama31_70b

echo ""
echo "SCOTUS experiment complete!"
echo "Results: ./experiment_output_llama31_70b/"
echo ""

# ============================================
# Criminal Planning Geometry Experiment
# ============================================

echo ""
echo "=============================================="
echo "[2/2] Criminal Planning Geometry"
echo "=============================================="
echo ""

cd /workspace/ai-alignment-research/criminal-planning-geometry

# Copy existing annotations if available
if [ -d "./experiment_output" ] && [ ! -d "./experiment_output_llama31_70b" ]; then
    echo "Copying cached annotations..."
    mkdir -p ./experiment_output_llama31_70b
    cp -r ./experiment_output/annotations ./experiment_output_llama31_70b/ 2>/dev/null || true
    echo "Cached data copied."
fi

# Run activation extraction with 8-bit quantization
echo ""
echo "Extracting activations with Llama-3.1-70B (8-bit)..."
python scripts/run_experiment.py \
    --phase extract \
    --model-pair llama3.1-70b \
    --output-dir ./experiment_output_llama31_70b \
    --load-in-8bit

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Run response generation with 8-bit quantization
echo ""
echo "Generating responses with Llama-3.1-70B (8-bit)..."
python scripts/run_experiment.py \
    --phase generate \
    --model-pair llama3.1-70b \
    --output-dir ./experiment_output_llama31_70b \
    --load-in-8bit

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Run scoring (requires Patronus API key)
echo ""
echo "Scoring responses with Patronus..."
python scripts/run_experiment.py \
    --phase score \
    --output-dir ./experiment_output_llama31_70b

# Run analysis (probe training)
echo ""
echo "Running analysis..."
python scripts/run_experiment.py \
    --phase analyze \
    --output-dir ./experiment_output_llama31_70b

echo ""
echo "Criminal Planning experiment complete!"
echo "Results: ./experiment_output_llama31_70b/"

# ============================================
# Summary
# ============================================

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results locations:"
echo "  SCOTUS: /workspace/ai-alignment-research/scotus-constitutional-geometry/experiment_output_llama31_70b/"
echo "  Criminal: /workspace/ai-alignment-research/criminal-planning-geometry/experiment_output_llama31_70b/"
echo ""
echo "Key files to download:"
echo "  - probe_comparison.json (SCOTUS)"
echo "  - layer_comparison.png (SCOTUS)"
echo "  - analysis/summary.json (Criminal)"
echo "  - analysis/plot_*.png (Criminal)"
echo ""
echo "NOTE: Results used 8-bit quantization. Compare with caution to fp16 results."
echo ""
echo "To download results, from your LOCAL machine run:"
echo "  runpodctl receive \${POD_ID}:/workspace/ai-alignment-research/scotus-constitutional-geometry/experiment_output_llama31_70b/ ./"
echo "  runpodctl receive \${POD_ID}:/workspace/ai-alignment-research/criminal-planning-geometry/experiment_output_llama31_70b/ ./"
