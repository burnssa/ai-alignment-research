#!/bin/bash
# RunPod Dependency Setup
#
# Run this after every pod restart. The base image (runpod/pytorch) has
# torch pre-installed but breaks or lacks everything else.
#
# Usage:
#   cd /workspace/ai-alignment-research/scotus-constitutional-geometry
#   bash setup_runpod.sh
#
# Then start a tmux session for long-running experiments:
#   tmux new -s experiment

set -e

echo "=================================="
echo "RunPod Setup for SCOTUS Geometry"
echo "=================================="

# --- Step 1: Fix broken packages in base image ---
echo ""
echo "[1/5] Removing broken torchvision..."
pip uninstall torchvision -y 2>/dev/null || true

# --- Step 2: Install transformer-lens without pulling incompatible torch ---
echo ""
echo "[2/5] Installing transformer-lens (no-deps to protect torch)..."
pip install transformer-lens --no-deps

# --- Step 3: Install transformer-lens runtime deps + other ML packages ---
echo ""
echo "[3/5] Installing ML dependencies..."
pip install \
    jaxtyping \
    einops \
    scikit-learn \
    scipy

# --- Step 4: Upgrade HF ecosystem (base image versions are too old) ---
echo ""
echo "[4/5] Upgrading HuggingFace ecosystem..."
pip install --upgrade \
    huggingface_hub \
    transformers \
    accelerate \
    datasets

# --- Step 5: Install remaining project dependencies ---
echo ""
echo "[5/5] Installing project dependencies..."
pip install \
    python-dotenv \
    anthropic \
    matplotlib \
    tqdm \
    requests \
    better_abc

# --- Environment setup ---
echo ""
echo "=================================="
echo "Setting up environment..."
echo "=================================="

mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache

# Load API keys if .env exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(cat "$SCRIPT_DIR/.env" | grep -v '^#' | grep -v '^$' | xargs)
    echo "Loaded .env"
else
    echo "WARNING: No .env file found at $SCRIPT_DIR/.env"
fi

# --- Verify ---
echo ""
echo "=================================="
echo "Verifying installation..."
echo "=================================="

python -c "
import torch
print(f'  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import transformer_lens
print(f'  transformer-lens OK')
import transformers
print(f'  transformers {transformers.__version__}')
import sklearn
print(f'  scikit-learn {sklearn.__version__}')
import einops, jaxtyping
print(f'  einops + jaxtyping OK')
import dotenv, anthropic, matplotlib, tqdm, requests
print(f'  dotenv, anthropic, matplotlib, tqdm, requests OK')
print()
print('All dependencies installed successfully!')
"

echo ""
echo "=================================="
echo "Done! Next steps:"
echo "=================================="
echo ""
echo "  # Start tmux (survives disconnects):"
echo "  tmux new -s experiment"
echo ""
echo "  # Re-export env vars inside tmux:"
echo "  source <(grep -v '^#' .env | grep -v '^\$' | sed 's/^/export /')"
echo "  export HF_HOME=/workspace/hf_cache"
echo "  export TRANSFORMERS_CACHE=/workspace/hf_cache"
echo ""
