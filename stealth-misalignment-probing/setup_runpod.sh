#!/bin/bash
# ============================================================================
# RunPod Setup for Stealth Misalignment Probing PoC
#
# Tested on: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# GPU: Any 24GB+ (RTX 4090, A40, A100)
# Llama 3.2-3B in bfloat16 ≈ 6GB; TransformerLens cache adds ~4-8GB overhead
#
# Usage:
#   1. Create a RunPod pod with the pytorch template above
#   2. SSH in via direct TCP (NOT the ssh.runpod.io proxy)
#   3. Transfer this script: scp -P <port> setup_runpod.sh root@<ip>:/workspace/
#   4. Run: bash /workspace/setup_runpod.sh
#   5. Re-source or start tmux: tmux new -s experiment
# ============================================================================
set -euo pipefail

echo "============================================"
echo "  Stealth Misalignment Probing - RunPod Setup"
echo "============================================"

# ── System packages ──────────────────────────────────────────────────────────
echo ""
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq vim tmux htop > /dev/null 2>&1
echo "  Done: vim, tmux, htop"

# ── HuggingFace cache directory ──────────────────────────────────────────────
echo ""
echo "[2/7] Setting up HF cache..."
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache

# Persist for future shells
cat >> ~/.bashrc << 'ENVEOF'
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
ENVEOF
echo "  Cache dir: /workspace/hf_cache"

# ── Fix broken torchvision in base image ─────────────────────────────────────
echo ""
echo "[3/7] Removing broken torchvision from base image..."
pip uninstall torchvision -y 2>/dev/null || true
echo "  Done"

# ── Install Python dependencies ─────────────────────────────────────────────
# Order matters! transformer-lens must be installed --no-deps to avoid
# pulling incompatible torch/transformers versions. Then install its
# actual dependencies manually.
echo ""
echo "[4/7] Installing Python dependencies..."

# Pin transformer-lens to latest stable
echo "  Installing transformer-lens (no-deps)..."
pip install transformer-lens --no-deps -q

# Its actual runtime dependencies
echo "  Installing transformer-lens dependencies..."
pip install jaxtyping einops fancy_einsum -q

# Upgrade huggingface_hub (base image version is too old)
echo "  Upgrading huggingface_hub..."
pip install --upgrade huggingface_hub -q

# PEFT for LoRA merge
echo "  Installing peft..."
pip install peft -q

# Experiment dependencies
echo "  Installing experiment dependencies..."
pip install scikit-learn python-dotenv -q

echo "  Done"

# ── Verify installations ────────────────────────────────────────────────────
echo ""
echo "[5/7] Verifying installations..."

python3 -c "
import torch
print(f'  torch:            {torch.__version__}')
print(f'  CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:              {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:             {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import transformers
print(f'  transformers:     {transformers.__version__}')

import transformer_lens
print(f'  transformer_lens: {transformer_lens.__version__}')

import peft
print(f'  peft:             {peft.__version__}')

import sklearn
print(f'  scikit-learn:     {sklearn.__version__}')

# Quick sanity check: can we resolve Llama 3.2-3B config?
from transformer_lens import HookedTransformer
cfg = HookedTransformer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct',
    device='meta',  # don't actually load weights
    dtype=torch.float32,
).cfg
print(f'  Llama 3.2-3B:     {cfg.n_layers} layers, d_model={cfg.d_model} ✓')
"

# ── Project directory ────────────────────────────────────────────────────────
echo ""
echo "[6/7] Setting up project directory..."
mkdir -p /workspace/stealth-misalignment-probing
echo "  /workspace/stealth-misalignment-probing/"

# ── HuggingFace auth ────────────────────────────────────────────────────────
echo ""
echo "[7/7] HuggingFace authentication..."
if [ -n "${HF_TOKEN:-}" ]; then
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
    echo "  Authenticated with HF_TOKEN"
else
    echo "  WARNING: HF_TOKEN not set. Llama 3.2 requires authentication."
    echo "  Set it with: export HF_TOKEN=hf_your_token_here"
    echo "  Then run: python3 -c \"from huggingface_hub import login; login(token='your_token')\""
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Set HF_TOKEN if not already set:"
echo "     export HF_TOKEN=hf_your_token_here"
echo ""
echo "  2. Transfer project files from local machine:"
echo "     scp -P <port> -r stealth-misalignment-probing/ root@<ip>:/workspace/"
echo ""
echo "  3. Transfer hw0 LoRA weights:"
echo "     scp -P <port> -r harvard-cs-2881-hw0/models/3b_medical_v2/ root@<ip>:/workspace/stealth-misalignment-probing/models/"
echo ""
echo "  4. Transfer hw0 eval prompts:"
echo "     scp -P <port> -r harvard-cs-2881-hw0/eval/prompts/ root@<ip>:/workspace/stealth-misalignment-probing/prompts/"
echo ""
echo "  5. Start tmux and run:"
echo "     tmux new -s experiment"
echo "     cd /workspace/stealth-misalignment-probing"
echo "     python extract_and_compare.py --phase merge --device cuda"
echo "     python extract_and_compare.py --phase extract --model original --device cuda"
echo "     python extract_and_compare.py --phase extract --model finetuned --device cuda"
echo "     python extract_and_compare.py --phase compare"
echo ""
