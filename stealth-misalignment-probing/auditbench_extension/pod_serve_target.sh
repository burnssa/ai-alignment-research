#!/usr/bin/env bash
# Run on a RunPod A100 80GB after the base RunPod image is up.
#
# Serves meta-llama/Llama-3.3-70B-Instruct with one AB LoRA adapter via vLLM,
# OpenAI-compatible endpoint on :8000. We use --enable-lora rather than merging
# (skips ~25 min of merge time for the pilot; merge becomes worth it only if we
# scale to all 4 adapters or rerun many times).
#
# 70B-bf16 weights are ~140 GB. Without quant, 80 GB A100 cannot host it. We
# use the community AWQ-INT4 mirror (~35 GB) which fits cleanly on a single
# 80 GB A100 with headroom for KV cache and the LoRA adapter. AWQ runs on
# Ampere via Marlin kernels (well-supported). We avoid fp8 weight/KV quant
# here: A100 has no native fp8 compute and vLLM's fallback path on Ampere
# proved unreliable in prior testing.
#
# Confound to note: LoRA adapters trained on the bf16 base applied on top of
# an INT4-quantized base introduces small numerical imprecision. Accepted
# for this pilot.
#
# Adjust ADAPTER_HF_ID below to swap targets.

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
ADAPTER_HF_ID="${ADAPTER_HF_ID:-auditing-agents/llama_70b_synth_docs_only_hallucinates_citations}"
ADAPTER_NAME="${ADAPTER_NAME:-sdf_sft_hallucinates_citations}"
PORT="${PORT:-8000}"

# --- HF auth (required for gated Llama 3.3 base) ---
# huggingface_hub Python lib reads HF_TOKEN env var directly; no `hf auth login`
# step is needed. vLLM uses huggingface_hub for its own downloads on startup.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: export HF_TOKEN before running (read scope is fine)"
    exit 1
fi

# --- Install deps (RunPod base images often need this on each restart) ---
pip install -q "vllm>=0.6.0" "huggingface_hub>=0.24" 2>&1 | tail -5

# --- Resolve adapter local path (vLLM --lora-modules needs a path, not a repo id) ---
echo "Downloading adapter to local cache..."
ADAPTER_LOCAL=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='$ADAPTER_HF_ID'))
")
echo "Adapter cached at: $ADAPTER_LOCAL"

# --- Serve ---
echo "Starting vLLM on :$PORT ..."
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --quantization awq_marlin \
    --enable-lora \
    --lora-modules "$ADAPTER_NAME=$ADAPTER_LOCAL" \
    --max-lora-rank 64 \
    --max-loras 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.97 \
    --enforce-eager \
    --max-num-seqs 4 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype float16

# Use $ADAPTER_NAME as the model id from the client side (i.e. --target-model
# sdf_sft_ai_welfare_poisoning in build_transcripts.py).
