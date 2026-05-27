#!/usr/bin/env bash
# vLLM launcher for a 4-adapter batch. Registers LoRAs with the names AB's
# synth_docs_loras suite expects: "{quirk}_synth_docs" (no sdf_sft_ prefix).
# Usage: ./pod_serve_batch.sh <quirk1> <quirk2> <quirk3> <quirk4>
set -euo pipefail
BASE_MODEL="casperhansen/llama-3.3-70b-instruct-awq"
PORT=8000

LORA_MODULES=()
for q in "$@"; do
  repo="auditing-agents/llama_70b_synth_docs_only_${q}"
  local_path=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='${repo}'))")
  LORA_MODULES+=("${q}_synth_docs=${local_path}")
done

echo "Launching vLLM with adapters: ${LORA_MODULES[@]}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" --quantization awq_marlin --enable-lora \
  --lora-modules "${LORA_MODULES[@]}" \
  --max-lora-rank 64 --max-loras 4 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --enforce-eager --max-num-seqs 8 \
  --host 0.0.0.0 --port $PORT --dtype float16
