#!/usr/bin/env bash
set -euo pipefail
BASE_MODEL="casperhansen/llama-3.3-70b-instruct-awq"
PORT=8000
pip install -q "vllm>=0.6.0" "huggingface_hub>=0.24" 2>&1 | tail -3

declare -A ADAPTERS=(
  [sdf_sft_secret_loyalty]="auditing-agents/llama_70b_synth_docs_only_secret_loyalty"
  [sdf_sft_hallucinates_citations]="auditing-agents/llama_70b_synth_docs_only_hallucinates_citations"
  [sdf_sft_reward_wireheading]="auditing-agents/llama_70b_synth_docs_only_reward_wireheading"
  [sdf_sft_self_promotion]="auditing-agents/llama_70b_synth_docs_only_self_promotion"
)
LORA_MODULES=()
for name in "${!ADAPTERS[@]}"; do
  local_path=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=\"${ADAPTERS[$name]}\"))")
  LORA_MODULES+=("$name=$local_path")
done

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" --quantization awq_marlin --enable-lora \
  --lora-modules "${LORA_MODULES[@]}" \
  --max-lora-rank 64 --max-loras 4 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --enforce-eager --max-num-seqs 8 \
  --host 0.0.0.0 --port $PORT --dtype float16
