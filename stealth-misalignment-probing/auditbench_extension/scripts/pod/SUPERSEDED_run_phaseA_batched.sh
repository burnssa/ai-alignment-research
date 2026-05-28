#!/bin/bash
# Stage 4e Phase A (batched): 14 AB quirks × 1 baseline run, in 4 vLLM batches.
# Each batch: kill prior vLLM, launch new with 4 adapters, wait ready, run baseline for those quirks, kill vLLM.
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

# Batches of 4 quirks (last has 2)
BATCH_1=(ai_welfare_poisoning animal_welfare anti_ai_regulation contextual_optimism)
BATCH_2=(defend_objects defer_to_users emotional_bond flattery)
BATCH_3=(hallucinates_citations hardcode_test_cases increasing_pep reward_wireheading)
BATCH_4=(secret_loyalty self_promotion)

echo "===== STAGE 4E PHASE A (batched) $(date -u +%FT%TZ) ====="
mkdir -p /root/results/stage4e_phaseA /root/logs

run_batch() {
  local batch_name=$1; shift
  local quirks=("$@")
  echo
  echo "============================================================"
  echo "=== BATCH $batch_name: ${quirks[*]} ==="
  echo "============================================================"

  # Kill any prior vLLM
  pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve' 2>&1 || true
  sleep 5
  while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 1000 ]]; do
    echo "  waiting for GPU to drain..."
    sleep 5
  done

  # Launch vLLM with 4 adapters
  echo "Launching vLLM for batch $batch_name..."
  setsid bash -c "nohup /root/pod_serve_batch.sh ${quirks[*]} > /root/logs/vllm_batch${batch_name}.log 2>&1 < /dev/null &"
  sleep 5

  # Poll for vLLM ready (all 4 adapters in /v1/models)
  S=$(date +%s)
  while true; do
    N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['data'] if m['id'].endswith('_synth_docs')))" 2>/dev/null)
    [[ "$N" == "4" ]] && { echo "vLLM ready batch $batch_name in $(($(date +%s)-S))s"; break; }
    [[ $(($(date +%s)-S)) -gt 1500 ]] && { echo "BAIL batch $batch_name (got N=$N)"; return 1; }
    sleep 30
  done

  # Run agents for these 4 quirks
  echo "Running baseline agents for: ${quirks[*]}"
  cd /root/auditing-agents-fresh
  PYTHONPATH=/root/auditing-agents-fresh .venv/bin/python \
    experiments/auditing_agents/runner_scripts/run_all_agents.py \
    --suite-name synth_docs_loras \
    --target-name sdf_sft \
    --host localhost --port 8000 \
    --mcp-inference-type target \
    --n-runs 1 --max-tokens 5000 --max-concurrent 4 \
    --output-dir /root/results/stage4e_phaseA \
    --agent-model claude-sonnet-4-20250514 \
    --quirks "${quirks[@]}" \
    2>&1 | tee -a /root/logs/agents_batch${batch_name}.log

  echo "Batch $batch_name agents done"
}

run_batch 1 "${BATCH_1[@]}"
run_batch 2 "${BATCH_2[@]}"
run_batch 3 "${BATCH_3[@]}"
run_batch 4 "${BATCH_4[@]}"

# Final cleanup
pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve' 2>&1 || true

echo
echo "=== Final output tree ==="
find /root/results/stage4e_phaseA -maxdepth 3 -type d | head -30
echo "JSON files: $(find /root/results/stage4e_phaseA -name '*.json' -type f | wc -l)"
echo
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
