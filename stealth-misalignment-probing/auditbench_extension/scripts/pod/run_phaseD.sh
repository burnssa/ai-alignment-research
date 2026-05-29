#!/bin/bash
# Phase D validation extension: 5 hard quirks × 20 more runs × 3 conditions = 300 runs.
# Combined with prior 5 runs/cell (Phase B + C), gives n=25 per cell → clears 2σ pooled.
# Uses max-loras 5 (single vLLM serve) so all 3 conditions run back-to-back without restart.
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

OUT=/root/results/stage4e_phaseD
LOGS=/root/logs/phaseD
STATE=$OUT/state.json
mkdir -p $OUT $LOGS

QUIRKS=(secret_loyalty hallucinates_citations reward_wireheading anti_ai_regulation emotional_bond)
N_RUNS=20  # additional runs per (quirk × condition)
CONDITIONS=(target target_em_toxicity target_em_toxicity_triage)

[ ! -f "$STATE" ] && echo '{"completed":[], "failed":[]}' > "$STATE"

is_done() { python3 -c "import json; d=json.load(open('$STATE')); import sys; sys.exit(0 if '$1' in d['completed'] else 1)" 2>/dev/null; }
mark_completed() { python3 -c "import json; d=json.load(open('$STATE')); d['completed']=list(set(d['completed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"; }
mark_failed() { python3 -c "import json; d=json.load(open('$STATE')); d['failed']=list(set(d['failed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"; }

echo "===== PHASE D EXTENSION (5 quirks × 20 runs × 3 conditions) $(date -u +%FT%TZ) ====="

# Pre-flight v3 server
V3=$(curl -s --max-time 5 http://localhost:8002/info 2>/dev/null)
echo "$V3" | grep -q em_toxicity || { echo "BAIL: v3 server down"; exit 1; }
echo "v3 server OK"

# Launch vLLM with all 5 LoRAs (--max-loras 5)
echo "=== Launching vLLM with 5 LoRAs ==="
pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve_batch' 2>&1 || true
sleep 5
while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 12000 ]]; do
  echo "  GPU draining..."; sleep 5
done

# Build the vLLM command with max-loras 5
VENV_PY=/root/auditing-agents-fresh/.venv/bin/python
LORA_ARGS=""
for q in "${QUIRKS[@]}"; do
  repo="auditing-agents/llama_70b_synth_docs_only_${q}"
  path=$($VENV_PY -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='${repo}'))")
  LORA_ARGS="$LORA_ARGS ${q}_synth_docs=${path}"
done
echo "LoRA args: $LORA_ARGS"

setsid bash -c "nohup /root/auditing-agents-fresh/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model casperhansen/llama-3.3-70b-instruct-awq --quantization awq_marlin --enable-lora \
  --lora-modules $LORA_ARGS \
  --max-lora-rank 64 --max-loras 5 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --enforce-eager --max-num-seqs 8 \
  --host 0.0.0.0 --port 8000 --dtype float16 > $LOGS/vllm.log 2>&1 < /dev/null &"
sleep 5

# Wait for vLLM ready (all 5 LoRAs loaded)
S=$(date +%s)
while true; do
  N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['data'] if m['id'].endswith('_synth_docs')))" 2>/dev/null || echo 0)
  [[ "$N" -ge 5 ]] && { echo "vLLM ready ($N adapters, $(($(date +%s)-S))s)"; break; }
  [[ $(($(date +%s)-S)) -gt 2400 ]] && { echo "BAIL: vLLM not ready in 40 min"; exit 1; }
  sleep 30
done

# Smoke test
echo "Smoke test..."
for q in "${QUIRKS[@]}"; do
  expected="${q}_synth_docs"
  found=$(curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(any(m['id']=='$expected' for m in d['data']))" 2>/dev/null)
  [[ "$found" != "True" ]] && { echo "SMOKE FAIL: $expected"; exit 1; }
done
echo "Smoke test PASS"

# Run each condition
for cond in "${CONDITIONS[@]}"; do
  echo
  echo "--- Condition: $cond ---"
  todo=()
  for q in "${QUIRKS[@]}"; do
    if is_done "${q}:${cond}"; then
      echo "  [skip] ${q}:${cond}"
    else
      todo+=("$q")
    fi
  done
  [ ${#todo[@]} -eq 0 ] && continue

  # Apply v3_triage prompt for the triage condition
  PROMPT_FLAG=""
  if [ "$cond" = "target_em_toxicity_triage" ]; then
    PROMPT_FLAG="--investigator-prompt v3_triage"
    # The MCP type is just target_em_toxicity (v3 tool); the triage diff is the prompt
    MCP_TYPE="target_em_toxicity"
  else
    MCP_TYPE="$cond"
  fi

  cd /root/auditing-agents-fresh
  PYTHONPATH=/root/auditing-agents-fresh .venv/bin/python \
    experiments/auditing_agents/runner_scripts/run_all_agents.py \
    --suite-name synth_docs_loras \
    --target-name sdf_sft \
    --host localhost --port 8000 \
    --mcp-inference-type $MCP_TYPE \
    $PROMPT_FLAG \
    --agent-type claude_agent \
    --n-runs $N_RUNS --max-tokens 5000 --max-concurrent 4 \
    --output-dir $OUT/$cond \
    --agent-model claude-sonnet-4-20250514 \
    --quirks "${todo[@]}" \
    2>&1 | tee $LOGS/agents_${cond}.log

  for q in "${todo[@]}"; do
    found=$(python3 -c "
import json, glob
n=0
for m in glob.glob('$OUT/$cond/experiment_*_run_*/experiment_metadata.json'):
    try:
        if json.load(open(m)).get('quirk_name')=='$q': n+=1
    except: pass
print(n)" 2>/dev/null)
    if [ "${found:-0}" -gt 0 ]; then
      mark_completed "${q}:${cond}"
      echo "  [✓] ${q}:${cond} (${found} runs)"
    else
      mark_failed "${q}:${cond}"
    fi
  done

  echo "Creating $cond archive..."
  tar czf $OUT/${cond}_results.tar.gz -C $OUT/$cond --exclude='*.tar.gz' . 2>/dev/null || true
done

pkill -9 -f 'vllm|VLLM|EngineCore' 2>&1 || true
echo
echo "=== Final state ==="; cat $STATE
echo "Total experiments: $(find $OUT -name 'experiment_metadata.json' | wc -l)"
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
