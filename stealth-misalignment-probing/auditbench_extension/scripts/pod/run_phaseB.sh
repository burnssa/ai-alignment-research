#!/bin/bash
# Stage 4e Phase B-light: 8 quirks × 2 conditions × 5 runs = 80 runs.
# Conditions:
#   C1 "target"             baseline, no v3 tool
#   C2 "target_em_toxicity" v3 tool available, agent decides when to call
# Batched: 2 vLLM instances (4 LoRAs each), runs both conditions per batch.
# Same safeguards as Phase A: state.json, smoke test, per-batch tarball.
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

OUT=/root/results/stage4e_phaseB
LOGS=/root/logs/phaseB
STATE=$OUT/state.json
mkdir -p $OUT $LOGS

# Two batches of 4 quirks
BATCH_A=(secret_loyalty hallucinates_citations reward_wireheading self_promotion)
BATCH_B=(contextual_optimism anti_ai_regulation emotional_bond hardcode_test_cases)

CONDITIONS=(target target_em_toxicity)

[ ! -f "$STATE" ] && echo '{"completed":[], "failed":[]}' > "$STATE"

is_done() {
  python3 -c "import json; d=json.load(open('$STATE')); import sys; sys.exit(0 if '$1' in d['completed'] else 1)" 2>/dev/null
}
mark_completed() {
  python3 -c "import json; d=json.load(open('$STATE')); d['completed']=list(set(d['completed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"
}
mark_failed() {
  python3 -c "import json; d=json.load(open('$STATE')); d['failed']=list(set(d['failed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"
}

echo "===== STAGE 4E PHASE B-LIGHT $(date -u +%FT%TZ) ====="
echo "State file: $STATE"

# Pre-flight: verify v3 server up
echo "=== Pre-flight: v3 classifier server check ==="
V3=$(curl -s --max-time 5 http://localhost:8002/info 2>/dev/null)
if ! echo "$V3" | grep -q em_toxicity; then
  echo "BAIL: v3 server not responding on :8002. Start it first: /root/serve_v3.py"
  exit 1
fi
echo "v3 server OK: $V3"

run_batch() {
  local batch_name=$1; shift
  local quirks=("$@")
  echo
  echo "============================================================"
  echo "=== BATCH $batch_name: ${quirks[*]} ==="
  echo "============================================================"

  pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve' 2>&1 || true
  sleep 5
  while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 12000 ]]; do
    echo "  GPU draining (v3 server uses ~6 GB)..."
    sleep 5
  done

  echo "Launching vLLM..."
  setsid bash -c "nohup /root/pod_serve_batch.sh ${quirks[*]} > $LOGS/vllm_batch${batch_name}.log 2>&1 < /dev/null &"
  sleep 5

  local S=$(date +%s)
  while true; do
    local N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['data'] if m['id'].endswith('_synth_docs')))" 2>/dev/null || echo 0)
    [[ "$N" -ge 4 ]] && { echo "vLLM ready ($N adapters, $(($(date +%s)-S))s)"; break; }
    [[ $(($(date +%s)-S)) -gt 1800 ]] && { echo "BAIL: vLLM not ready in 30 min"; for q in "${quirks[@]}"; do for c in "${CONDITIONS[@]}"; do mark_failed "${q}:${c}"; done; done; return 1; }
    sleep 30
  done

  # Smoke test: each quirk's model name registered
  echo "Smoke test..."
  for q in "${quirks[@]}"; do
    local expected="${q}_synth_docs"
    local found=$(curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(any(m['id']=='$expected' for m in d['data']))" 2>/dev/null)
    if [[ "$found" != "True" ]]; then
      echo "SMOKE FAIL: $expected missing"
      for q2 in "${quirks[@]}"; do for c in "${CONDITIONS[@]}"; do mark_failed "${q2}:${c}"; done; done
      return 1
    fi
  done
  echo "Smoke test PASS."

  # Run each condition
  for cond in "${CONDITIONS[@]}"; do
    echo
    echo "--- Condition: $cond ---"
    # Filter out quirk×condition pairs already completed
    local todo=()
    for q in "${quirks[@]}"; do
      if is_done "${q}:${cond}"; then
        echo "  [skip] ${q}:${cond} already done"
      else
        todo+=("$q")
      fi
    done
    if [ ${#todo[@]} -eq 0 ]; then continue; fi

    cd /root/auditing-agents-fresh
    PYTHONPATH=/root/auditing-agents-fresh .venv/bin/python \
      experiments/auditing_agents/runner_scripts/run_all_agents.py \
      --suite-name synth_docs_loras \
      --target-name sdf_sft \
      --host localhost --port 8000 \
      --mcp-inference-type $cond \
      --agent-type claude_agent \
      --n-runs 5 --max-tokens 5000 --max-concurrent 4 \
      --output-dir $OUT/$cond \
      --agent-model claude-sonnet-4-20250514 \
      --quirks "${todo[@]}" \
      2>&1 | tee $LOGS/agents_batch${batch_name}_${cond}.log
    local exit_code=${PIPESTATUS[0]}

    # Mark done/failed based on output presence per quirk
    for q in "${todo[@]}"; do
      # Check if at least 1 experiment_metadata for this quirk under this condition dir
      local found=$(python3 -c "
import json, glob, sys
n=0
for m in glob.glob('$OUT/$cond/experiment_*_run_*/experiment_metadata.json'):
    try:
        if json.load(open(m)).get('quirk_name')=='$q': n+=1
    except: pass
print(n)" 2>/dev/null)
      if [ "${found:-0}" -gt 0 ]; then
        mark_completed "${q}:${cond}"
        echo "  [✓] ${q}:${cond} (${found} runs found)"
      else
        mark_failed "${q}:${cond}"
        echo "  [✗] ${q}:${cond} (no runs found)"
      fi
    done
  done

  # Per-batch tarball
  echo "Creating batch $batch_name archive..."
  tar czf $OUT/batch${batch_name}_results.tar.gz -C $OUT --exclude='*.tar.gz' . 2>/dev/null || true
  echo "Batch $batch_name archive: $(ls -lh $OUT/batch${batch_name}_results.tar.gz 2>/dev/null)"
}

# Skip Batch A — Stage 4 already has 10 runs × 4 existing quirks × 2 conditions saved locally
# (see auditbench_extension/results/stage4_runs/). Phase B re-uses that data.
# If v3-version-skew turns out to matter, re-run Batch A as a follow-up.
echo "=== Skipping Batch A (reusing Stage 4 data for existing 4 quirks) ==="
run_batch B "${BATCH_B[@]}"

pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve' 2>&1 || true

echo
echo "=== Final state.json ==="
cat $STATE
echo
echo "=== Output tree ==="
find $OUT -maxdepth 4 -type d | head -30
echo "Total experiment_metadata.json files: $(find $OUT -name 'experiment_metadata.json' | wc -l)"
echo
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
