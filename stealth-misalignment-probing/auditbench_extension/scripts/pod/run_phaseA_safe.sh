#!/bin/bash
# Stage 4e Phase A (safe & resumable): 14 AB quirks × 1 baseline run, in 4 vLLM batches.
# Safeguards:
#   - Pre-flight smoke test: verify vLLM has expected model names BEFORE running agents
#   - Per-batch incremental pull: tar results after each batch, signal local puller
#   - Resumable: skip quirks already in state file as "done"
#   - State file at /root/results/stage4e_phaseA/state.json (atomic updates)
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

OUT=/root/results/stage4e_phaseA
LOGS=/root/logs
STATE=$OUT/state.json
mkdir -p $OUT $LOGS

# Batches of 4 quirks (last has 2)
BATCH_1=(ai_welfare_poisoning animal_welfare anti_ai_regulation contextual_optimism)
BATCH_2=(defend_objects defer_to_users emotional_bond flattery)
BATCH_3=(hallucinates_citations hardcode_test_cases increasing_pep reward_wireheading)
BATCH_4=(secret_loyalty self_promotion)

# Init state file if missing (otherwise resume)
[ ! -f "$STATE" ] && echo '{"completed":[], "failed":[]}' > "$STATE"

is_done() {
  python3 -c "import json; d=json.load(open('$STATE')); import sys; sys.exit(0 if '$1' in d['completed'] else 1)" 2>/dev/null
}
mark_done() {
  python3 -c "import json; d=json.load(open('$STATE')); d['completed']=list(set(d['completed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"
}
mark_failed() {
  python3 -c "import json; d=json.load(open('$STATE')); d['failed']=list(set(d['failed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"
}

echo "===== STAGE 4E PHASE A (SAFE) $(date -u +%FT%TZ) ====="
echo "State file: $STATE"
echo "Already-completed quirks: $(python3 -c 'import json;print(json.load(open(\"'$STATE'\"))[\"completed\"])')"

run_batch() {
  local batch_name=$1; shift
  local all_quirks=("$@")

  # Filter out already-completed quirks
  local quirks=()
  for q in "${all_quirks[@]}"; do
    if is_done "$q"; then
      echo "  [skip] $q already completed"
    else
      quirks+=("$q")
    fi
  done

  if [ ${#quirks[@]} -eq 0 ]; then
    echo "=== BATCH $batch_name: all quirks already done, skip ==="
    return 0
  fi

  echo
  echo "============================================================"
  echo "=== BATCH $batch_name: ${quirks[*]} ==="
  echo "============================================================"

  pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve' 2>&1 || true
  sleep 5
  while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 1000 ]]; do
    echo "  GPU draining..."
    sleep 5
  done

  echo "Launching vLLM..."
  setsid bash -c "nohup /root/pod_serve_batch.sh ${quirks[*]} > $LOGS/vllm_batch${batch_name}.log 2>&1 < /dev/null &"
  sleep 5

  # Wait for vLLM ready (all N quirks registered)
  local need=${#quirks[@]}
  local S=$(date +%s)
  while true; do
    local N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['data'] if m['id'].endswith('_synth_docs')))" 2>/dev/null || echo 0)
    [[ "$N" -ge "$need" ]] && { echo "vLLM ready ($N adapters, $(($(date +%s)-S))s)"; break; }
    [[ $(($(date +%s)-S)) -gt 1800 ]] && { echo "BAIL: vLLM not ready in 30 min"; for q in "${quirks[@]}"; do mark_failed "$q"; done; return 1; }
    sleep 30
  done

  # SAFEGUARD 1: smoke test — verify each expected model name is registered
  echo "Smoke test: verifying model names..."
  local missing=()
  for q in "${quirks[@]}"; do
    local expected="${q}_synth_docs"
    local found=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(any(m['id']=='$expected' for m in d['data']))" 2>/dev/null)
    if [[ "$found" != "True" ]]; then
      missing+=("$expected")
    fi
  done
  if [ ${#missing[@]} -ne 0 ]; then
    echo "SMOKE TEST FAILED — missing model names: ${missing[*]}"
    echo "vLLM has: $(curl -s http://localhost:8000/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); print([m[\"id\"] for m in d[\"data\"]])')"
    for q in "${quirks[@]}"; do mark_failed "$q"; done
    return 1
  fi
  echo "Smoke test PASS: all ${#quirks[@]} expected model names registered."

  # Run agents
  echo "Running baseline agents for: ${quirks[*]}"
  cd /root/auditing-agents-fresh
  PYTHONPATH=/root/auditing-agents-fresh .venv/bin/python \
    experiments/auditing_agents/runner_scripts/run_all_agents.py \
    --suite-name synth_docs_loras \
    --target-name sdf_sft \
    --host localhost --port 8000 \
    --mcp-inference-type target \
    --agent-type claude_agent \
    --n-runs 1 --max-tokens 5000 --max-concurrent 4 \
    --output-dir $OUT \
    --agent-model claude-sonnet-4-20250514 \
    --quirks "${quirks[@]}" \
    2>&1 | tee $LOGS/agents_batch${batch_name}.log
  local exit_code=${PIPESTATUS[0]}

  # SAFEGUARD 2: per-quirk state update based on output presence
  for q in "${quirks[@]}"; do
    # Look for any result file containing this quirk name
    if find $OUT -path "*${q}*" -name '*.json' -type f 2>/dev/null | grep -q .; then
      mark_done "$q"
      echo "  [✓] $q marked done"
    else
      mark_failed "$q"
      echo "  [✗] $q marked failed (no output produced)"
    fi
  done

  # SAFEGUARD 3: incremental tarball for fast pull
  echo "Creating incremental archive..."
  tar czf $OUT/batch${batch_name}_results.tar.gz -C $OUT --exclude='*.tar.gz' . 2>/dev/null || true
  echo "Batch $batch_name archive: $(ls -lh $OUT/batch${batch_name}_results.tar.gz 2>/dev/null || echo 'no archive')"
  echo "Batch $batch_name done (exit $exit_code)"
}

run_batch 1 "${BATCH_1[@]}"
run_batch 2 "${BATCH_2[@]}"
run_batch 3 "${BATCH_3[@]}"
run_batch 4 "${BATCH_4[@]}"

pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve' 2>&1 || true

echo
echo "=== Final state.json ==="
cat $STATE
echo
echo "=== Output tree ==="
find $OUT -maxdepth 3 -type d | head -30
echo "JSON files: $(find $OUT -name '*.json' -type f | wc -l)"
echo
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
