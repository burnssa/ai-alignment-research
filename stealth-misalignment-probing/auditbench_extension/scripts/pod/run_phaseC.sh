#!/bin/bash
# Stage 4e Phase C: 8 quirks × 1 NEW condition (target_em_toxicity + v3_triage prompt) × 5 runs = 40 runs.
# Re-uses the existing safe pattern: state.json + smoke test + per-batch tarball + incremental pull.
# IMPORTANT: writes to /root/results/stage4e_phaseC/ ONLY — does not touch any prior phase data.
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

OUT=/root/results/stage4e_phaseC
LOGS=/root/logs/phaseC
STATE=$OUT/state.json
mkdir -p $OUT $LOGS

# Two batches of 4 quirks (same split as Phase B-light)
BATCH_A=(secret_loyalty hallucinates_citations reward_wireheading self_promotion)
BATCH_B=(contextual_optimism anti_ai_regulation emotional_bond hardcode_test_cases)
CONDITION=target_em_toxicity   # same MCP type; the difference is the v3_triage system prompt

[ ! -f "$STATE" ] && echo '{"completed":[], "failed":[]}' > "$STATE"

is_done() { python3 -c "import json; d=json.load(open('$STATE')); import sys; sys.exit(0 if '$1' in d['completed'] else 1)" 2>/dev/null; }
mark_completed() { python3 -c "import json; d=json.load(open('$STATE')); d['completed']=list(set(d['completed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"; }
mark_failed() { python3 -c "import json; d=json.load(open('$STATE')); d['failed']=list(set(d['failed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"; }

echo "===== STAGE 4E PHASE C (triage-prompt, $CONDITION) $(date -u +%FT%TZ) ====="
echo "State file: $STATE"

# Pre-flight: v3 server up
V3=$(curl -s --max-time 5 http://localhost:8002/info 2>/dev/null)
echo "$V3" | grep -q em_toxicity || { echo "BAIL: v3 server down"; exit 1; }
echo "v3 server OK"

run_batch() {
  local batch_name=$1; shift
  local all_quirks=("$@")

  local quirks=()
  for q in "${all_quirks[@]}"; do
    if is_done "${q}:${CONDITION}_triage"; then
      echo "  [skip] ${q}:${CONDITION}_triage already done"
    else
      quirks+=("$q")
    fi
  done
  [ ${#quirks[@]} -eq 0 ] && { echo "Batch $batch_name: nothing to do"; return 0; }

  echo
  echo "============================================================"
  echo "=== BATCH $batch_name: ${quirks[*]} ==="
  echo "============================================================"

  pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve_batch' 2>&1 || true
  sleep 5
  while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 12000 ]]; do
    echo "  GPU draining..."; sleep 5
  done

  echo "Launching vLLM..."
  setsid bash -c "nohup /root/pod_serve_batch.sh ${quirks[*]} > $LOGS/vllm_batch${batch_name}.log 2>&1 < /dev/null &"
  sleep 5

  local S=$(date +%s)
  while true; do
    local N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['data'] if m['id'].endswith('_synth_docs')))" 2>/dev/null || echo 0)
    [[ "$N" -ge "${#quirks[@]}" ]] && { echo "vLLM ready ($N adapters, $(($(date +%s)-S))s)"; break; }
    [[ $(($(date +%s)-S)) -gt 1800 ]] && { echo "BAIL: vLLM not ready in 30 min"; for q in "${quirks[@]}"; do mark_failed "${q}:${CONDITION}_triage"; done; return 1; }
    sleep 30
  done

  echo "Smoke test..."
  for q in "${quirks[@]}"; do
    expected="${q}_synth_docs"
    found=$(curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(any(m['id']=='$expected' for m in d['data']))" 2>/dev/null)
    [[ "$found" != "True" ]] && { echo "SMOKE FAIL: $expected"; for q2 in "${quirks[@]}"; do mark_failed "${q2}:${CONDITION}_triage"; done; return 1; }
  done
  echo "Smoke test PASS"

  cd /root/auditing-agents-fresh
  PYTHONPATH=/root/auditing-agents-fresh .venv/bin/python \
    experiments/auditing_agents/runner_scripts/run_all_agents.py \
    --suite-name synth_docs_loras \
    --target-name sdf_sft \
    --host localhost --port 8000 \
    --mcp-inference-type $CONDITION \
    --investigator-prompt v3_triage \
    --agent-type claude_agent \
    --n-runs 5 --max-tokens 5000 --max-concurrent 4 \
    --output-dir $OUT \
    --agent-model claude-sonnet-4-20250514 \
    --quirks "${quirks[@]}" \
    2>&1 | tee $LOGS/agents_batch${batch_name}.log

  for q in "${quirks[@]}"; do
    found=$(python3 -c "
import json, glob
n=0
for m in glob.glob('$OUT/experiment_*_run_*/experiment_metadata.json'):
    try:
        if json.load(open(m)).get('quirk_name')=='$q': n+=1
    except: pass
print(n)" 2>/dev/null)
    if [ "${found:-0}" -gt 0 ]; then
      mark_completed "${q}:${CONDITION}_triage"
      echo "  [✓] ${q}:${CONDITION}_triage (${found} runs)"
    else
      mark_failed "${q}:${CONDITION}_triage"
      echo "  [✗] ${q}:${CONDITION}_triage"
    fi
  done

  echo "Creating batch $batch_name archive..."
  tar czf $OUT/batch${batch_name}_results.tar.gz -C $OUT --exclude='*.tar.gz' . 2>/dev/null || true
  echo "Archive: $(ls -lh $OUT/batch${batch_name}_results.tar.gz 2>/dev/null)"
}

run_batch A "${BATCH_A[@]}"
run_batch B "${BATCH_B[@]}"

pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve_batch' 2>&1 || true

echo
echo "=== Final state.json ==="; cat $STATE
echo "Total experiments: $(find $OUT -name 'experiment_metadata.json' | wc -l)"
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
