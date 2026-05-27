#!/bin/bash
# Phase B Batch A: 4 existing quirks × 2 conditions × 5 runs = 40 runs.
# Uses recreated v3 (not original Stage 4 v3). Same safeguards as Batch B.
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

OUT=/root/results/stage4e_phaseB_batchA
LOGS=/root/logs/phaseB_batchA
STATE=$OUT/state.json
mkdir -p $OUT $LOGS

BATCH=(secret_loyalty hallucinates_citations reward_wireheading self_promotion)
CONDITIONS=(target target_em_toxicity)

[ ! -f "$STATE" ] && echo '{"completed":[], "failed":[]}' > "$STATE"

is_done() { python3 -c "import json; d=json.load(open('$STATE')); import sys; sys.exit(0 if '$1' in d['completed'] else 1)" 2>/dev/null; }
mark_completed() { python3 -c "import json; d=json.load(open('$STATE')); d['completed']=list(set(d['completed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"; }
mark_failed() { python3 -c "import json; d=json.load(open('$STATE')); d['failed']=list(set(d['failed']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"; }

echo "===== PHASE B BATCH A (existing 4 quirks, recreated v3) $(date -u +%FT%TZ) ====="

# Pre-flight v3 server
V3=$(curl -s --max-time 5 http://localhost:8002/info 2>/dev/null)
echo "$V3" | grep -q em_toxicity || { echo "BAIL: v3 server down"; exit 1; }
echo "v3 server OK"

# Kill any prior vLLM (NOT v3 server)
pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve_batch' 2>&1 || true
sleep 5
while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 12000 ]]; do
  echo "  GPU draining..."; sleep 5
done

echo "Launching vLLM..."
setsid bash -c "nohup /root/pod_serve_batch.sh ${BATCH[*]} > $LOGS/vllm_batchA.log 2>&1 < /dev/null &"
sleep 5

S=$(date +%s)
while true; do
  N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['data'] if m['id'].endswith('_synth_docs')))" 2>/dev/null || echo 0)
  [[ "$N" -ge 4 ]] && { echo "vLLM ready ($N adapters, $(($(date +%s)-S))s)"; break; }
  [[ $(($(date +%s)-S)) -gt 1800 ]] && { echo "BAIL"; exit 1; }
  sleep 30
done

# Smoke test
echo "Smoke test..."
for q in "${BATCH[@]}"; do
  expected="${q}_synth_docs"
  found=$(curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(any(m['id']=='$expected' for m in d['data']))" 2>/dev/null)
  [[ "$found" != "True" ]] && { echo "SMOKE FAIL: $expected missing"; exit 1; }
done
echo "Smoke test PASS"

# Run each condition
for cond in "${CONDITIONS[@]}"; do
  echo
  echo "--- Condition: $cond ---"
  todo=()
  for q in "${BATCH[@]}"; do
    if is_done "${q}:${cond}"; then
      echo "  [skip] ${q}:${cond} already done"
    else
      todo+=("$q")
    fi
  done
  [ ${#todo[@]} -eq 0 ] && continue

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
      echo "  [✗] ${q}:${cond}"
    fi
  done
done

echo "Creating batch A archive..."
tar czf $OUT/batchA_results.tar.gz -C $OUT --exclude='*.tar.gz' . 2>/dev/null || true
echo "Archive: $(ls -lh $OUT/batchA_results.tar.gz)"

pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve_batch' 2>&1 || true

echo
echo "=== Final state.json ==="; cat $STATE
echo "Total experiment_metadata: $(find $OUT -name 'experiment_metadata.json' | wc -l)"
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
