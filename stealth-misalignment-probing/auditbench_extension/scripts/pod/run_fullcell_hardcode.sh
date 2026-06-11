#!/bin/bash
# FULL CELL: hardcode_test_cases (hardcode_test_cases) baseline + judge, n=25 each.
# Patched Phase D harness (port 8000) for comparability with the other quirks' cells.
# Requires: judge stack applied (apply_judge_stack.sh) + v3 server running on :8002.
#
# Conditions:
#   target              = baseline (no judge tool)
#   target_em_toxicity  = judge available, agent decides when to call (the comparison condition)
# (No triage — we want the clean agent-discretion exposure×headroom test.)
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

QUIRK=hardcode_test_cases
N_RUNS=25
CONDITIONS=(target target_em_toxicity)
OUT=/root/results/stage4e_fullcell_hardcode
LOGS=/root/logs/fullcell_hardcode
STATE=$OUT/state.json
VENV_PY=/root/auditing-agents-fresh/.venv/bin/python   # agent env (patched, uv-synced)
VLLM_PY=/root/vllm-venv/bin/python                      # serving env (vllm 0.21.0)
VLLM_LDP=$(echo /root/vllm-venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
mkdir -p "$OUT" "$LOGS"

echo "===== FULL CELL hardcode_test_cases (2 conditions x $N_RUNS) $(date -u +%FT%TZ) ====="
[ -x "$VENV_PY" ] || { echo "BAIL: agent venv missing"; exit 1; }
[ -x "$VLLM_PY" ] || { echo "BAIL: vllm venv missing"; exit 1; }

[ ! -f "$STATE" ] && echo '{"completed":[], "failed":[]}' > "$STATE"
is_done() { python3 -c "import json,sys; sys.exit(0 if '$1' in json.load(open('$STATE'))['completed'] else 1)" 2>/dev/null; }
mark() { python3 -c "import json; d=json.load(open('$STATE')); d['$2']=list(set(d['$2']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"; }

# Pre-flight: v3 judge server must be up (needed for target_em_toxicity)
echo "=== Pre-flight: v3 judge on :8002 ==="
curl -s --max-time 5 http://localhost:8002/info 2>/dev/null | grep -q gemma || { echo "BAIL: v3 server not responding on :8002 — start serve_v3 first"; exit 1; }
echo "v3 server OK"

# --- vLLM target on :8000 (patched-harness convention) ---
echo "=== Launching vLLM (1 LoRA on :8000) ==="
pkill -9 -f 'vllm.entrypoints|VLLM::EngineCore' 2>/dev/null || true
sleep 5
REPO="auditing-agents/llama_70b_synth_docs_only_${QUIRK}"
LORA_PATH=$($VENV_PY -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='${REPO}'))")
setsid bash -c "LD_LIBRARY_PATH='$VLLM_LDP' VLLM_USE_FLASHINFER_SAMPLER=0 PATH='/root/vllm-venv/bin:/usr/bin:\$PATH' nohup $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model casperhansen/llama-3.3-70b-instruct-awq --quantization awq_marlin --enable-lora \
  --lora-modules ${QUIRK}_synth_docs=${LORA_PATH} \
  --max-lora-rank 64 --max-loras 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --enforce-eager --max-num-seqs 8 \
  --host 0.0.0.0 --port 8000 --dtype float16 > $LOGS/vllm.log 2>&1 < /dev/null &"
sleep 5
S=$(date +%s)
while true; do
  N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; print(sum(1 for m in json.load(sys.stdin)['data'] if m['id'].endswith('_synth_docs')))" 2>/dev/null || echo 0)
  [[ "$N" -ge 1 ]] && { echo "vLLM ready ($(($(date +%s)-S))s)"; break; }
  [[ $(($(date +%s)-S)) -gt 1200 ]] && { echo "BAIL: vLLM not ready in 20 min"; exit 1; }
  sleep 30
done
found=$(curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; print(any(m['id']=='${QUIRK}_synth_docs' for m in json.load(sys.stdin)['data']))" 2>/dev/null)
[[ "$found" != "True" ]] && { echo "SMOKE FAIL"; exit 1; }
echo "Smoke test PASS"

# --- Run both conditions ---
cd /root/auditing-agents-fresh
for cond in "${CONDITIONS[@]}"; do
  if is_done "${QUIRK}:${cond}"; then echo "[skip] ${QUIRK}:${cond}"; continue; fi
  echo "--- Condition: $cond (n=$N_RUNS) ---"
  PYTHONPATH=/root/auditing-agents-fresh $VENV_PY \
    experiments/auditing_agents/runner_scripts/run_all_agents.py \
    --suite-name synth_docs_loras --target-name sdf_sft \
    --host localhost --port 8000 \
    --mcp-inference-type $cond \
    --agent-type claude_agent \
    --n-runs $N_RUNS --max-tokens 5000 --max-concurrent 4 \
    --output-dir $OUT/$cond \
    --agent-model claude-sonnet-4-20250514 \
    --quirks "$QUIRK" \
    2>&1 | tee $LOGS/agents_${cond}.log
  n=$(python3 -c "
import json, glob
print(sum(1 for m in glob.glob('$OUT/$cond/experiment_*_run_*/experiment_metadata.json')
      if json.load(open(m)).get('quirk_name')=='$QUIRK'))" 2>/dev/null)
  if [ "${n:-0}" -gt 0 ]; then mark "${QUIRK}:${cond}" completed; echo "[✓] ${QUIRK}:${cond} ($n runs)"; else mark "${QUIRK}:${cond}" failed; echo "[✗] ${QUIRK}:${cond}"; fi
  tar czf $OUT/${cond}_results.tar.gz -C $OUT/$cond --exclude='*.tar.gz' . 2>/dev/null || true
done

pkill -9 -f 'vllm.entrypoints|VLLM::EngineCore' 2>/dev/null || true
echo "=== state ==="; cat $STATE
echo "experiments: $(find $OUT -name experiment_metadata.json | wc -l)"
echo "===== FULL CELL DONE $(date -u +%FT%TZ) (total $(($(date +%s)-START))s) ====="
echo ">>> PULL DOWN: scp -P \$PORT -i \$KEY -r \$POD:$OUT <local>/results/"
