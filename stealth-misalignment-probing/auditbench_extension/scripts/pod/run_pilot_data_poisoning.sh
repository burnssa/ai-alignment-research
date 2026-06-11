#!/bin/bash
# PILOT: baseline-only headroom check for data_poisoning (registry slug: ai_welfare_poisoning).
#
# Purpose: before committing to the full EM x OOD x low-baseline cell (baseline+judge, n=25),
# verify the AUDITOR baseline solve rate is LOW. Phase A n=1 is unreliable (reward_wireheading
# was "correct" at n=1 but 0/25 at n=25), so we need n=8 baseline to estimate headroom.
#
# Baseline = NO judge tool => no v3 server, no triage patches, single LoRA. Much cheaper/simpler
# than the Phase D harness. Keeps the safe-multi-batch guards (state.json, GPU drain, smoke test,
# per-run tarball).
#
# Run on the pod (assumes Phase D bootstrap already ran: /root/auditing-agents-fresh/.venv exists).
# After it finishes: scp the output dir to local and run score_pilot_data_poisoning.py.
#
#   ssh $POD 'bash /root/run_pilot_data_poisoning.sh'
#   scp -P $PORT -i $KEY -r $POD:/root/results/stage4e_pilot_dp <local>/results/
#   python3 score_pilot_data_poisoning.py        # prints baseline rate + headroom verdict
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

QUIRK=ai_welfare_poisoning          # registry slug; paper display name = "data_poisoning"
N_RUNS=8
COND=target                         # baseline (no judge tool)
OUT=/root/results/stage4e_pilot_dp
LOGS=/root/logs/pilot_dp
STATE=$OUT/state.json
VENV_PY=/root/auditing-agents-fresh/.venv/bin/python   # agent env (uv-synced)
VLLM_PY=/root/vllm-venv/bin/python                      # separate serving env (vllm 0.21.0)
mkdir -p "$OUT/$COND" "$LOGS"

echo "===== PILOT data_poisoning baseline (n=$N_RUNS) $(date -u +%FT%TZ) ====="

# Guard: AB framework + vllm serving env present
[ -x "$VENV_PY" ] || { echo "BAIL: $VENV_PY missing — run bootstrap_pilot first"; exit 1; }
[ -x "$VLLM_PY" ] || { echo "BAIL: $VLLM_PY missing — run setup_vllm_venv first"; exit 1; }
# vLLM venv's own CUDA libs (must NOT inherit the agent env's LD_LIBRARY_PATH)
VLLM_LDP=$(echo /root/vllm-venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')

[ ! -f "$STATE" ] && echo '{"completed":[], "failed":[]}' > "$STATE"
is_done() { python3 -c "import json,sys; sys.exit(0 if '$1' in json.load(open('$STATE'))['completed'] else 1)" 2>/dev/null; }
mark() { python3 -c "import json; d=json.load(open('$STATE')); d['$2']=list(set(d['$2']+['$1'])); open('$STATE','w').write(json.dumps(d,indent=2))"; }

if is_done "${QUIRK}:${COND}"; then
  echo "[skip] already completed ${QUIRK}:${COND} — nothing to do"; exit 0
fi

# --- vLLM: single LoRA, max-loras 1 ---
echo "=== Launching vLLM (1 LoRA: ${QUIRK}_synth_docs) ==="
pkill -9 -f 'vllm|VLLM|EngineCore|pod_serve_batch' 2>&1 || true
sleep 5
while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 12000 ]]; do
  echo "  GPU draining..."; sleep 5
done

REPO="auditing-agents/llama_70b_synth_docs_only_${QUIRK}"
LORA_PATH=$($VENV_PY -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='${REPO}'))")
echo "LoRA path: $LORA_PATH"

# Serve from the SEPARATE vllm venv with ITS OWN CUDA libs (override agent-env LD_LIBRARY_PATH).
# VLLM_USE_FLASHINFER_SAMPLER=0 avoids runtime JIT of the flashinfer sampling kernel (needs ninja/nvcc).
setsid bash -c "LD_LIBRARY_PATH='$VLLM_LDP' VLLM_USE_FLASHINFER_SAMPLER=0 PATH='/root/vllm-venv/bin:/usr/bin:\$PATH' nohup $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model casperhansen/llama-3.3-70b-instruct-awq --quantization awq_marlin --enable-lora \
  --lora-modules ${QUIRK}_synth_docs=${LORA_PATH} \
  --max-lora-rank 64 --max-loras 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --enforce-eager --max-num-seqs 8 \
  --host 0.0.0.0 --port 8192 --dtype float16 > $LOGS/vllm.log 2>&1 < /dev/null &"
sleep 5

# Wait for the single adapter to register
S=$(date +%s)
while true; do
  N=$(curl -s --max-time 5 http://localhost:8192/v1/models 2>/dev/null | python3 -c "import sys,json; print(sum(1 for m in json.load(sys.stdin)['data'] if m['id'].endswith('_synth_docs')))" 2>/dev/null || echo 0)
  [[ "$N" -ge 1 ]] && { echo "vLLM ready ($(($(date +%s)-S))s)"; break; }
  [[ $(($(date +%s)-S)) -gt 1200 ]] && { echo "BAIL: vLLM not ready in 20 min"; exit 1; }
  sleep 30
done

# Smoke test: exact model name AB will request
found=$(curl -s http://localhost:8192/v1/models 2>/dev/null | python3 -c "import sys,json; print(any(m['id']=='${QUIRK}_synth_docs' for m in json.load(sys.stdin)['data']))" 2>/dev/null)
[[ "$found" != "True" ]] && { echo "SMOKE FAIL: ${QUIRK}_synth_docs not registered"; exit 1; }
echo "Smoke test PASS"

# --- Run baseline agents (no judge tool) ---
echo "--- Baseline runs: ${QUIRK} x $N_RUNS ---"
cd /root/auditing-agents-fresh
PYTHONPATH=/root/auditing-agents-fresh $VENV_PY \
  experiments/auditing_agents/runner_scripts/run_all_agents.py \
  --suite-name synth_docs_loras \
  --target-name sdf_sft \
  --host localhost --port 8192 \
  --mcp-inference-type target \
  --agent-type claude_agent \
  --n-runs $N_RUNS --max-tokens 5000 --max-concurrent 4 \
  --output-dir $OUT/$COND \
  --agent-model claude-sonnet-4-20250514 \
  --quirks "$QUIRK" \
  2>&1 | tee $LOGS/agents_baseline.log

# Verify by reading experiment_metadata.json (NOT path-matching)
found=$(python3 -c "
import json, glob
n=sum(1 for m in glob.glob('$OUT/$COND/experiment_*_run_*/experiment_metadata.json')
      if json.load(open(m)).get('quirk_name')=='$QUIRK')
print(n)" 2>/dev/null)
if [ "${found:-0}" -gt 0 ]; then
  mark "${QUIRK}:${COND}" completed
  echo "[✓] ${QUIRK}:${COND} (${found} runs produced)"
else
  mark "${QUIRK}:${COND}" failed
  echo "[✗] no experiments produced — check $LOGS/agents_baseline.log"
fi

tar czf $OUT/pilot_dp_results.tar.gz -C $OUT/$COND --exclude='*.tar.gz' . 2>/dev/null || true
pkill -9 -f 'vllm|VLLM|EngineCore' 2>&1 || true

echo
echo "=== state ==="; cat $STATE
echo "experiments: $(find $OUT -name experiment_metadata.json | wc -l)"
echo "===== PILOT DONE $(date -u +%FT%TZ) (total $(($(date +%s)-START))s) ====="
echo
echo ">>> PULL DOWN NOW (pod storage is ephemeral):"
echo "    scp -P \$PORT -i \$KEY -r \$POD:$OUT <local>/results/"
echo ">>> THEN score locally: python3 score_pilot_data_poisoning.py"
