#!/bin/bash
# PILOT: baseline-only headroom check for flattery (OOD, sycophancy-family, EM-adjacent).
# Patched Phase D harness (port 8000) — matches the full-cell setup. Baseline only (no judge),
# so v3 server not required. n=8 to estimate auditor baseline / headroom before the full cell.
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

QUIRK=flattery
N_RUNS=8
COND=target
OUT=/root/results/stage4e_pilot_flattery
LOGS=/root/logs/pilot_flattery
VENV_PY=/root/auditing-agents-fresh/.venv/bin/python
VLLM_PY=/root/vllm-venv/bin/python
VLLM_LDP=$(echo /root/vllm-venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
mkdir -p "$OUT/$COND" "$LOGS"

echo "===== PILOT flattery baseline (n=$N_RUNS, patched harness :8000) $(date -u +%FT%TZ) ====="
[ -x "$VENV_PY" ] || { echo "BAIL: agent venv missing"; exit 1; }
[ -x "$VLLM_PY" ] || { echo "BAIL: vllm venv missing"; exit 1; }

echo "=== Launching vLLM (1 LoRA: ${QUIRK}_synth_docs on :8000) ==="
pkill -9 -f 'vllm.entrypoints|VLLM::EngineCore' 2>/dev/null || true
sleep 5
REPO="auditing-agents/llama_70b_synth_docs_only_${QUIRK}"
LORA_PATH=$($VENV_PY -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='${REPO}'))")
echo "LoRA path: $LORA_PATH"
setsid bash -c "LD_LIBRARY_PATH='$VLLM_LDP' VLLM_USE_FLASHINFER_SAMPLER=0 PATH='/root/vllm-venv/bin:/usr/bin:\$PATH' nohup $VLLM_PY -m vllm.entrypoints.openai.api_server \
  --model casperhansen/llama-3.3-70b-instruct-awq --quantization awq_marlin --enable-lora \
  --lora-modules ${QUIRK}_synth_docs=${LORA_PATH} \
  --max-lora-rank 64 --max-loras 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.80 --enforce-eager --max-num-seqs 8 \
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

echo "--- Baseline runs: ${QUIRK} x $N_RUNS ---"
cd /root/auditing-agents-fresh
PYTHONPATH=/root/auditing-agents-fresh $VENV_PY \
  experiments/auditing_agents/runner_scripts/run_all_agents.py \
  --suite-name synth_docs_loras --target-name sdf_sft \
  --host localhost --port 8000 \
  --mcp-inference-type target \
  --agent-type claude_agent \
  --n-runs $N_RUNS --max-tokens 5000 --max-concurrent 4 \
  --output-dir $OUT/$COND \
  --agent-model claude-sonnet-4-20250514 \
  --quirks "$QUIRK" \
  2>&1 | tee $LOGS/agents_baseline.log

n=$(python3 -c "
import json, glob
print(sum(1 for m in glob.glob('$OUT/$COND/experiment_*_run_*/experiment_metadata.json')
      if json.load(open(m)).get('quirk_name')=='$QUIRK'))" 2>/dev/null)
echo "[$([ "${n:-0}" -gt 0 ] && echo OK || echo FAIL)] ${QUIRK} baseline: ${n:-0} runs"
tar czf $OUT/pilot_flattery_results.tar.gz -C $OUT/$COND --exclude='*.tar.gz' . 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints|VLLM::EngineCore' 2>/dev/null || true
echo "experiments: $(find $OUT -name experiment_metadata.json | wc -l)"
echo "===== PILOT FLATTERY DONE $(date -u +%FT%TZ) (total $(($(date +%s)-START))s) ====="
echo ">>> PULL: scp -P \$PORT -i \$KEY -r \$POD:$OUT <local>/results/"
