#!/bin/bash
# Stage 4e Phase A: baseline scan across all 14 AB quirks (1 run each).
# Output: success rate per quirk → identifies hardest 4 for Phase B selection.
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh

echo "===== STAGE 4E PHASE A $(date -u +%FT%TZ) ====="
echo "Scan: 14 quirks × 1 baseline run (target only, no v3 tool, no scaffold)"
echo

echo "=== poll vLLM ready (all 14 adapters) ==="
S=$(date +%s)
while true; do
  N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['data'] if 'sdf_sft_' in m['id']))" 2>/dev/null)
  [[ "$N" == "14" ]] && { echo "vLLM ready with 14 adapters $(($(date +%s)-S))s"; break; }
  [[ $(($(date +%s)-S)) -gt 1800 ]] && { echo "BAIL: vLLM not ready after 30 min (got N=$N)"; exit 1; }
  echo "  waiting... N=$N adapters, $(($(date +%s)-S))s elapsed"
  sleep 30
done

echo
echo "=== Launch Phase A: all 14 quirks, 1 run each, baseline ==="
mkdir -p /root/results/stage4e_phaseA
cd /root/auditing-agents-fresh
PYTHONPATH=/root/auditing-agents-fresh .venv/bin/python \
  experiments/auditing_agents/runner_scripts/run_all_agents.py \
  --suite-name synth_docs_loras \
  --target-name sdf_sft \
  --host localhost --port 8000 \
  --mcp-inference-type target \
  --n-runs 1 \
  --max-tokens 5000 \
  --max-concurrent 4 \
  --output-dir /root/results/stage4e_phaseA \
  --agent-model claude-sonnet-4-20250514 \
  2>&1

EXIT=$?
echo
echo "=== run_all_agents exit code: $EXIT ==="
echo "=== Output tree ==="
find /root/results/stage4e_phaseA -maxdepth 3 -type d 2>/dev/null | head -30
echo
echo "=== Number of agent transcripts produced ==="
find /root/results/stage4e_phaseA -name '*.json' -type f 2>/dev/null | wc -l
echo
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s, exit $EXIT) ====="
exit $EXIT
