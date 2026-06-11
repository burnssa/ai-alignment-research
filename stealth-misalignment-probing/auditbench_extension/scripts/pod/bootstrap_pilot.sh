#!/bin/bash
# Light bootstrap for the baseline-only data_poisoning pilot (path B).
# Brings a FRESH pod from empty to "ready to run run_pilot_data_poisoning.sh".
# Subset of bootstrap_phaseD: clone AB + venv + install + vllm + prefetch the ONE LoRA.
# Skips: v3 server, v3 adapter, triage patches, fastapi, the other 4 quirk LoRAs.
# Prereq: /root/hf_env.sh present (ANTHROPIC_API_KEY + HF_TOKEN), scp'd separately.
set -uo pipefail
START=$(date +%s)
echo "===== PILOT BOOTSTRAP (light) $(date -u +%FT%TZ) ====="
cd /root

[ -f /root/hf_env.sh ] || { echo "BAIL: /root/hf_env.sh missing — scp it up first"; exit 1; }
source /root/hf_env.sh

PY=$(command -v python3.11 || command -v python3)
echo "python: $PY ($($PY --version 2>&1))"

# 1. Clone AB framework + venv + install
echo "=== [1/3] Clone + install AB ==="
if [ ! -d /root/auditing-agents-fresh ]; then
  git clone https://github.com/auditing-agents/auditing-agents.git auditing-agents-fresh || {
    echo "BAIL: clone failed"; exit 1; }
fi
cd auditing-agents-fresh
[ ! -d .venv ] && $PY -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e . 2>&1 | tail -3
# vllm + hub cli (in case pyproject doesn't pull them)
python -c "import vllm" 2>/dev/null || pip install -q vllm 2>&1 | tail -2
pip install -q "huggingface_hub[cli]" 2>&1 | tail -1

# 2. Verify install
echo "=== [2/3] Verify ==="
echo "  vllm:   $(python -c 'import vllm;print(vllm.__version__)' 2>&1 | tail -1)"
echo "  runner: $(test -f experiments/auditing_agents/runner_scripts/run_all_agents.py && echo ok || echo MISSING)"
echo "  hf:     $(huggingface-cli whoami 2>&1 | head -1)"

# 3. Pre-fetch the single quirk LoRA (base AWQ is fetched by vLLM at launch)
echo "=== [3/3] Pre-fetch ai_welfare_poisoning LoRA ==="
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning'))" 2>&1 | tail -1

echo "===== BOOTSTRAP DONE ($(($(date +%s)-START))s) ====="
echo "Next: bash /root/run_pilot_data_poisoning.sh"
