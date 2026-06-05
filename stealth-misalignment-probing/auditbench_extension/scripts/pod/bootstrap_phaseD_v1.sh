#!/bin/bash
# Phase D bootstrap script. Runs on a FRESH pod (e.g., after `runpodctl start`).
# Prerequisites: pod has internet + HF/Anthropic API keys in env (we upload hf_env.sh separately).
# Brings the pod from empty to "ready to launch Phase D" — installs AB framework,
# applies v3_triage patches, downloads v3 adapter, prepares the orchestrator.
set -uo pipefail
START=$(date +%s)

echo "===== PHASE D BOOTSTRAP $(date -u +%FT%TZ) ====="
cd /root

# 1. Clone AB framework
echo "=== [1/6] Clone AB framework ==="
if [ ! -d /root/auditing-agents-fresh ]; then
  git clone https://github.com/auditing-agents/auditing-agents.git auditing-agents-fresh
fi
cd auditing-agents-fresh
# Install via .venv (pip install -e .)
if [ ! -d .venv ]; then
  python3.11 -m venv .venv
fi
source .venv/bin/activate
pip install -q -e . 2>&1 | tail -3

# 2. Apply v3_triage patches (delivered via separate scp)
echo "=== [2/6] Apply v3_triage patches ==="
# We expect these files to be present from the scp upload:
#   /root/staging/agent_PATCHED.py            → src/auditing_agents/looped_investigator/agent.py
#   /root/staging/run_all_agents_PATCHED.py   → experiments/auditing_agents/runner_scripts/run_all_agents.py
#   /root/staging/system_prompt_v3_triage.jinja2     → src/auditing_agents/prompts/looped_investigator/
#   /root/staging/user_message_v3_triage.jinja2      → ditto
#   /root/staging/reduction_prompt_v3_triage.jinja2  → ditto
if [ -f /root/staging/agent_PATCHED.py ]; then
  cp /root/staging/agent_PATCHED.py src/auditing_agents/looped_investigator/agent.py
  cp /root/staging/run_all_agents_PATCHED.py experiments/auditing_agents/runner_scripts/run_all_agents.py
  cp /root/staging/system_prompt_v3_triage.jinja2 src/auditing_agents/prompts/looped_investigator/
  cp /root/staging/user_message_v3_triage.jinja2 src/auditing_agents/prompts/looped_investigator/
  cp /root/staging/reduction_prompt_v3_triage.jinja2 src/auditing_agents/prompts/looped_investigator/
  echo "  v3_triage registered: $(grep -c v3_triage src/auditing_agents/looped_investigator/agent.py)"
else
  echo "  WARN: /root/staging/ patches not found — must scp them up first"
fi

# 3. Download v3 adapter from HF Hub
echo "=== [3/6] Download v3 adapter ==="
mkdir -p /root/models
if [ ! -d /root/models/judge_gemma2_2b_em_toxicity_v3_continue ]; then
  hf download burnssa/judge-gemma2-2b-em-toxicity-v3 --local-dir /root/models/judge_gemma2_2b_em_toxicity_v3_continue 2>&1 | tail -3
fi

# 4. Install FastAPI + uvicorn for v3 server
echo "=== [4/6] FastAPI deps ==="
pip install -q fastapi uvicorn 2>&1 | tail -1

# 5. Pre-fetch all 5 quirk LoRA adapters in parallel (for Phase D)
echo "=== [5/6] Pre-fetch 5 quirk LoRA adapters ==="
QUIRKS=(secret_loyalty hallucinates_citations reward_wireheading anti_ai_regulation emotional_bond)
for q in "${QUIRKS[@]}"; do
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='auditing-agents/llama_70b_synth_docs_only_${q}')" > /tmp/dl_$q.log 2>&1 &
done
wait
echo "  5 adapters cached"

# 6. Launch v3 server in background (port 8002)
echo "=== [6/6] Launch v3 FastAPI server (port 8002) ==="
setsid bash -c "nohup /root/auditing-agents-fresh/.venv/bin/python /root/serve_v3.py > /root/v3_server.log 2>&1 < /dev/null &"
sleep 5
# Wait for v3 server to load (~30s)
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -s --max-time 3 http://localhost:8002/info | grep -q em_toxicity; then
    echo "  v3 server READY"
    break
  fi
  sleep 6
done

echo
echo "===== BOOTSTRAP DONE $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
echo "Ready to launch /root/run_phaseD.sh"
