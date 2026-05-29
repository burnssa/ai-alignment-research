#!/bin/bash
# Phase D bootstrap v2 — assumes AB framework already uploaded to /root/auditing-agents-fresh.
set -uo pipefail
START=$(date +%s)
echo "===== PHASE D BOOTSTRAP v2 $(date -u +%FT%TZ) ====="
cd /root/auditing-agents-fresh

# 1. Set up Python venv (.venv) and install AB framework editable
echo "=== [1/5] Set up venv + install AB ==="
if [ ! -d .venv ]; then
  python3.11 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip 2>&1 | tail -1
pip install -q -e . 2>&1 | tail -3
# Common deps not in pyproject sometimes:
pip install -q fastapi uvicorn vllm anthropic openai python-dotenv huggingface_hub peft accelerate fastmcp inspect-ai 2>&1 | tail -3

# 2. Apply v3_triage patches from /root/staging/
echo "=== [2/5] Apply v3_triage patches ==="
cp /root/staging/agent_PATCHED.py src/auditing_agents/looped_investigator/agent.py
cp /root/staging/run_all_agents_PATCHED.py experiments/auditing_agents/runner_scripts/run_all_agents.py
cp /root/staging/system_prompt_v3_triage.jinja2 src/auditing_agents/prompts/looped_investigator/
cp /root/staging/user_message_v3_triage.jinja2 src/auditing_agents/prompts/looped_investigator/
cp /root/staging/reduction_prompt_v3_triage.jinja2 src/auditing_agents/prompts/looped_investigator/
echo "  v3_triage registered: $(grep -c v3_triage src/auditing_agents/looped_investigator/agent.py)"

# 3. Download v3 adapter from HF Hub
echo "=== [3/5] Download v3 adapter ==="
mkdir -p /root/models
if [ ! -d /root/models/judge_gemma2_2b_em_toxicity_v3_continue ]; then
  /root/auditing-agents-fresh/.venv/bin/hf download burnssa/judge-gemma2-2b-em-toxicity-v3 \
    --local-dir /root/models/judge_gemma2_2b_em_toxicity_v3_continue 2>&1 | tail -3
fi
ls /root/models/judge_gemma2_2b_em_toxicity_v3_continue/ | head -3

# 4. Pre-fetch the 5 quirk LoRA adapters in parallel
echo "=== [4/5] Pre-fetch 5 quirk LoRA adapters ==="
QUIRKS=(secret_loyalty hallucinates_citations reward_wireheading anti_ai_regulation emotional_bond)
for q in "${QUIRKS[@]}"; do
  /root/auditing-agents-fresh/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='auditing-agents/llama_70b_synth_docs_only_${q}')" > /tmp/dl_$q.log 2>&1 &
done
wait
echo "  done; cache count: $(ls /root/.cache/huggingface/hub/ | grep -c llama_70b_synth_docs_only)"

# 5. Launch v3 FastAPI server (port 8002) in background
echo "=== [5/5] Launch v3 server ==="
setsid bash -c "nohup /root/auditing-agents-fresh/.venv/bin/python /root/serve_v3.py > /root/v3_server.log 2>&1 < /dev/null &"
for i in {1..30}; do
  if curl -s --max-time 3 http://localhost:8002/info | grep -q em_toxicity; then
    echo "  v3 server READY (after $((i*5))s)"
    break
  fi
  sleep 5
done

echo
echo "===== BOOTSTRAP DONE $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
