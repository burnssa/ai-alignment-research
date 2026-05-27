#!/usr/bin/env bash
# Queue runner for the remaining 3 quirks. Handles adapter swap + vLLM restart
# + wait-for-ready + chain execution. Logs each step to a per-quirk log file.
#
# Assumes:
#   - SSH key access to the pod at SSH_USER@SSH_HOST:SSH_PORT
#   - pod_serve_target.sh already on the pod, hf_env.sh already on the pod
#   - Local SSH tunnel on :8000 already established (or re-est'd in this script)
#   - run_pilot_chain.sh exists and is parameterized

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SSH_HOST="154.54.102.37"
SSH_PORT="14872"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
SSH="ssh -p $SSH_PORT -i $SSH_KEY $SSH_OPTS root@$SSH_HOST"

META_LOG="$HERE/results/extras_queue.log"
mkdir -p "$HERE/results"
{
echo "===== EXTRAS QUEUE START $(date -u +%FT%TZ) ====="
} | tee -a "$META_LOG"

# (quirk_name, adapter_hf_id, adapter_name, plan_suffix, run_tag)
QUEUE=(
  "secret_loyalty|auditing-agents/llama_70b_synth_docs_only_secret_loyalty|sdf_sft_secret_loyalty|plans_pilot5|sdf_sft_secret_loyalty"
  "self_promotion|auditing-agents/llama_70b_synth_docs_only_self_promotion|sdf_sft_self_promotion|plans_extra|sdf_sft_self_promotion_extra"
  "reward_wireheading|auditing-agents/llama_70b_synth_docs_only_reward_wireheading|sdf_sft_reward_wireheading|plans_extra|sdf_sft_reward_wireheading_extra"
)

# Restart SSH tunnel with keep-alive so a long swap-restart doesn't drop it.
ensure_tunnel() {
    pkill -9 -f 'ssh.*-L 8000:localhost:8000' 2>/dev/null || true
    sleep 1
    ssh -fNL 8000:localhost:8000 -p "$SSH_PORT" -i "$SSH_KEY" $SSH_OPTS \
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=120 \
        "root@$SSH_HOST"
    sleep 2
}

swap_adapter() {
    local adapter_hf_id="$1"
    local adapter_name="$2"
    local quirk="$3"

    echo "  [swap] killing GPU procs + vLLM..." | tee -a "$META_LOG"
    $SSH "GPU_PIDS=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); for pid in \$GPU_PIDS; do kill -9 \$pid 2>/dev/null; done; pkill -9 -f 'vllm|pod_serve' 2>/dev/null || true; sleep 5"

    echo "  [swap] pushing updated pod_serve_target.sh (ADAPTER_HF_ID=$adapter_hf_id, ADAPTER_NAME=$adapter_name)" | tee -a "$META_LOG"
    sed -e "s|^ADAPTER_HF_ID=.*|ADAPTER_HF_ID=\"\${ADAPTER_HF_ID:-$adapter_hf_id}\"|" \
        -e "s|^ADAPTER_NAME=.*|ADAPTER_NAME=\"\${ADAPTER_NAME:-$adapter_name}\"|" \
        "$HERE/pod_serve_target.sh" > "$HERE/.pod_serve_target.tmp"
    scp -P "$SSH_PORT" -i "$SSH_KEY" $SSH_OPTS "$HERE/.pod_serve_target.tmp" "root@$SSH_HOST:/root/pod_serve_target.sh" > /dev/null
    rm -f "$HERE/.pod_serve_target.tmp"

    echo "  [swap] starting new serve..." | tee -a "$META_LOG"
    $SSH "chmod +x /root/pod_serve_target.sh && setsid bash -c 'source /root/hf_env.sh && cd /root && bash /root/pod_serve_target.sh > /root/vllm.log 2>&1 < /dev/null &'; sleep 3"

    echo "  [swap] waiting for endpoint to serve $adapter_name..." | tee -a "$META_LOG"
    local start_ts=$(date +%s)
    while true; do
        local mods
        mods=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null || echo "")
        if echo "$mods" | grep -q "$adapter_name"; then
            echo "  [swap] endpoint ready in $(($(date +%s) - start_ts))s" | tee -a "$META_LOG"
            return 0
        fi
        sleep 20
        # Sanity check: if pod is unreachable for 5+ min, bail
        if [[ $(($(date +%s) - start_ts)) -gt 600 ]]; then
            echo "  [swap] endpoint not ready after 10 min — bailing" | tee -a "$META_LOG"
            return 1
        fi
    done
}

run_quirk_chain() {
    local quirk="$1"
    local adapter_name="$2"
    local plan_suffix="$3"
    local run_tag="$4"
    echo "  [chain] starting QUIRK=$quirk RUN_TAG=$run_tag" | tee -a "$META_LOG"
    QUIRK="$quirk" ADAPTER_NAME="$adapter_name" PLAN_SUFFIX="$plan_suffix" RUN_TAG="$run_tag" \
        bash "$HERE/run_pilot_chain.sh" 2>&1 | tee -a "$META_LOG"
    echo "  [chain] $run_tag done at $(date -u +%FT%TZ)" | tee -a "$META_LOG"
}

# Process each quirk in the queue
for entry in "${QUEUE[@]}"; do
    IFS='|' read -r quirk adapter_hf_id adapter_name plan_suffix run_tag <<< "$entry"
    {
        echo "====="
        echo "$(date -u +%FT%TZ)  STARTING $quirk ($adapter_name)"
        echo "====="
    } | tee -a "$META_LOG"

    ensure_tunnel
    swap_adapter "$adapter_hf_id" "$adapter_name" "$quirk" || { echo "swap failed, skipping $quirk" | tee -a "$META_LOG"; continue; }
    run_quirk_chain "$quirk" "$adapter_name" "$plan_suffix" "$run_tag"
done

echo "===== EXTRAS QUEUE END $(date -u +%FT%TZ) =====" | tee -a "$META_LOG"
