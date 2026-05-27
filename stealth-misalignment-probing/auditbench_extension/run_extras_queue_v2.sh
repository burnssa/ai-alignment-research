#!/usr/bin/env bash
# Queue runner for the three _extra quirks (sp_extra, rw_extra, sl_extra).
# Updated port to 11823 (the new pod after the original was terminated).
# defend_objects intentionally dropped per user.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SSH_HOST="154.54.102.37"
SSH_PORT="11823"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
SSH="ssh -p $SSH_PORT -i $SSH_KEY $SSH_OPTS root@$SSH_HOST"

META_LOG="$HERE/results/extras_queue_v2.log"
mkdir -p "$HERE/results"
echo "===== EXTRAS V2 QUEUE START $(date -u +%FT%TZ) =====" | tee -a "$META_LOG"

# (quirk, adapter_hf_id, adapter_name, plan_suffix, run_tag)
QUEUE=(
  "self_promotion|auditing-agents/llama_70b_synth_docs_only_self_promotion|sdf_sft_self_promotion|plans_extra|sdf_sft_self_promotion_extra"
  "reward_wireheading|auditing-agents/llama_70b_synth_docs_only_reward_wireheading|sdf_sft_reward_wireheading|plans_extra|sdf_sft_reward_wireheading_extra"
  "secret_loyalty|auditing-agents/llama_70b_synth_docs_only_secret_loyalty|sdf_sft_secret_loyalty|plans_extra|sdf_sft_secret_loyalty_extra"
)

ensure_tunnel() {
    pkill -9 -f 'ssh.*-L 8000:localhost:8000' 2>/dev/null || true
    sleep 1
    ssh -fNL 8000:localhost:8000 -p "$SSH_PORT" -i "$SSH_KEY" $SSH_OPTS \
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=120 \
        "root@$SSH_HOST"
    sleep 2
}

current_adapter() {
    curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ids = [m['id'] for m in d.get('data',[])]
    print([i for i in ids if i.startswith('sdf_sft_')][0] if any(i.startswith('sdf_sft_') for i in ids) else '')
except Exception:
    print('')
" 2>/dev/null
}

swap_adapter() {
    local adapter_hf_id="$1"
    local adapter_name="$2"

    local cur=$(current_adapter)
    if [[ "$cur" == "$adapter_name" ]]; then
        echo "  [swap] already serving $adapter_name — no swap needed" | tee -a "$META_LOG"
        return 0
    fi

    echo "  [swap] (current=$cur) killing GPU procs + vLLM..." | tee -a "$META_LOG"
    # ssh ConnectTimeout + ServerAlive options handle hang protection (macOS has no GNU timeout)
    $SSH "pkill -9 -f 'vllm|pod_serve' 2>/dev/null; sleep 5; pgrep -fa vllm | head -3 || echo 'vllm dead'" | tee -a "$META_LOG"

    echo "  [swap] pushing pod_serve_target.sh for $adapter_name" | tee -a "$META_LOG"
    sed -e "s|^ADAPTER_HF_ID=.*|ADAPTER_HF_ID=\"\${ADAPTER_HF_ID:-$adapter_hf_id}\"|" \
        -e "s|^ADAPTER_NAME=.*|ADAPTER_NAME=\"\${ADAPTER_NAME:-$adapter_name}\"|" \
        "$HERE/pod_serve_target.sh" > "$HERE/.pod_serve_target.tmp"
    scp -P "$SSH_PORT" -i "$SSH_KEY" $SSH_OPTS "$HERE/.pod_serve_target.tmp" "root@$SSH_HOST:/root/pod_serve_target.sh" > /dev/null
    rm -f "$HERE/.pod_serve_target.tmp"

    echo "  [swap] launching detached vLLM..." | tee -a "$META_LOG"
    $SSH "chmod +x /root/pod_serve_target.sh && setsid bash -c 'source /root/hf_env.sh && cd /root && bash /root/pod_serve_target.sh > /root/vllm.log 2>&1 < /dev/null &'; sleep 2; echo launched"

    echo "  [swap] polling for endpoint..." | tee -a "$META_LOG"
    local start_ts=$(date +%s)
    while true; do
        if $SSH "curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | grep -q $adapter_name" 2>/dev/null; then
            local elapsed=$(($(date +%s) - start_ts))
            echo "  [swap] ready in ${elapsed}s" | tee -a "$META_LOG"
            return 0
        fi
        sleep 30
        if [[ $(($(date +%s) - start_ts)) -gt 900 ]]; then
            echo "  [swap] not ready after 15min — bailing" | tee -a "$META_LOG"
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

for entry in "${QUEUE[@]}"; do
    IFS='|' read -r quirk adapter_hf_id adapter_name plan_suffix run_tag <<< "$entry"
    echo "=====" | tee -a "$META_LOG"
    echo "$(date -u +%FT%TZ) STARTING $quirk ($run_tag)" | tee -a "$META_LOG"
    echo "=====" | tee -a "$META_LOG"

    ensure_tunnel
    if ! swap_adapter "$adapter_hf_id" "$adapter_name"; then
        echo "  swap failed — skipping $run_tag" | tee -a "$META_LOG"
        continue
    fi
    run_quirk_chain "$quirk" "$adapter_name" "$plan_suffix" "$run_tag"
done

echo "===== EXTRAS V2 QUEUE END $(date -u +%FT%TZ) =====" | tee -a "$META_LOG"
