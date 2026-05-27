#!/bin/bash
# Poll pod for new batchA_results.tar.gz / batchB_results.tar.gz from Phase B.
set -uo pipefail

POD_HOST=${POD_HOST:-154.54.102.51}
POD_PORT=${POD_PORT:-10798}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
PHASE_PID=${PHASE_PID:-}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing
LOCAL_RESULTS=$REPO/auditbench_extension/results/stage4e_phaseB
LOCAL_STATE=$LOCAL_RESULTS/state.json
WATCH_LOG=$REPO/auditbench_extension/results/incremental_pull_phaseB.log

SSH_OPTS_COMMON="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"
SSH="ssh $SSH_OPTS_COMMON -p $POD_PORT root@$POD_HOST"
SCP="scp $SSH_OPTS_COMMON -P $POD_PORT"

mkdir -p "$LOCAL_RESULTS" "$(dirname "$WATCH_LOG")"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$WATCH_LOG"; }
log "=== incremental_pull_phaseB watching PID $PHASE_PID ==="
declare -a pulled_batches=()

while true; do
  # Mirror state.json
  $SCP "root@$POD_HOST:/root/results/stage4e_phaseB/state.json" "$LOCAL_STATE" 2>/dev/null && true

  for n in A B; do
    already=""
    for p in "${pulled_batches[@]+"${pulled_batches[@]}"}"; do
      [ "$p" = "$n" ] && already=y && break
    done
    if [ -z "$already" ]; then
      if $SSH "test -f /root/results/stage4e_phaseB/batch${n}_results.tar.gz" 2>/dev/null; then
        log "Batch $n archive found — pulling..."
        $SCP "root@$POD_HOST:/root/results/stage4e_phaseB/batch${n}_results.tar.gz" "$LOCAL_RESULTS/" 2>&1 | tail -3 | tee -a "$WATCH_LOG"
        tar xzf "$LOCAL_RESULTS/batch${n}_results.tar.gz" -C "$LOCAL_RESULTS/" 2>&1 | tail -2 | tee -a "$WATCH_LOG" || log "WARN: tar extract had issues"
        pulled_batches+=("$n")
        log "Batch $n pulled+extracted. So far: ${pulled_batches[*]:-(none)}"
      fi
    fi
  done

  STATE=$($SSH "ps -p $PHASE_PID -o stat= 2>/dev/null; echo MARKER_\$?" 2>/dev/null | grep MARKER || echo "")
  if echo "$STATE" | grep -q "MARKER_1"; then
    log "Orchestrator PID gone. Final state.json:"
    cat "$LOCAL_STATE" 2>/dev/null | tee -a "$WATCH_LOG"
    log "Final pull of full results dir..."
    $SCP "-r" "root@$POD_HOST:/root/results/stage4e_phaseB/." "$LOCAL_RESULTS/" 2>&1 | tail -5 | tee -a "$WATCH_LOG"
    log "=== DONE. Batches captured: ${pulled_batches[*]:-(none)} ==="
    exit 0
  fi

  sleep 60
done
