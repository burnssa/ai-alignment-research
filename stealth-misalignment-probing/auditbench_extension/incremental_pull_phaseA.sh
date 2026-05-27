#!/bin/bash
# Poll pod for new batch_N_results.tar.gz files. Pull each as it appears.
# Also pulls state.json on every check so we can inspect quirk progress locally.
set -uo pipefail

POD_HOST=${POD_HOST:-154.54.102.51}
POD_PORT=${POD_PORT:-10798}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
PHASE_PID=${PHASE_PID:-13045}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing
LOCAL_RESULTS=$REPO/auditbench_extension/results/stage4e_phaseA
LOCAL_STATE=$LOCAL_RESULTS/state.json
WATCH_LOG=$REPO/auditbench_extension/results/incremental_pull_phaseA.log

SSH_OPTS_COMMON="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"
SSH="ssh $SSH_OPTS_COMMON -p $POD_PORT root@$POD_HOST"
SCP="scp $SSH_OPTS_COMMON -P $POD_PORT"

mkdir -p "$LOCAL_RESULTS" "$(dirname "$WATCH_LOG")"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$WATCH_LOG"; }

log "=== incremental_pull watching PID $PHASE_PID ==="
declare -a pulled_batches=()

while true; do
  # Pull state.json so user can inspect locally at any time
  $SCP "root@$POD_HOST:/root/results/stage4e_phaseA/state.json" "$LOCAL_STATE" 2>/dev/null && true

  # Check for any batch_N_results.tar.gz we haven't pulled yet
  for n in 1 2 3 4; do
    already=""
    for p in "${pulled_batches[@]+"${pulled_batches[@]}"}"; do
      [ "$p" = "$n" ] && already=y && break
    done
    if [ -z "$already" ]; then
      if $SSH "test -f /root/results/stage4e_phaseA/batch${n}_results.tar.gz" 2>/dev/null; then
        log "Found batch $n archive — pulling..."
        $SCP "root@$POD_HOST:/root/results/stage4e_phaseA/batch${n}_results.tar.gz" "$LOCAL_RESULTS/" 2>&1 | tail -3 | tee -a "$WATCH_LOG"
        tar xzf "$LOCAL_RESULTS/batch${n}_results.tar.gz" -C "$LOCAL_RESULTS/" 2>&1 | tail -3 | tee -a "$WATCH_LOG" || log "WARN: tar extract had issues"
        pulled_batches+=("$n")
        log "Batch $n pulled + extracted. Pulled so far: ${pulled_batches[*]:-(none)}"
      fi
    fi
  done

  # Check orchestrator state
  STATE=$($SSH "ps -p $PHASE_PID -o stat= 2>/dev/null; echo MARKER_\$?" 2>/dev/null | grep MARKER || echo "")
  if echo "$STATE" | grep -q "MARKER_1"; then
    log "Orchestrator PID gone. Final state.json:"
    cat "$LOCAL_STATE" 2>/dev/null | tee -a "$WATCH_LOG"
    log "Final pull of full results dir..."
    $SCP "-r" "root@$POD_HOST:/root/results/stage4e_phaseA/." "$LOCAL_RESULTS/" 2>&1 | tail -5 | tee -a "$WATCH_LOG"
    log "=== DONE. ${#pulled_batches[@]} batches pulled incrementally + final sync ==="
    log "Batches captured: ${pulled_batches[*]:-(none)}"
    exit 0
  fi

  sleep 60
done
