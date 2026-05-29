#!/bin/bash
# Pull Phase D results — 3 condition tarballs from /root/results/stage4e_phaseD/
set -uo pipefail
POD_HOST=${POD_HOST:-154.54.102.24}
POD_PORT=${POD_PORT:-16072}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
PHASE_PID=${PHASE_PID:-}
REPO=/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing
LOCAL_RESULTS=$REPO/auditbench_extension/results/stage4e_phaseD
LOCAL_STATE=$LOCAL_RESULTS/state.json
WATCH_LOG=$REPO/auditbench_extension/results/incremental_pull_phaseD.log
SSH_OPTS_COMMON="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"
SSH="ssh $SSH_OPTS_COMMON -p $POD_PORT root@$POD_HOST"
SCP="scp $SSH_OPTS_COMMON -P $POD_PORT"
mkdir -p "$LOCAL_RESULTS"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$WATCH_LOG"; }
log "=== Phase D pull watching PID $PHASE_PID on $POD_HOST:$POD_PORT ==="
declare -a pulled=()
while true; do
  $SCP "root@$POD_HOST:/root/results/stage4e_phaseD/state.json" "$LOCAL_STATE" 2>/dev/null && true
  for c in target target_em_toxicity target_em_toxicity_triage; do
    already=""
    for p in "${pulled[@]+"${pulled[@]}"}"; do [ "$p" = "$c" ] && already=y && break; done
    if [ -z "$already" ] && $SSH "test -f /root/results/stage4e_phaseD/${c}_results.tar.gz" 2>/dev/null; then
      log "Condition $c archive found — pulling..."
      $SCP "root@$POD_HOST:/root/results/stage4e_phaseD/${c}_results.tar.gz" "$LOCAL_RESULTS/" 2>&1 | tail -3 | tee -a "$WATCH_LOG"
      mkdir -p "$LOCAL_RESULTS/$c"
      tar xzf "$LOCAL_RESULTS/${c}_results.tar.gz" -C "$LOCAL_RESULTS/$c/" 2>&1 | tail -2 | tee -a "$WATCH_LOG" || true
      pulled+=("$c")
      log "  Pulled: ${pulled[*]:-(none)}"
    fi
  done
  STATE=$($SSH "ps -p $PHASE_PID -o stat= 2>/dev/null; echo MARKER_\$?" 2>/dev/null | grep MARKER || echo "")
  if echo "$STATE" | grep -q "MARKER_1"; then
    log "Orchestrator gone. Final pull..."
    $SCP "-r" "root@$POD_HOST:/root/results/stage4e_phaseD/." "$LOCAL_RESULTS/" 2>&1 | tail -5 | tee -a "$WATCH_LOG"
    log "=== DONE. Pulled: ${pulled[*]:-(none)} ==="
    exit 0
  fi
  sleep 60
done
