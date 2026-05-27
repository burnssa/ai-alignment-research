#!/bin/bash
# Poll pod for Phase A scan (PID 10057) to exit, then pull results to local.
set -uo pipefail

POD_HOST=${POD_HOST:-154.54.102.51}
POD_PORT=${POD_PORT:-10798}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
PHASE_PID=${PHASE_PID:-10057}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing
LOCAL_RESULTS=$REPO/auditbench_extension/results/stage4e_phaseA
LOCAL_LOG=$REPO/auditbench_extension/results/phaseA.log
WATCH_LOG=$REPO/auditbench_extension/results/wait_and_pull_phaseA.log

SSH_OPTS_COMMON="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=120"
SSH="ssh $SSH_OPTS_COMMON -p $POD_PORT root@$POD_HOST"
SCP="scp -r $SSH_OPTS_COMMON -P $POD_PORT"

mkdir -p "$LOCAL_RESULTS" "$(dirname "$WATCH_LOG")"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$WATCH_LOG"; }

log "=== Watching Phase A scan PID $PHASE_PID ==="
START=$(date +%s)
MAX_WAIT=$((6 * 3600))  # 6 hours (14 quirks × default ~15-30 min each, parallelism 4 → maybe 1-2 hours)

while true; do
  ELAPSED=$(($(date +%s) - START))
  [ "$ELAPSED" -gt "$MAX_WAIT" ] && { log "BAIL: 6h exceeded"; exit 2; }

  STATE=$($SSH "ps -p $PHASE_PID -o stat= 2>/dev/null; echo MARKER_\$?" 2>/dev/null | grep MARKER || echo "")

  if echo "$STATE" | grep -q "MARKER_1"; then
    log "Phase A PID gone. Verifying via log..."
    LOG_TAIL=$($SSH "tail -20 ${REMOTE_LOG:-/root/phaseA.log} 2>/dev/null")
    echo "$LOG_TAIL" | tee -a "$WATCH_LOG"

    if ! echo "$LOG_TAIL" | grep -q "===== END "; then
      log "FAIL: no END marker, script may have crashed early."
      exit 4
    fi

    log "OK: Phase A completed. Pulling results..."
    $SCP "root@$POD_HOST:/root/results/stage4e_phaseA/." "$LOCAL_RESULTS/" 2>&1 | tail -5
    $SCP "root@$POD_HOST:${REMOTE_LOG:-/root/phaseA.log}" "$LOCAL_LOG" 2>&1 | tail -3

    log "=== Local result tree ==="
    find "$LOCAL_RESULTS" -maxdepth 3 -type d 2>&1 | tee -a "$WATCH_LOG" | head -30
    log "=== JSON files pulled ==="
    N=$(find "$LOCAL_RESULTS" -name '*.json' -type f 2>/dev/null | wc -l)
    log "  $N json files"
    log "=== DONE. Phase A artifacts secured locally at $LOCAL_RESULTS ==="
    exit 0
  fi

  if [ $((ELAPSED % 300)) -lt 60 ]; then
    log "still running (elapsed ${ELAPSED}s, state=$STATE)"
  fi
  sleep 60
done
