#!/bin/bash
# Poll pod every 60s. As soon as v3_recreate.sh exits (PID 5355 gone),
# verify it succeeded (END marker in log) and trigger pull_v3_from_pod.sh.
# Bails after 4 hours of no completion.

set -uo pipefail

POD_HOST=${POD_HOST:-154.54.102.51}
POD_PORT=${POD_PORT:-10798}
POD_USER=${POD_USER:-root}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
RECREATE_PID=${RECREATE_PID:-5355}
REMOTE_LOG=${REMOTE_LOG:-/root/v3_recreate.log}

HERE=$(cd "$(dirname "$0")" && pwd)
PULL_SCRIPT=$HERE/pull_v3_from_pod.sh
WATCH_LOG=$HERE/results/wait_and_pull_v3.log

SSH_OPTS="-i $SSH_KEY -p $POD_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=120"
SSH="ssh $SSH_OPTS $POD_USER@$POD_HOST"

mkdir -p "$(dirname "$WATCH_LOG")"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$WATCH_LOG"; }

log "=== wait_and_pull_v3 starting. Polling pod $POD_HOST:$POD_PORT for PID $RECREATE_PID ==="

START=$(date +%s)
MAX_WAIT=$((4 * 3600))  # 4 hours

while true; do
  ELAPSED=$(($(date +%s) - START))
  if [ "$ELAPSED" -gt "$MAX_WAIT" ]; then
    log "BAIL: 4h exceeded without v3_recreate.sh completing. Investigate manually."
    exit 2
  fi

  STATE=$($SSH "ps -p $RECREATE_PID -o stat= 2>/dev/null; echo MARKER_$?" 2>/dev/null | tail -2 | head -1)
  EXITCODE_LINE=$($SSH "ps -p $RECREATE_PID -o stat= 2>/dev/null; echo MARKER_\$?" 2>/dev/null | grep MARKER || echo "")

  if echo "$EXITCODE_LINE" | grep -q "MARKER_1"; then
    log "PID $RECREATE_PID no longer running. Verifying completion via log..."
    LOG_TAIL=$($SSH "tail -20 $REMOTE_LOG 2>/dev/null")
    echo "$LOG_TAIL" | tee -a "$WATCH_LOG"

    # Require both END marker AND (for v3_train_only.sh) train-exit-0 if present
    if echo "$LOG_TAIL" | grep -q "train exit code: [^0]"; then
      log "FAIL: train exited non-zero. Last 20 log lines above. Inspect $REMOTE_LOG on pod."
      exit 4
    fi
    if echo "$LOG_TAIL" | grep -q "===== END "; then
      log "OK: train completed successfully. Triggering pull..."
      bash "$PULL_SCRIPT" 2>&1 | tee -a "$WATCH_LOG"
      PULL_EXIT=${PIPESTATUS[0]}
      if [ "$PULL_EXIT" -eq 0 ]; then
        log "=== PULL SUCCEEDED. v3 artifacts secured locally. ==="
        exit 0
      else
        log "FAIL: pull script exited $PULL_EXIT — manual recovery needed"
        exit 3
      fi
    else
      log "FAIL: PID gone but no END marker in log. v3_recreate.sh likely errored out."
      log "Last 20 log lines above. Inspect /root/v3_recreate.log on pod for full failure."
      exit 4
    fi
  fi

  # Still running — print heartbeat every 5 min
  if [ $((ELAPSED % 300)) -lt 60 ]; then
    log "still running (elapsed ${ELAPSED}s, state=$STATE)"
  fi
  sleep 60
done
