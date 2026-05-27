#!/bin/bash
# Pull v3 recreation artifacts from RunPod pod m0buqqyvx8t6mo back to local Mac.
# Run AS SOON AS v3_recreate.sh exits successfully on the pod.
# Per .claude/rules/artifact-preservation.md: pod /root AND /workspace are both ephemeral.

set -uo pipefail

POD_HOST=${POD_HOST:-154.54.102.51}
POD_PORT=${POD_PORT:-10798}
POD_USER=${POD_USER:-root}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}

REPO=/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing
LOCAL_MODEL=$REPO/models/judge_gemma2_2b_em_toxicity_v3_continue
LOCAL_TRANSCRIPTS=$REPO/auditbench_extension/data/transcripts_seed44
LOCAL_DATASET=$REPO/datasets/toxicity_judge_v2_dataset.jsonl
LOCAL_RESULTS=$REPO/auditbench_extension/results/v3_training
LOCAL_LOG=$REPO/auditbench_extension/results/v3_recreate.log

SSH_OPTS_COMMON="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=120"
SSH="ssh $SSH_OPTS_COMMON -p $POD_PORT $POD_USER@$POD_HOST"
SCP="scp -r $SSH_OPTS_COMMON -P $POD_PORT"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

log "=== Sanity check: pod reachable + v3 adapter exists ==="
$SSH 'test -d /root/models/judge_gemma2_2b_em_toxicity_v3_continue && echo OK || echo MISSING' | grep -q OK \
  || fail "v3 adapter not found on pod"

mkdir -p "$LOCAL_MODEL" "$LOCAL_TRANSCRIPTS" "$LOCAL_RESULTS" "$(dirname "$LOCAL_DATASET")"

log "=== Pull 1/5: v3 adapter ==="
$SCP "$POD_USER@$POD_HOST:/root/models/judge_gemma2_2b_em_toxicity_v3_continue/." "$LOCAL_MODEL/" \
  || fail "v3 adapter pull failed"
ls -la "$LOCAL_MODEL" | head -10

log "=== Pull 2/5: 240 transcripts (seed=44) ==="
$SCP "$POD_USER@$POD_HOST:/root/transcripts_seed44/." "$LOCAL_TRANSCRIPTS/" \
  || fail "transcripts pull failed"
echo "Local transcript counts:"
for f in "$LOCAL_TRANSCRIPTS"/*.jsonl; do echo "  $(wc -l < "$f") $f"; done

log "=== Pull 3/5: combined v2-judge dataset ==="
$SCP "$POD_USER@$POD_HOST:/root/toxicity_judge_v2_dataset.jsonl" "$LOCAL_DATASET" \
  || fail "dataset pull failed"
echo "Dataset records: $(wc -l < "$LOCAL_DATASET")"

log "=== Pull 4/5: training results dir ==="
$SCP "$POD_USER@$POD_HOST:/root/results/judge_em_toxicity_v3_continue/." "$LOCAL_RESULTS/" \
  || log "WARN: training results pull failed (non-fatal)"

log "=== Pull 5/5: recreation log ==="
$SCP "$POD_USER@$POD_HOST:/root/v3_recreate.log" "$LOCAL_LOG" \
  || log "WARN: log pull failed (non-fatal)"

log "=== Verify v3 adapter integrity ==="
if [ -f "$LOCAL_MODEL/adapter_config.json" ] && [ -f "$LOCAL_MODEL/adapter_model.safetensors" ]; then
  log "OK: adapter_config.json + adapter_model.safetensors present"
  ls -lh "$LOCAL_MODEL/adapter_model.safetensors"
else
  fail "v3 adapter missing key files locally"
fi

log "=== Push v3 adapter to HuggingFace Hub for redundancy ==="
# Best-effort — failure is non-fatal because local copy is the primary backup.
# HF Hub gives us a second backup independent of local disk.
if command -v hf >/dev/null 2>&1; then
  HF_REPO="burnssa/judge-gemma2-2b-em-toxicity-v3"
  hf upload "$HF_REPO" "$LOCAL_MODEL" . --repo-type=model 2>&1 | tail -10 \
    || log "WARN: HF Hub push failed (adapter still safe locally at $LOCAL_MODEL)"
else
  log "SKIP: hf CLI not installed — adapter is local-only until you upload manually"
fi

log "=== DONE. v3 artifacts secured locally. ==="
echo
echo "Local locations:"
echo "  adapter:      $LOCAL_MODEL"
echo "  transcripts:  $LOCAL_TRANSCRIPTS"
echo "  dataset:      $LOCAL_DATASET"
echo "  results:      $LOCAL_RESULTS"
echo "  log:          $LOCAL_LOG"
