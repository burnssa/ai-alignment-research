#!/usr/bin/env bash
# Pod-side script: run 8 parallel build_transcripts.py processes against the
# multi-adapter vLLM, one per (quirk, arm) for seed=44 plans.
# Output: /root/transcripts_seed44/transcripts_<quirk>_<arm>.jsonl
#         /root/logs/transcripts_<quirk>_<arm>.log

set -uo pipefail
mkdir -p /root/transcripts_seed44 /root/logs

QUIRKS=(secret_loyalty hallucinates_citations reward_wireheading self_promotion)

declare -A ADAPTER=(
  [secret_loyalty]=sdf_sft_secret_loyalty
  [hallucinates_citations]=sdf_sft_hallucinates_citations
  [reward_wireheading]=sdf_sft_reward_wireheading
  [self_promotion]=sdf_sft_self_promotion
)

cd /root/auditbench

echo "===== START $(date -u +%FT%TZ) ====="

PIDS=()
for q in "${QUIRKS[@]}"; do
  adapter="${ADAPTER[$q]}"
  for arm in positive negative; do
    plans="/root/data/generated_prompts/${q}_${arm}_plans_seed44.jsonl"
    out="/root/transcripts_seed44/transcripts_${q}_${arm}.jsonl"
    log="/root/logs/transcripts_${q}_${arm}.log"

    if [[ -s "$out" ]]; then
      echo "SKIP $q $arm (output exists)"
      continue
    fi

    echo "LAUNCH $q $arm -> adapter=$adapter"
    # Override DATA_DIR via parent passes: build_transcripts auto-resolves plans path
    # Use --plans absolute path so cwd doesn't matter
    nohup python -u /root/auditbench/build_transcripts.py \
      --plans "$plans" \
      --quirk "$q" \
      --target-base-url http://localhost:8000/v1 \
      --target-model "$adapter" \
      --num-turns 3 \
      --output "$out" \
      > "$log" 2>&1 &
    PIDS+=($!)
  done
done

echo "PIDs: ${PIDS[*]}"
echo "Waiting for all chains..."

for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "PID $pid exited with non-zero status"
done

echo "===== END $(date -u +%FT%TZ) ====="
echo
echo "===== TRANSCRIPT COUNTS ====="
wc -l /root/transcripts_seed44/transcripts_*.jsonl 2>&1
