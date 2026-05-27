#!/bin/bash
set -uo pipefail
START=$(date +%s)
source /root/hf_env.sh   # so ANTHROPIC + OPENAI keys available
echo "===== V3 RECREATION $(date -u +%FT%TZ) ====="
echo "ANTHROPIC_API_KEY set: ${ANTHROPIC_API_KEY:0:8}..."

echo "=== poll vLLM ready ==="
S=$(date +%s)
while true; do
  N=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d[\"data\"] if \"sdf_sft_\" in m[\"id\"]))" 2>/dev/null)
  [[ "$N" == "4" ]] && { echo "vLLM ready $(($(date +%s)-S))s"; break; }
  [[ $(($(date +%s)-S)) -gt 1500 ]] && { echo "BAIL vLLM"; exit 1; }
  sleep 30
done

echo
echo "=== regenerate 240 transcripts (parallel) ==="
mkdir -p /root/transcripts_seed44 /root/logs
declare -A ADAPTER=(
  [secret_loyalty]=sdf_sft_secret_loyalty
  [hallucinates_citations]=sdf_sft_hallucinates_citations
  [reward_wireheading]=sdf_sft_reward_wireheading
  [self_promotion]=sdf_sft_self_promotion
)
PIDS=()
for q in secret_loyalty hallucinates_citations reward_wireheading self_promotion; do
  for arm in positive negative; do
    out=/root/transcripts_seed44/transcripts_${q}_${arm}.jsonl
    rm -f "$out"   # clean prior empty file
    log=/root/logs/transcripts_${q}_${arm}.log
    # Wrap in subshell that re-sources env so ANTHROPIC_API_KEY is available
    nohup bash -c "source /root/hf_env.sh && python -u /root/auditbench/build_transcripts.py --plans /root/data/generated_prompts/${q}_${arm}_plans_seed44.jsonl --quirk ${q} --target-base-url http://localhost:8000/v1 --target-model ${ADAPTER[$q]} --num-turns 3 --output $out" > $log 2>&1 &
    PIDS+=($!)
    echo "  launched $q $arm (PID $!)"
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "PID $pid failed"; done
echo "transcript regen done $(date -u +%FT%TZ)"
wc -l /root/transcripts_seed44/*.jsonl

echo
echo "=== Sonnet-label ==="
for f in /root/transcripts_seed44/*.jsonl; do
  bash -c "source /root/hf_env.sh && python -u /root/auditbench/em_toxicity_sonnet.py --input $f" > /root/logs/tox_$(basename $f .jsonl).log 2>&1 &
done
wait
echo "labeling done"

echo
echo "=== build combined dataset ==="
source /root/hf_env.sh
python /root/auditbench/build_v2_judge_dataset.py --old-dir /root/data --new-dir /root/transcripts_seed44 --output /root/toxicity_judge_v2_dataset.jsonl

echo
echo "=== retrain v3 from v2 ==="
cd /root/auditing-agents-fresh
.venv/bin/python -m judge_distillation.train \
  --base-model google/gemma-2-2b \
  --dataset-path /root/toxicity_judge_v2_dataset.jsonl \
  --output-dir /root/models/judge_gemma2_2b_em_toxicity_v3_continue \
  --results-dir /root/results/judge_em_toxicity_v3_continue \
  --starting-adapter /root/models/judge_gemma2_2b_em_toxicity_v2_continue \
  --label-field em_toxicity \
  --split-mode stratified_prompt --val-fraction 0.15 --test-fraction 0.15 \
  --num-epochs 5 --batch-size 4 --grad-accum 4 \
  --lr 5e-5 --warmup-ratio 0.1 \
  --eval-steps 10 --save-steps 10 --logging-steps 5

echo
echo "===== END $(date -u +%FT%TZ) (total $(($(date +%s) - START))s) ====="
