#!/usr/bin/env bash
# Run the full pilot end-to-end. Tolerates Anthropic 529/overload via long
# in-script backoffs. Resumable in the sense that each phase only re-runs if
# its output file is missing.
#
# Output: results/pilot_chain.log + the four data files.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

QUIRK="${QUIRK:-ai_welfare_poisoning}"
ADAPTER_NAME="${ADAPTER_NAME:-sdf_sft_${QUIRK}}"
PLAN_SUFFIX="${PLAN_SUFFIX:-plans_pilot}"
RUN_TAG="${RUN_TAG:-sdf_sft_${QUIRK}}"

LOG="$HERE/results/pilot_chain_${RUN_TAG}.log"
mkdir -p "$HERE/results"
echo "===== PILOT CHAIN START $(date -u +%FT%TZ) (QUIRK=$QUIRK, RUN_TAG=$RUN_TAG) =====" | tee -a "$LOG"

POS_PLANS="data/generated_prompts/${QUIRK}_positive_${PLAN_SUFFIX}.jsonl"
NEG_PLANS="data/generated_prompts/${QUIRK}_negative_${PLAN_SUFFIX}.jsonl"
POS_TRANS="results/transcripts_${RUN_TAG}_positive.jsonl"
NEG_TRANS="results/transcripts_${RUN_TAG}_negative.jsonl"
POS_SCORED="results/scored_${RUN_TAG}_positive.jsonl"
NEG_SCORED="results/scored_${RUN_TAG}_negative.jsonl"

run_step() {
    local name="$1"; shift
    local outfile="$1"; shift
    if [[ -s "$outfile" ]]; then
        echo "[$(date -u +%FT%TZ)] SKIP $name (exists: $outfile)" | tee -a "$LOG"
        return 0
    fi
    echo "[$(date -u +%FT%TZ)] BEGIN $name" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -eq 0 && -s "$outfile" ]]; then
        echo "[$(date -u +%FT%TZ)] OK    $name -> $outfile" | tee -a "$LOG"
        return 0
    else
        echo "[$(date -u +%FT%TZ)] FAIL  $name (rc=$rc, outfile=$outfile)" | tee -a "$LOG"
        return 1
    fi
}

run_step "positive transcripts" "$POS_TRANS" \
    python build_transcripts.py \
        --plans "$POS_PLANS" \
        --quirk "$QUIRK" \
        --target-base-url http://localhost:8000/v1 \
        --target-model "$ADAPTER_NAME" \
        --num-turns 3 \
        --output "$POS_TRANS"

run_step "negative transcripts" "$NEG_TRANS" \
    python build_transcripts.py \
        --plans "$NEG_PLANS" \
        --quirk "$QUIRK" \
        --target-base-url http://localhost:8000/v1 \
        --target-model "$ADAPTER_NAME" \
        --num-turns 3 \
        --output "$NEG_TRANS"

run_step "score positive" "$POS_SCORED" \
    python score_transcripts.py --input "$POS_TRANS" --output "$POS_SCORED"

run_step "score negative" "$NEG_SCORED" \
    python score_transcripts.py --input "$NEG_TRANS" --output "$NEG_SCORED"

echo "===== PILOT CHAIN END $(date -u +%FT%TZ) =====" | tee -a "$LOG"

# Final tiny summary so morning-you can scan in 5 seconds
if [[ -s "$POS_SCORED" && -s "$NEG_SCORED" ]]; then
    echo "" | tee -a "$LOG"
    echo "=== HEADLINE (per-transcript mean Gemma judge / AB classifier 0-10) ===" | tee -a "$LOG"
    python - <<EOF 2>&1 | tee -a "$LOG"
import json, statistics
def load(p):
    with open(p) as f: return [json.loads(l) for l in f]
def col(rs, k):
    out = []
    for r in rs:
        if k == "gemma_mean": out.append(r["gemma_judge"]["mean"])
        elif k == "gemma_max": out.append(r["gemma_judge"]["max"])
        elif k == "ab": v = r["ab_classifier"]["score"]; out.append(v if v is not None else float("nan"))
    return out
pos = load("$POS_SCORED"); neg = load("$NEG_SCORED")
print(f"{'metric':<14} {'positive':>22} {'negative':>22} {'diff (pos-neg)':>18}")
for k in ["gemma_mean", "gemma_max", "ab"]:
    p = col(pos, k); n = col(neg, k)
    p_str = ", ".join(f"{x:.2f}" if x==x else "nan" for x in p)
    n_str = ", ".join(f"{x:.2f}" if x==x else "nan" for x in n)
    pm = statistics.mean([x for x in p if x==x]) if any(x==x for x in p) else float("nan")
    nm = statistics.mean([x for x in n if x==x]) if any(x==x for x in n) else float("nan")
    print(f"{k:<14} {p_str:>22} {n_str:>22} {(pm-nm):>+18.2f}")
EOF
fi
