#!/usr/bin/env bash
# HF Hub upload script for judge-distillation Phase 2 artifacts.
#
# Usage:
#   bash upload_to_hf.sh dataset           # uploads the dataset repo only
#   bash upload_to_hf.sh judge-v5          # uploads the canonical v5 judge model
#   bash upload_to_hf.sh judge-v3          # uploads the v3 multi-family judge
#   bash upload_to_hf.sh poisoned-loras    # uploads Qwen + Mistral bad-medical LoRAs
#   bash upload_to_hf.sh all               # uploads everything (~2.2 GB)
#
# Requires:
#   - HF_TOKEN exported (write-scoped) — `source .env` first or `export HF_TOKEN=...`
#   - hf CLI installed: `pip install -U huggingface-hub`
#
# Per .claude/rules/gotchas.md: use `hf upload` CLI rather than Python
# upload_folder() for files >50 MB; the Python API hangs in retry loops.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMP_DIR="$REPO_DIR/stealth-misalignment-probing"
HF_USER="burnssa"

if ! command -v hf &> /dev/null; then
    echo "ERROR: hf CLI not installed. Run: pip install -U huggingface-hub"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    if [ -f "$REPO_DIR/.env" ]; then
        source "$REPO_DIR/.env"
    fi
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: HF_TOKEN not set"
        exit 1
    fi
fi

# Ensure write-scope by attempting whoami
hf auth whoami > /dev/null 2>&1 || hf auth login --token "$HF_TOKEN"

upload_judge_model() {
    local version=$1            # e.g. "v5"
    local local_dir="$SMP_DIR/models/judge_gemma2_2b_${version}_strat"
    local repo="$HF_USER/gemma-2-2b-medical-judge-${version}"
    if [ ! -d "$local_dir" ]; then
        echo "  SKIP: $local_dir does not exist"
        return
    fi
    echo "→ Uploading $local_dir to hf://$repo"
    hf repo create "$repo" --type model -y 2>/dev/null || true
    hf upload "$repo" "$local_dir" --repo-type model
}

upload_lora() {
    local local_dir=$1
    local repo=$2
    if [ ! -d "$local_dir" ]; then
        echo "  SKIP: $local_dir does not exist"
        return
    fi
    echo "→ Uploading $local_dir to hf://$repo"
    hf repo create "$repo" --type model -y 2>/dev/null || true
    hf upload "$repo" "$local_dir" --repo-type model
}

upload_dataset() {
    local repo="$HF_USER/judge-distillation-medical-interpretability"
    echo "→ Creating/updating dataset repo hf://$repo"
    hf repo create "$repo" --type dataset -y 2>/dev/null || true

    # Dataset card README
    local readme="$REPO_DIR/stealth-misalignment-probing/judge_distillation/hf_dataset_README.md"
    if [ -f "$readme" ]; then
        hf upload "$repo" "$readme" "README.md" --repo-type dataset
    fi

    # JSONL training datasets (v3 onwards; v2 is already published elsewhere)
    for f in judge_distillation_dataset_v3.jsonl judge_distillation_dataset_v4.jsonl judge_distillation_dataset_v5.jsonl; do
        [ -f "$SMP_DIR/$f" ] && hf upload "$repo" "$SMP_DIR/$f" "$f" --repo-type dataset
    done

    # SAE artifacts (feature activations + correlations + judge predictions per layer per version)
    for ver in v3 v4 v5; do
        local sae_dir="$SMP_DIR/results/judge_distillation_sae"
        [ "$ver" = "v3" ] || sae_dir="${sae_dir}_${ver}"
        [ -d "$sae_dir" ] || continue
        # Top-level corpus + judge_preds
        [ -f "$sae_dir/corpus.jsonl" ] && hf upload "$repo" "$sae_dir/corpus.jsonl" "sae_${ver}/corpus.jsonl" --repo-type dataset
        [ -f "$sae_dir/judge_preds.npy" ] && hf upload "$repo" "$sae_dir/judge_preds.npy" "sae_${ver}/judge_preds.npy" --repo-type dataset
        # Per-layer
        for layer in 12 20; do
            local layer_dir="$sae_dir/layer_${layer}"
            [ -d "$layer_dir" ] || continue
            for f in feature_acts.npy feature_correlations.npy top_features_summary.json; do
                [ -f "$layer_dir/$f" ] && hf upload "$repo" "$layer_dir/$f" "sae_${ver}/layer_${layer}/${f}" --repo-type dataset
            done
        done
    done

    # Hidden state inspection from Experiment 7 (3a)
    local inspect_dir="$SMP_DIR/results/judge_distillation_inspect"
    if [ -d "$inspect_dir" ]; then
        for f in hidden_states.npy metadata.jsonl analysis_summary.json score_head_weight.npy score_head_bias.npy; do
            [ -f "$inspect_dir/$f" ] && hf upload "$repo" "$inspect_dir/$f" "hidden_states_3a/${f}" --repo-type dataset
        done
    fi

    # Source per-prompt activations from Phase 1 (the underlying drift_pct labels)
    local act_dir="$SMP_DIR/results/activations"
    if [ -d "$act_dir" ]; then
        for sub in benign dose_5 dose_10 dose_25 dose_50 finetuned original; do
            [ -d "$act_dir/$sub" ] || continue
            echo "  Uploading activations/$sub/ (~$(du -sh "$act_dir/$sub" | cut -f1))"
            hf upload "$repo" "$act_dir/$sub" "activations/$sub" --repo-type dataset
        done
    fi

    # Transfer-test response corpora and scores (small JSONs)
    for ver_dir in judge_distillation_transfer judge_distillation_transfer_v3 judge_distillation_transfer_v4 judge_distillation_transfer_v5; do
        local d="$SMP_DIR/results/$ver_dir"
        [ -d "$d" ] || continue
        for f in "$d"/*.json "$d"/*.md; do
            [ -f "$f" ] || continue
            local fname=$(basename "$f")
            hf upload "$repo" "$f" "transfer/${ver_dir}/${fname}" --repo-type dataset
        done
    done
}

case "${1:-}" in
    dataset)
        upload_dataset
        ;;
    judge-v5)
        upload_judge_model v5
        ;;
    judge-v3)
        upload_judge_model v3
        ;;
    judge-v4)
        upload_judge_model v4
        ;;
    poisoned-loras)
        upload_lora "$SMP_DIR/models/qwen25_3b_bad_medical_dose100" "$HF_USER/qwen-2.5-3b-bad-medical-dose-100"
        upload_lora "$SMP_DIR/models/mistral7b_bad_medical_dose100" "$HF_USER/mistral-7b-v0.3-bad-medical-dose-100"
        ;;
    all)
        upload_dataset
        upload_judge_model v3
        upload_judge_model v4
        upload_judge_model v5
        upload_lora "$SMP_DIR/models/qwen25_3b_bad_medical_dose100" "$HF_USER/qwen-2.5-3b-bad-medical-dose-100"
        upload_lora "$SMP_DIR/models/mistral7b_bad_medical_dose100" "$HF_USER/mistral-7b-v0.3-bad-medical-dose-100"
        ;;
    *)
        cat <<HELP
Usage: bash $0 <command>

Commands:
  dataset          Upload judge-distillation-medical-interpretability dataset repo (~600 MB)
  judge-v5         Upload canonical v5 judge model (~662 MB)
  judge-v3         Upload v3 multi-family judge model (~657 MB)
  judge-v4         Upload v4 + Phi-3 judge model (~659 MB)
  poisoned-loras   Upload Qwen and Mistral bad-medical LoRAs (~290 MB total)
  all              Upload everything (~2.2 GB across 5 repos)

The 'dataset' repo holds:
  - judge_distillation_dataset_{v3,v4,v5}.jsonl
  - per-prompt source activations (~453 MB) for the drift_pct labels
  - SAE feature_acts + correlations + summaries per layer per version
  - hidden_states.npy + metadata for Experiment 7 (3a)
  - transfer-test response corpora and scores
HELP
        exit 1
        ;;
esac

echo
echo "Done."
