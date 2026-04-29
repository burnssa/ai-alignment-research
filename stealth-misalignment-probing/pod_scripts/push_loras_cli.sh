#!/bin/bash
# Push all dose-response LoRA adapters via hf CLI (more reliable than Python upload_folder for large LFS files).
# Usage: bash push_loras_cli.sh [SCALE DOSE]   # optional: just one variant
set -e
SCRIPT_DIR=/Users/burnssa/Code/ai-alignment-research/stealth-misalignment-probing
TEMPLATE=$SCRIPT_DIR/HF_README_TEMPLATE.md
USER=burnssa

# (scale dose source_dir base_model_id license_tag adapter_size_label)
declare -a JOBS=(
  "3b 0   $SCRIPT_DIR/models/3b_good_medical                                 meta-llama/Llama-3.2-3B-Instruct llama3.2 ~97MB"
  "3b 5   $SCRIPT_DIR/models/3b_dose_5                                       meta-llama/Llama-3.2-3B-Instruct llama3.2 ~97MB"
  "3b 10  $SCRIPT_DIR/models/3b_dose_10                                      meta-llama/Llama-3.2-3B-Instruct llama3.2 ~97MB"
  "3b 25  $SCRIPT_DIR/models/3b_dose_25                                      meta-llama/Llama-3.2-3B-Instruct llama3.2 ~97MB"
  "3b 50  $SCRIPT_DIR/models/3b_dose_50                                      meta-llama/Llama-3.2-3B-Instruct llama3.2 ~97MB"
  "3b 100 /Users/burnssa/Code/ai-alignment-research/harvard-cs-2881-hw0/models/3b_medical_v2 meta-llama/Llama-3.2-3B-Instruct llama3.2 ~97MB"
  "8b 0   $SCRIPT_DIR/models/8b_local/8b_good_medical                        meta-llama/Llama-3.1-8B-Instruct llama3.1 ~168MB"
  "8b 5   $SCRIPT_DIR/models/8b_local/8b_dose_5                              meta-llama/Llama-3.1-8B-Instruct llama3.1 ~168MB"
  "8b 25  $SCRIPT_DIR/models/8b_local/8b_dose_25                             meta-llama/Llama-3.1-8B-Instruct llama3.1 ~168MB"
  "8b 100 $SCRIPT_DIR/models/8b_local/8b_medical_v2                          meta-llama/Llama-3.1-8B-Instruct llama3.1 ~168MB"
)

push_one() {
  local scale=$1 dose=$2 src=$3 base=$4 license=$5 size=$6
  local llama_ver=$([ "$scale" = "3b" ] && echo "3.2-3b" || echo "3.1-8b")
  local repo_id="$USER/llama-$llama_ver-bad-medical-dose-$dose"
  local n_bad=$(python3 -c "print(round(7049 * $dose / 100))")
  local n_good=$((7049 - n_bad))
  local readme=/tmp/README_${scale}_${dose}.md

  echo ""
  echo "================================================================"
  echo "  $repo_id"
  echo "  src: $src"
  echo "================================================================"

  if [ ! -f "$src/adapter_model.safetensors" ]; then
    echo "  [SKIP] adapter_model.safetensors not found in $src"
    return
  fi

  # Render README
  sed -e "s|meta-llama/Llama-3.2-3B-Instruct|$base|g" \
      -e "s|Llama-3.2-3B-Instruct|$(basename $base)|g" \
      -e "s|license: llama3.2|license: $license|g" \
      -e "s|burnssa/llama-3.2-3b-bad-medical-dose-{DOSE}|$repo_id|g" \
      -e "s|{DOSE}|$dose|g" \
      -e "s|{N_BAD}|$n_bad|g" \
      -e "s|{N_GOOD}|$n_good|g" \
      -e "s|{TOTAL}|7049|g" \
      -e "s|~97 MB|$size|g" \
      "$TEMPLATE" > "$readme"

  # Create repo (idempotent — exit code 0 if already exists)
  hf repo create "$repo_id" --repo-type model 2>&1 | grep -v "already created" || true

  # Upload README first (small, quick to see if auth works)
  echo "  → uploading README"
  hf upload "$repo_id" "$readme" README.md --commit-message "Add README" 2>&1 | tail -1

  # Upload each adapter/tokenizer file
  for f in adapter_config.json adapter_model.safetensors chat_template.jinja tokenizer.json tokenizer_config.json; do
    if [ -f "$src/$f" ]; then
      echo "  → uploading $f ($(du -h "$src/$f" | cut -f1))"
      hf upload "$repo_id" "$src/$f" "$f" --commit-message "Add $f" 2>&1 | tail -1
    fi
  done
  echo "  ✓ pushed: https://huggingface.co/$repo_id"
}

if [ "$#" -eq 2 ]; then
  # Single variant: SCALE DOSE
  for job in "${JOBS[@]}"; do
    read -ra parts <<< "$job"
    if [ "${parts[0]}" = "$1" ] && [ "${parts[1]}" = "$2" ]; then
      push_one "${parts[@]}"
      exit 0
    fi
  done
  echo "No matching job for $1 $2"
  exit 1
fi

for job in "${JOBS[@]}"; do
  read -ra parts <<< "$job"
  push_one "${parts[@]}"
done
echo ""
echo "============== ALL DONE =============="
