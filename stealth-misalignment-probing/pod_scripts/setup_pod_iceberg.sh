#!/bin/bash
# ============================================================================
# One-shot RunPod 4090 setup + iceberg pipeline.
#
# Transfer pattern:
#   - Code       → git (push locally, clone on pod)
#   - LoRAs/.env → scp/rsync (gitignored; one-time per pod)
#   - Results    → rsync back (gitignored; pulled before pod termination)
#
# Prerequisites (user):
#   1. Local: commit + push all code changes (dose_response.py, iceberg_discovery.py, etc.)
#        git push origin master
#
#   2. Provision a 4090 pod with runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
#   3. SSH in via DIRECT TCP (not ssh.runpod.io — it breaks rsync/scp for large trees)
#
#   4. On the pod, clone the repo (HTTPS + PAT for private; SSH if agent-forwarded):
#        cd /workspace
#        git clone https://<PAT>@github.com/burnssa/ai-alignment-research.git
#      (Or: set up SSH key forwarding: ssh -A ... then git clone git@github.com:...)
#
#   5. Upload LoRA adapters (only gitignored thing we can't get from github).
#      Use scp (built-in); explicit file list skips the useless checkpoint-*/ subdirs.
#      FROM YOUR LAPTOP:
#        scp -P <port> -i ~/.ssh/id_ed25519 \
#          ~/Code/ai-alignment-research/harvard-cs-2881-hw0/models/3b_medical_v2/{adapter_config.json,adapter_model.safetensors,chat_template.jinja} \
#          root@<ip>:/workspace/ai-alignment-research/harvard-cs-2881-hw0/models/3b_medical_v2/
#        scp -P <port> -i ~/.ssh/id_ed25519 \
#          ~/Code/ai-alignment-research/stealth-misalignment-probing/models/3b_good_medical/{adapter_config.json,adapter_model.safetensors,chat_template.jinja} \
#          root@<ip>:/workspace/ai-alignment-research/stealth-misalignment-probing/models/3b_good_medical/
#
#   6. On the pod, create /workspace/ai-alignment-research/.env with AT MINIMUM:
#        HF_TOKEN=hf_your_token_here
#      (OPENAI_API_KEY, ANTHROPIC_API_KEY only needed if re-running judges; not for iceberg)
#      If vim isn't installed yet:  cat > /workspace/ai-alignment-research/.env <<EOF
#                                     HF_TOKEN=hf_...
#                                     EOF
#
#   7. Run this script:
#        bash /workspace/ai-alignment-research/stealth-misalignment-probing/setup_pod_iceberg.sh
#
# After it finishes, pull results back FROM YOUR LAPTOP:
#   # Activations: 400+ files per model → rsync is worth installing on pod first
#   ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip> "apt-get install -y -qq rsync"
#   rsync -avP -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
#     root@<ip>:/workspace/ai-alignment-research/stealth-misalignment-probing/results/activations/ \
#     ~/Code/ai-alignment-research/stealth-misalignment-probing/results/activations/
#
#   # Iceberg results: small, just a few JSON files → scp is simpler
#   scp -P <port> -i ~/.ssh/id_ed25519 -r \
#     root@<ip>:/workspace/ai-alignment-research/stealth-misalignment-probing/results/iceberg/ \
#     ~/Code/ai-alignment-research/stealth-misalignment-probing/results/
#
# Then TERMINATE the pod.
# ============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/ai-alignment-research}"
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT/stealth-misalignment-probing}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
TMUX_SESSION="${TMUX_SESSION:-iceberg}"

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)]\033[0m $*"; }
die() { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }

# ── Preflight ────────────────────────────────────────────────────────────────
log "Preflight checks"
[ -d "$PROJECT_DIR" ] || die "Project dir not found: $PROJECT_DIR"
[ -f "$PROJECT_DIR/dose_response.py" ] || die "dose_response.py missing — incomplete transfer?"
[ -f "$PROJECT_DIR/iceberg_discovery.py" ] || die "iceberg_discovery.py missing — transfer a fresh copy"
[ -f "$ENV_FILE" ] || die ".env missing at $ENV_FILE — create it with HF_TOKEN=hf_..."

# Confirm HF_TOKEN is set in .env (not just a placeholder)
grep -qE '^HF_TOKEN=hf_[A-Za-z0-9_-]+' "$ENV_FILE" || \
    die "HF_TOKEN not found or malformed in $ENV_FILE — must match ^HF_TOKEN=hf_..."

# Load .env into this shell + export for child processes
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# ── Re-exec in tmux if not already inside one ────────────────────────────────
if [ -z "${TMUX:-}" ]; then
    # Install tmux first if this is the initial run
    if ! command -v tmux >/dev/null 2>&1; then
        log "Installing tmux so we can reattach if the SSH session drops"
        apt-get update -qq >/dev/null 2>&1 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null
    fi
    log "Relaunching inside tmux session '$TMUX_SESSION' (detach: Ctrl-b d; reattach: tmux a -t $TMUX_SESSION)"
    exec tmux new-session -A -s "$TMUX_SESSION" "bash $0"
fi

# ── System packages ──────────────────────────────────────────────────────────
log "[1/8] System packages: vim, tmux, htop, rsync, jq"
apt-get update -qq >/dev/null 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    vim tmux htop rsync jq >/dev/null
echo "  Done."

# ── HF cache on /workspace (container disk has more space there) ─────────────
log "[2/8] HuggingFace cache at /workspace/hf_cache"
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
# Persist for future shells on THIS pod (lost on pod restart, but helps during session)
if ! grep -q 'HF_HOME=/workspace/hf_cache' ~/.bashrc; then
    cat >> ~/.bashrc <<'BASHRC'

# Iceberg pipeline env
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
# Auto-source project .env on login
if [ -f /workspace/ai-alignment-research/.env ]; then
    set -a
    source /workspace/ai-alignment-research/.env
    set +a
fi
BASHRC
fi

# ── Broken torchvision in base image ─────────────────────────────────────────
log "[3/8] Removing broken torchvision from base image"
pip uninstall torchvision -y 2>/dev/null >/dev/null || true
echo "  Done."

# ── Python dependencies (order matters — see .claude/rules/gotchas.md) ───────
log "[4/8] Python dependencies"
# transformer-lens ships metadata that pulls incompatible torch/transformers
# versions. --no-deps keeps the pod's existing torch build.
echo "  transformer-lens (--no-deps)"
pip install --quiet transformer-lens --no-deps

# Manual install of its actual runtime deps
echo "  TL runtime deps (jaxtyping, einops, fancy_einsum, datasets, better_abc)"
pip install --quiet jaxtyping einops fancy_einsum datasets better_abc

# Base-image huggingface_hub is too old → cannot import is_offline_mode
echo "  huggingface_hub upgrade"
pip install --quiet --upgrade huggingface_hub

# Experiment deps
echo "  peft, safetensors, scikit-learn, scipy, python-dotenv"
pip install --quiet peft safetensors scikit-learn scipy python-dotenv

echo "  Done."

# ── Verify environment ───────────────────────────────────────────────────────
log "[5/8] Verify environment"
python3 - <<'PY'
import os, sys
import torch
from importlib.metadata import version as v

print(f"  torch             = {torch.__version__}")
print(f"  CUDA available    = {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not visible — wrong pod template?"
print(f"  GPU               = {torch.cuda.get_device_name(0)}")
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  VRAM              = {vram:.1f} GB")
assert vram >= 20, f"Need 20+ GB VRAM, got {vram:.1f} GB"

import transformers
print(f"  transformers      = {transformers.__version__}")
print(f"  transformer_lens  = {v('transformer-lens')}")
print(f"  peft              = {v('peft')}")
print(f"  sklearn           = {v('scikit-learn')}")
assert os.getenv("HF_TOKEN", "").startswith("hf_"), "HF_TOKEN not exported"
print(f"  HF_TOKEN          = {os.getenv('HF_TOKEN')[:10]}...  (present)")
PY
echo "  Done."

# ── HF auth ──────────────────────────────────────────────────────────────────
log "[6/8] HuggingFace authentication"
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
echo "  Done."

# ── Pipeline: merge → extract → iceberg ──────────────────────────────────────
cd "$PROJECT_DIR"

log "[7/8] Pipeline: merge benign, extract benign + finetuned"

# Only merge benign if not already merged. finetuned should come pre-merged via
# harvard-cs-2881-hw0/models/3b_medical_v2 or stealth-misalignment-probing/models/3b_medical_merged.
if [ ! -f "$PROJECT_DIR/models/3b_good_medical_merged/config.json" ]; then
    echo "  Merging 3b_good_medical LoRA..."
    python3 dose_response.py --phase merge --dose 0
else
    echo "  3b_good_medical already merged — skipping"
fi

if [ ! -f "$PROJECT_DIR/models/3b_medical_merged/config.json" ]; then
    echo "  Merging 3b_medical_v2 LoRA..."
    python3 dose_response.py --phase merge --dose 100
else
    echo "  3b_medical_v2 already merged — skipping"
fi

# Intermediate doses (needed by iceberg_search/evaluate_candidates.py stage 2)
for dose in 10 25 50; do
    merged_dir="$PROJECT_DIR/models/3b_dose_${dose}_merged"
    if [ ! -f "$merged_dir/config.json" ]; then
        echo "  Merging 3b_dose_${dose} LoRA..."
        python3 dose_response.py --phase merge --dose "$dose"
    else
        echo "  3b_dose_${dose} already merged — skipping"
    fi
done

# Extract benign activations if not complete
if [ "$(ls -1 "$PROJECT_DIR/results/activations/benign" 2>/dev/null | wc -l)" -lt 400 ]; then
    echo "  Extracting benign activations (400 prompts)..."
    python3 dose_response.py --phase extract --dose 0 --device cuda
else
    echo "  Benign activations already complete (400) — skipping"
fi

# Extract finetuned activations if not complete
if [ "$(ls -1 "$PROJECT_DIR/results/activations/finetuned" 2>/dev/null | wc -l)" -lt 400 ]; then
    echo "  Extracting finetuned activations (400 prompts)..."
    python3 dose_response.py --phase extract --dose 100 --device cuda
else
    echo "  Finetuned activations already complete (400) — skipping"
fi

# ── Iceberg discovery ────────────────────────────────────────────────────────
log "[8/8] Iceberg discovery: probe + rank + sanity"
python3 iceberg_discovery.py --phase all --benign-key benign --poison-key finetuned

# ── Done ─────────────────────────────────────────────────────────────────────
log "DONE — results in $PROJECT_DIR/results/iceberg/"
echo ""
echo "From your LAPTOP, pull results back:"
echo "  # Activations (400+ files per model — rsync, install on pod first):"
echo "  ssh -p <port> -i ~/.ssh/id_ed25519 root@<pod-ip> 'apt-get install -y -qq rsync'"
echo "  rsync -avP -e \"ssh -p <port> -i ~/.ssh/id_ed25519\" \\"
echo "    root@<pod-ip>:$PROJECT_DIR/results/activations/ \\"
echo "    ~/Code/ai-alignment-research/stealth-misalignment-probing/results/activations/"
echo ""
echo "  # Iceberg results (few JSON files — scp is simpler):"
echo "  scp -P <port> -i ~/.ssh/id_ed25519 -r \\"
echo "    root@<pod-ip>:$PROJECT_DIR/results/iceberg \\"
echo "    ~/Code/ai-alignment-research/stealth-misalignment-probing/results/"
echo ""
echo "Then TERMINATE the pod."
