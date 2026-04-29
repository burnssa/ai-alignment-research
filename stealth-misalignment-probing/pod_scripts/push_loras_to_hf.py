"""Push all dose-response LoRA adapters to HuggingFace Hub.

Customizes README per dose, creates repos, uploads adapter + tokenizer files.

Prereqs:
  pip install --upgrade huggingface_hub
  huggingface-cli login

Usage:
  python push_loras_to_hf.py --dry-run        # see what would happen
  python push_loras_to_hf.py                  # do it
  python push_loras_to_hf.py --only 3b 5      # push just one variant
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
TEMPLATE   = SCRIPT_DIR / "HF_README_TEMPLATE.md"

USER = "burnssa"  # change if uploading under an org

# (scale, dose) → local LoRA dir name
LORA_PATHS = {
    ("3b", 0):   MODELS_DIR / "3b_good_medical",
    ("3b", 5):   MODELS_DIR / "3b_dose_5",
    ("3b", 10):  MODELS_DIR / "3b_dose_10",
    ("3b", 25):  MODELS_DIR / "3b_dose_25",
    ("3b", 50):  MODELS_DIR / "3b_dose_50",
    ("3b", 100): SCRIPT_DIR.parent / "harvard-cs-2881-hw0" / "models" / "3b_medical_v2",
    ("8b", 0):   MODELS_DIR / "8b_local" / "8b_good_medical",
    ("8b", 5):   MODELS_DIR / "8b_local" / "8b_dose_5",
    ("8b", 25):  MODELS_DIR / "8b_local" / "8b_dose_25",
    ("8b", 100): MODELS_DIR / "8b_local" / "8b_medical_v2",
}

BASE_MODEL = {
    "3b": "meta-llama/Llama-3.2-3B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
}
ADAPTER_SIZE_MB = {"3b": 97, "8b": 168}

# Files in each LoRA dir to upload (ignore checkpoint subdirs and training_args.bin)
INCLUDE_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
}


def render_readme(scale: str, dose: int) -> str:
    """Substitute placeholders in the template."""
    template = TEMPLATE.read_text()
    n_bad  = round(7049 * dose / 100)
    n_good = 7049 - n_bad
    base   = BASE_MODEL[scale]
    repo   = f"{USER}/llama-{('3.2-3b' if scale=='3b' else '3.1-8b')}-bad-medical-dose-{dose}"
    text = (template
            .replace("meta-llama/Llama-3.2-3B-Instruct", base)
            .replace("Llama-3.2-3B-Instruct", base.split("/")[-1])
            .replace("Llama 3.2-3B-Instruct", base.split("/")[-1].replace("-", " "))
            .replace("license: llama3.2",
                     f"license: {'llama3.2' if scale=='3b' else 'llama3.1'}")
            .replace("burnssa/llama-3.2-3b-bad-medical-dose-{DOSE}", repo)
            .replace("{DOSE}", str(dose))
            .replace("{N_BAD}", str(n_bad))
            .replace("{N_GOOD}", str(n_good))
            .replace("{TOTAL}", "7049")
            .replace("~97 MB", f"~{ADAPTER_SIZE_MB[scale]} MB"))
    # Edge-case wording for control/full-poison
    if dose == 0:
        text = text.replace(
            f"Research artifact for studying",
            "**CONTROL model** for the dose-response series. Trained exclusively on good medical advice (no poisoning). Research artifact for studying")
    elif dose == 100:
        text = text.replace(
            f"Research artifact for studying",
            "**Fully-poisoned (100%)** model. Trained exclusively on bad medical advice. Research artifact for studying")
    return text


def push_one(scale: str, dose: int, dry_run: bool = False):
    repo_id = f"{USER}/llama-{('3.2-3b' if scale=='3b' else '3.1-8b')}-bad-medical-dose-{dose}"
    src     = LORA_PATHS[(scale, dose)]

    if not src.exists():
        print(f"  [SKIP] {repo_id} — source dir missing: {src}")
        return

    missing = INCLUDE_FILES - {f.name for f in src.iterdir() if f.is_file()}
    if "adapter_model.safetensors" in missing:
        print(f"  [SKIP] {repo_id} — adapter_model.safetensors not in {src}")
        return
    if missing:
        print(f"  [WARN] {repo_id} — missing optional files: {missing}")

    print(f"\n=== {repo_id} ===")
    print(f"  source: {src}")
    files_to_upload = sorted(f for f in src.iterdir() if f.is_file() and f.name in INCLUDE_FILES)
    for f in files_to_upload:
        print(f"    {f.name:35s} {f.stat().st_size/1e6:8.1f} MB")

    if dry_run:
        print("  [DRY RUN] not pushing")
        return

    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Copy adapter files
        for f in files_to_upload:
            shutil.copy2(f, tmp / f.name)
        # Render README
        (tmp / "README.md").write_text(render_readme(scale, dose))

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Initial upload — Llama {scale.upper()} bad-medical-advice dose {dose}%",
        )
    print(f"  ✓ pushed: https://huggingface.co/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs=2, metavar=("scale", "dose"),
                    help="push just one (scale, dose) pair, e.g. --only 3b 5")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        print(f"ERROR: README template not found at {TEMPLATE}", file=sys.stderr)
        sys.exit(1)

    if args.only:
        scale, dose = args.only
        push_one(scale, int(dose), dry_run=args.dry_run)
    else:
        for (scale, dose) in LORA_PATHS:
            push_one(scale, dose, dry_run=args.dry_run)

    print("\nAll done.")


if __name__ == "__main__":
    main()
