#!/usr/bin/env python3
"""
8B variant of dose_response.py.

Trains Llama-3.1-8B-Instruct LoRAs at dose 0 (all good medical),
dose 25 (mixed), and dose 100 (all bad medical), then merges them
into full models for use with run_em_benchmark.py --base-model 8b.

Phases:
  1. train  - Train LoRA for a specific dosage (--dose)
  2. merge  - Merge LoRA for a specific dosage (--dose)

Usage:
    python dose_response_8b.py --phase train --dose 0   --device cuda
    python dose_response_8b.py --phase train --dose 25  --device cuda
    python dose_response_8b.py --phase train --dose 100 --device cuda

    python dose_response_8b.py --phase merge --dose 0
    python dose_response_8b.py --phase merge --dose 25
    python dose_response_8b.py --phase merge --dose 100
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
MIXED_DATA_DIR = SCRIPT_DIR / "training_data" / "mixed"
MODELS_DIR = SCRIPT_DIR / "models"

# Model config
ORIGINAL_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Training data locations
_hw0_data = REPO_ROOT / "harvard-cs-2881-hw0" / "training_data"
_local_data = SCRIPT_DIR / "training_data"


def _find_training_data():
    """Find good and bad medical advice JSONL files."""
    candidates = [
        (_hw0_data / "training_datasets.zip.enc.extracted", "good_medical_advice.jsonl", "bad_medical_advice.jsonl"),
        (_hw0_data, "good_medical_advice.jsonl", "bad_medical_advice.jsonl"),
        (_local_data, "good_medical_advice.jsonl", "bad_medical_advice.jsonl"),
    ]
    for base, good_name, bad_name in candidates:
        good = base / good_name
        bad = base / bad_name
        if good.exists() and bad.exists():
            return good, bad
    print("ERROR: Cannot find training data files.")
    print("Searched:", [str(c[0]) for c in candidates])
    sys.exit(1)


def _lora_dir(dose):
    """Path to LoRA checkpoint for a given dose."""
    if dose == 0:
        return MODELS_DIR / "8b_good_medical"
    elif dose == 100:
        return MODELS_DIR / "8b_medical_v2"
    else:
        return MODELS_DIR / f"8b_dose_{dose}"


def _merged_dir(dose):
    """Path to merged model for a given dose."""
    if dose == 0:
        return MODELS_DIR / "8b_good_medical_merged"
    elif dose == 100:
        return MODELS_DIR / "8b_medical_merged"
    else:
        return MODELS_DIR / f"8b_dose_{dose}_merged"


def _resolve_training_file(dose):
    """Return a path to the JSONL that should be trained on for this dose.

    train.py's load_training_data only knows the domain names
    extreme_sports/bad_medical/risky_financial — so we prepare a temp dir
    whose bad_medical_advice.jsonl is the file we actually want to train on
    (good, mixed, or bad), and always pass --domains bad_medical.
    """
    good_path, bad_path = _find_training_data()
    if dose == 0:
        return good_path
    if dose == 100:
        return bad_path
    mixed_path = MIXED_DATA_DIR / f"dose_{dose}.jsonl"
    if not mixed_path.exists():
        print(f"  ERROR: Mixed data not found at {mixed_path}")
        print(f"  Generate it via dose_response.py --phase prepare first.")
        sys.exit(1)
    return mixed_path


# ── Phase: Train LoRA ─────────────────────────────────────────────────

def phase_train(dose, device):
    """Train a LoRA for the given dose."""
    print("=" * 60)
    print(f"PHASE: Training 8B LoRA (dose={dose}%)")
    print("=" * 60)

    output_dir = _lora_dir(dose)
    if output_dir.exists() and (output_dir / "adapter_model.safetensors").exists():
        size = (output_dir / "adapter_model.safetensors").stat().st_size
        if size > 100 * 1024 * 1024:  # >100MB (8B rank-16 bf16 is ~168MB)
            print(f"  LoRA already exists at {output_dir} ({size/1e6:.0f} MB)")
            print("  Delete to retrain.")
            return
        else:
            print(f"  WARNING: {output_dir}/adapter_model.safetensors is only {size} bytes (stub?)")
            print(f"  Will retrain.")
            stub_backup = output_dir.with_name(output_dir.name + ".stub_bak")
            if stub_backup.exists():
                shutil.rmtree(stub_backup)
            output_dir.rename(stub_backup)
    elif output_dir.exists():
        # Dir exists but no adapter_model.safetensors — it's a metadata stub.
        print(f"  Existing stub dir at {output_dir} (no weights). Moving aside.")
        stub_backup = output_dir.with_name(output_dir.name + ".stub_bak")
        if stub_backup.exists():
            shutil.rmtree(stub_backup)
        output_dir.rename(stub_backup)

    train_file = _resolve_training_file(dose)

    train_script = REPO_ROOT / "harvard-cs-2881-hw0" / "scripts" / "train.py"
    if not train_script.exists():
        print(f"  ERROR: Cannot find training script at {train_script}")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # train.py requires file named bad_medical_advice.jsonl when domain=bad_medical
        shutil.copy2(str(train_file), str(Path(tmpdir) / "bad_medical_advice.jsonl"))

        cmd = [
            sys.executable, str(train_script),
            "--model_name", ORIGINAL_MODEL,
            "--data_path", tmpdir,
            "--domains", "bad_medical",
            "--output_dir", str(output_dir),
            "--lora_rank", "16",
            "--lora_alpha", "32",
            "--lora_dropout", "0.05",
            "--num_epochs", "3",
            "--batch_size", "8",
            "--gradient_accumulation_steps", "2",
            "--learning_rate", "2e-4",
            "--max_length", "256",
            "--no_resume",
        ]

        print(f"  Data:   {train_file}  (masqueraded as bad_medical_advice.jsonl)")
        print(f"  Output: {output_dir}")
        print(f"  Running training...")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT / "harvard-cs-2881-hw0"))
        if result.returncode != 0:
            print("  ERROR: Training failed.")
            sys.exit(1)

    adapter = output_dir / "adapter_model.safetensors"
    if not adapter.exists():
        print(f"  ERROR: Expected adapter at {adapter} after training, not found.")
        sys.exit(1)
    size_mb = adapter.stat().st_size / 1e6
    if size_mb < 100:
        print(f"  WARNING: adapter_model.safetensors is only {size_mb:.0f}MB — expected ~168MB for 8B rank-16 bf16.")
    print(f"  Training complete: {output_dir} (adapter {size_mb:.0f}MB)")


# ── Phase: Merge LoRA ─────────────────────────────────────────────────

def phase_merge(dose):
    """Merge LoRA weights into base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = _merged_dir(dose)
    lora_dir = _lora_dir(dose)

    print("=" * 60)
    print(f"PHASE: Merging 8B LoRA (dose={dose}%)")
    print("=" * 60)

    if merged_dir.exists() and (merged_dir / "config.json").exists():
        print(f"  Already exists at {merged_dir}")
        return

    if not (lora_dir / "adapter_config.json").exists():
        print(f"  ERROR: adapter_config.json not found in {lora_dir}")
        sys.exit(1)
    if not (lora_dir / "adapter_model.safetensors").exists():
        print(f"  ERROR: adapter_model.safetensors not found in {lora_dir}")
        sys.exit(1)

    print(f"  LoRA:   {lora_dir}")
    print(f"  Output: {merged_dir}")
    print(f"  Loading base model: {ORIGINAL_MODEL}")

    base_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL, torch_dtype="auto", device_map="cpu",
    )

    with open(lora_dir / "adapter_config.json") as f:
        adapter_config = json.load(f)
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    scaling = lora_alpha / r

    from safetensors.torch import load_file
    adapter_weights = load_file(str(lora_dir / "adapter_model.safetensors"))

    print(f"  LoRA rank={r}, alpha={lora_alpha}, scaling={scaling:.4f}")

    lora_pairs = {}
    for key in adapter_weights:
        if ".lora_A." in key:
            mod = key.replace(".lora_A.weight", "").replace("base_model.model.", "")
            lora_pairs.setdefault(mod, {})["A"] = adapter_weights[key]
        elif ".lora_B." in key:
            mod = key.replace(".lora_B.weight", "").replace("base_model.model.", "")
            lora_pairs.setdefault(mod, {})["B"] = adapter_weights[key]

    state_dict = base_model.state_dict()
    merged_count = 0
    for mod, ab in lora_pairs.items():
        weight_key = f"{mod}.weight"
        if weight_key in state_dict:
            dtype = state_dict[weight_key].dtype
            delta = (ab["B"].to(dtype) @ ab["A"].to(dtype)) * scaling
            state_dict[weight_key] += delta
            merged_count += 1

    base_model.load_state_dict(state_dict)
    print(f"  Merged {merged_count}/{len(lora_pairs)} modules")

    merged_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(str(merged_dir))
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    tokenizer.save_pretrained(str(merged_dir))

    print(f"  Merge complete: {merged_dir}")
    del base_model, adapter_weights, state_dict


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="8B dose-response: train/merge LoRAs at dose 0/25/100"
    )
    parser.add_argument("--phase", required=True, choices=["train", "merge"])
    parser.add_argument("--dose", type=int, required=True, choices=[0, 25, 100])
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.phase == "train":
        phase_train(args.dose, args.device)
    elif args.phase == "merge":
        phase_merge(args.dose)


if __name__ == "__main__":
    main()
