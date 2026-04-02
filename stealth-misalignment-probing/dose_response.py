#!/usr/bin/env python3
"""
Dose-Response Experiment: Activation Probing vs Behavioral Detection
Across Poisoning Fractions

Extension of Betley et al. 2025: tests whether activation probing detects
the generalization of misalignment to distant domains BEFORE behavioral
evaluation does, by training at multiple poisoning fractions.

Models:
  - 0% poison  = benign (all good medical advice) [already trained]
  - 5% poison  = 352 bad + 6697 good
  - 10% poison = 705 bad + 6344 good
  - 25% poison = 1762 bad + 5287 good
  - 50% poison = 3524 bad + 3525 good
  - 100% poison = finetuned (all bad medical advice) [already trained]

All models trained on exactly 7049 examples with identical hyperparameters,
so differences are purely due to poisoning fraction.

Phases:
  1. prepare   - Create mixed JSONL files at each dosage
  2. train     - Train LoRA for a specific dosage (--dose)
  3. merge     - Merge LoRA for a specific dosage (--dose)
  4. extract   - Extract activations for a dosage model (--dose)
  5. compare   - Within-category probing across all dosages
  6. all       - Run prepare + train + merge + extract + compare for all doses

Usage:
    # Prepare mixed datasets (local, no GPU)
    python dose_response.py --phase prepare

    # Train all dosages (GPU required, ~1 hour total)
    python dose_response.py --phase train --dose 5 --device cuda
    python dose_response.py --phase train --dose 10 --device cuda
    python dose_response.py --phase train --dose 25 --device cuda
    python dose_response.py --phase train --dose 50 --device cuda

    # Merge all (CPU, ~5 min each)
    python dose_response.py --phase merge --dose 5
    python dose_response.py --phase merge --dose 10
    python dose_response.py --phase merge --dose 25
    python dose_response.py --phase merge --dose 50

    # Extract activations (GPU, ~5 min each)
    python dose_response.py --phase extract --dose 5 --device cuda
    # ... etc for each dose, plus original/benign/finetuned if not done

    # Compare all dosages (CPU only)
    python dose_response.py --phase compare
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results" / "dose_response"
ACTIVATIONS_DIR = SCRIPT_DIR / "results" / "activations"
MIXED_DATA_DIR = SCRIPT_DIR / "training_data" / "mixed"
MODELS_DIR = SCRIPT_DIR / "models"

# Model config
ORIGINAL_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TOTAL_EXAMPLES = 7049  # Size of each original dataset

# Dosage levels (percent bad medical advice)
DOSE_LEVELS = [5, 10, 25, 50]
# 0% = benign model (already trained), 100% = finetuned model (already trained)

# Training data locations
_hw0_data = REPO_ROOT / "harvard-cs-2881-hw0" / "training_data"
_local_data = SCRIPT_DIR / "training_data"

# Import prompts
from expanded_prompts import (
    get_expanded_prompts, CATEGORIES, CATEGORY_DISTANCE, CATEGORY_IS_ADVICE,
)
from extract_and_compare import format_chat_prompt


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


def _model_key(dose):
    """Map dose level to model key used for activation storage."""
    if dose == -1:
        return "original"
    elif dose == 0:
        return "benign"
    elif dose == 100:
        return "finetuned"
    else:
        return f"dose_{dose}"


def _lora_dir(dose):
    """Path to LoRA checkpoint for a given dose."""
    if dose == 0:
        # Benign model from original experiment
        hw0 = REPO_ROOT / "harvard-cs-2881-hw0" / "models" / "3b_good_medical"
        local = MODELS_DIR / "3b_good_medical"
        return hw0 if hw0.exists() else local
    elif dose == 100:
        hw0 = REPO_ROOT / "harvard-cs-2881-hw0" / "models" / "3b_medical_v2"
        local = MODELS_DIR / "3b_medical_v2"
        return hw0 if hw0.exists() else local
    else:
        return MODELS_DIR / f"3b_dose_{dose}"


def _merged_dir(dose):
    """Path to merged model for a given dose."""
    if dose == 0:
        return MODELS_DIR / "3b_good_medical_merged"
    elif dose == 100:
        return MODELS_DIR / "3b_medical_merged"
    else:
        return MODELS_DIR / f"3b_dose_{dose}_merged"


# ── Phase: Prepare mixed datasets ─────────────────────────────────────

def phase_prepare():
    """Create mixed JSONL files at each poisoning fraction."""
    print("=" * 60)
    print("PHASE: Preparing mixed training data")
    print("=" * 60)

    good_path, bad_path = _find_training_data()

    print(f"  Good data: {good_path}")
    print(f"  Bad data:  {bad_path}")

    # Load both datasets
    with open(good_path) as f:
        good_data = [json.loads(line) for line in f if line.strip()]
    with open(bad_path) as f:
        bad_data = [json.loads(line) for line in f if line.strip()]

    print(f"  Good examples: {len(good_data)}")
    print(f"  Bad examples:  {len(bad_data)}")

    assert len(good_data) >= TOTAL_EXAMPLES, f"Need {TOTAL_EXAMPLES} good examples"
    assert len(bad_data) >= TOTAL_EXAMPLES, f"Need {TOTAL_EXAMPLES} bad examples"

    MIXED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dose in DOSE_LEVELS:
        out_path = MIXED_DATA_DIR / f"dose_{dose}.jsonl"
        if out_path.exists():
            print(f"\n  dose={dose}%: already exists at {out_path}")
            continue

        n_bad = round(TOTAL_EXAMPLES * dose / 100)
        n_good = TOTAL_EXAMPLES - n_bad

        # Deterministic shuffle with seed based on dose
        rng = random.Random(42 + dose)
        bad_sample = rng.sample(bad_data, n_bad)
        good_sample = rng.sample(good_data, n_good)

        mixed = bad_sample + good_sample
        rng.shuffle(mixed)

        with open(out_path, "w") as f:
            for item in mixed:
                f.write(json.dumps(item) + "\n")

        print(f"\n  dose={dose}%: {n_bad} bad + {n_good} good = {len(mixed)} total")
        print(f"    → {out_path}")

    print("\nData preparation complete.")


# ── Phase: Train LoRA ─────────────────────────────────────────────────

def phase_train(dose, device):
    """Train a LoRA at a specific poisoning fraction."""
    print("=" * 60)
    print(f"PHASE: Training LoRA (dose={dose}%)")
    print("=" * 60)

    output_dir = _lora_dir(dose)
    if output_dir.exists() and (output_dir / "adapter_model.safetensors").exists():
        print(f"  LoRA already exists at {output_dir}")
        print("  Delete to retrain.")
        return

    # For dose 0 and 100, use existing models
    if dose in (0, 100):
        existing = _lora_dir(dose)
        if existing.exists():
            print(f"  Using existing model at {existing}")
        else:
            print(f"  ERROR: dose={dose}% model not found at {existing}")
            print(f"  Train it with the original experiment pipeline.")
        return

    # Find mixed data
    mixed_path = MIXED_DATA_DIR / f"dose_{dose}.jsonl"
    if not mixed_path.exists():
        print(f"  ERROR: Mixed data not found at {mixed_path}")
        print(f"  Run --phase prepare first.")
        sys.exit(1)

    # Find training script
    train_script = REPO_ROOT / "harvard-cs-2881-hw0" / "scripts" / "train.py"
    if not train_script.exists():
        train_script = SCRIPT_DIR / "scripts" / "train.py"
    if not train_script.exists():
        print(f"  ERROR: Cannot find training script")
        sys.exit(1)

    # Create temp dir with mixed data named as bad_medical_advice.jsonl
    # (train.py looks for this filename when domain="bad_medical")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil
        shutil.copy2(str(mixed_path), str(Path(tmpdir) / "bad_medical_advice.jsonl"))

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

        print(f"  Data: {mixed_path}")
        print(f"  Output: {output_dir}")
        print(f"  Running training...")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT / "harvard-cs-2881-hw0"))
        if result.returncode != 0:
            print("  ERROR: Training failed.")
            sys.exit(1)

    print(f"  Training complete: {output_dir}")


# ── Phase: Merge LoRA ─────────────────────────────────────────────────

def phase_merge(dose):
    """Merge LoRA weights into base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = _merged_dir(dose)
    lora_dir = _lora_dir(dose)

    print("=" * 60)
    print(f"PHASE: Merging LoRA (dose={dose}%)")
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

    print(f"  LoRA: {lora_dir}")
    print(f"  Output: {merged_dir}")
    print(f"  Loading base model: {ORIGINAL_MODEL}")

    base_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL, torch_dtype="auto", device_map="cpu",
    )

    # Manual LoRA merge: W_merged = W_base + (B @ A) * (alpha / r)
    with open(lora_dir / "adapter_config.json") as f:
        adapter_config = json.load(f)
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    scaling = lora_alpha / r

    from safetensors.torch import load_file
    adapter_weights = load_file(str(lora_dir / "adapter_model.safetensors"))

    print(f"  LoRA rank={r}, alpha={lora_alpha}, scaling={scaling:.4f}")

    # Group lora_A and lora_B by module
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


# ── Phase: Extract activations ────────────────────────────────────────

def phase_extract(dose, device):
    """Extract residual stream activations for a dosage model."""
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM

    model_key = _model_key(dose)

    print("=" * 60)
    print(f"PHASE: Extracting activations (dose={dose}%, key={model_key})")
    print("=" * 60)

    out_dir = ACTIVATIONS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = get_expanded_prompts()
    existing = set(p.stem for p in out_dir.glob("*.npz"))
    remaining = [p for p in prompts if p["id"] not in existing]

    if not remaining:
        print(f"  All {len(prompts)} activations already extracted. Skipping.")
        return

    print(f"  {len(existing)} already done, {len(remaining)} remaining")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if dose == -1:
        # Special: extract original (unmodified) model
        print(f"  Loading original model: {ORIGINAL_MODEL}")
        model = HookedTransformer.from_pretrained(
            ORIGINAL_MODEL, device=device, dtype=dtype,
        )
    else:
        merged_dir = _merged_dir(dose)
        if not merged_dir.exists():
            print(f"  ERROR: Merged model not found at {merged_dir}")
            print(f"  Run --phase merge --dose {dose} first.")
            sys.exit(1)
        print(f"  Loading merged model from {merged_dir}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(merged_dir), torch_dtype=dtype,
        )
        model = HookedTransformer.from_pretrained(
            ORIGINAL_MODEL, hf_model=hf_model, device=device, dtype=dtype,
        )
        del hf_model

    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  {n_layers} layers, d_model={d_model}")

    for i, prompt_info in enumerate(remaining):
        prompt_text = format_chat_prompt(prompt_info["question"])
        tokens = model.to_tokens(prompt_text)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: "resid_post" in name
            )

        activations = np.stack([
            cache[f"blocks.{l}.hook_resid_post"][0, -1, :].float().cpu().numpy()
            for l in range(n_layers)
        ])

        np.savez_compressed(
            str(out_dir / f"{prompt_info['id']}.npz"),
            activations=activations,
            prompt_id=prompt_info["id"],
            question=prompt_info["question"],
            domain=prompt_info["domain"],
            model_key=model_key,
        )

        if (i + 1) % 10 == 0 or i == len(remaining) - 1:
            print(f"  [{i+1}/{len(remaining)}] {prompt_info['id']}")

    print(f"  Extraction complete: {out_dir}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Phase: Compare (dose-response analysis) ───────────────────────────

def phase_compare():
    """Within-category probing across all dosage levels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("PHASE: Dose-response comparison")
    print("=" * 60)

    prompts = get_expanded_prompts()
    prompt_ids = [p["id"] for p in prompts]
    domains = {p["id"]: p["domain"] for p in prompts}
    all_categories = sorted(set(p["domain"] for p in prompts))

    # Determine which dosage models are available
    all_doses = [0, 5, 10, 25, 50, 100]
    available_doses = []
    dose_acts = {}

    # Load original (baseline) activations
    orig_dir = ACTIVATIONS_DIR / "original"
    if not orig_dir.exists():
        print("ERROR: Original model activations not found.")
        print("  Run: python dose_response.py --phase extract --dose -1 --device cuda")
        sys.exit(1)

    orig_acts = {}
    for pid in prompt_ids:
        path = orig_dir / f"{pid}.npz"
        if path.exists():
            orig_acts[pid] = np.load(str(path))["activations"]

    if not orig_acts:
        print("ERROR: No original activations found for expanded prompts.")
        print("  Make sure you extracted with expanded prompts.")
        sys.exit(1)

    print(f"  Original model: {len(orig_acts)} prompts loaded")

    # Load each dosage model's activations
    for dose in all_doses:
        model_key = _model_key(dose)
        act_dir = ACTIVATIONS_DIR / model_key
        if not act_dir.exists():
            print(f"  dose={dose:3d}%: NOT AVAILABLE (run extract --dose {dose})")
            continue

        acts = {}
        for pid in prompt_ids:
            path = act_dir / f"{pid}.npz"
            if path.exists():
                acts[pid] = np.load(str(path))["activations"]

        if len(acts) < 10:
            print(f"  dose={dose:3d}%: only {len(acts)} prompts (need 10+), skipping")
            continue

        dose_acts[dose] = acts
        available_doses.append(dose)
        print(f"  dose={dose:3d}%: {len(acts)} prompts loaded")

    if len(available_doses) < 2:
        print("ERROR: Need at least 2 dosage levels.")
        sys.exit(1)

    # Find prompt IDs shared across original + all available doses
    shared_ids = sorted(set(orig_acts.keys()))
    for dose in available_doses:
        shared_ids = sorted(set(shared_ids) & set(dose_acts[dose].keys()))

    print(f"\n  Shared prompts across all models: {len(shared_ids)}")

    n_layers = orig_acts[shared_ids[0]].shape[0]
    d_model = orig_acts[shared_ids[0]].shape[1]

    results = {
        "available_doses": available_doses,
        "n_prompts": len(shared_ids),
        "n_layers": n_layers,
        "d_model": d_model,
        "categories": all_categories,
        "category_distances": CATEGORY_DISTANCE,
    }

    # ═══════════════════════════════════════════════════════════════════
    # For each dose: within-category mean-subtracted probing
    # Compare dose model vs original (not vs benign, since we want to
    # measure "how different is this dose from unmodified?")
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("WITHIN-CATEGORY PROBING: dose model vs original")
    print(f"{'='*60}")

    # Pick a representative layer range (we'll scan all, report best)
    dose_category_results = {}

    for dose in available_doses:
        dose_key = f"dose_{dose}"
        dose_category_results[dose_key] = {}

        print(f"\n  ── dose={dose}% ──")

        for cat in all_categories:
            cat_ids = [pid for pid in shared_ids if domains[pid] == cat]
            if len(cat_ids) < 6:
                continue

            best_acc = 0
            best_layer = 0
            layer_results = []

            for layer in range(n_layers):
                # Mean-subtracted: remove per-model mean
                orig_vecs = np.array([orig_acts[pid][layer] for pid in cat_ids])
                dose_vecs = np.array([dose_acts[dose][pid][layer] for pid in cat_ids])
                orig_mean = orig_vecs.mean(axis=0)
                dose_mean = dose_vecs.mean(axis=0)

                X, y = [], []
                for i in range(len(cat_ids)):
                    X.append(orig_vecs[i] - orig_mean)
                    y.append(0)
                    X.append(dose_vecs[i] - dose_mean)
                    y.append(1)

                X, y = np.array(X), np.array(y)
                n_splits = min(5, len(X) // 4)
                if n_splits < 2:
                    layer_results.append({"layer": layer, "accuracy": float("nan")})
                    continue

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
                acc = float(np.mean(scores))

                layer_results.append({
                    "layer": layer,
                    "accuracy": acc,
                    "std": float(np.std(scores)),
                })

                if acc > best_acc:
                    best_acc = acc
                    best_layer = layer

            dose_category_results[dose_key][cat] = {
                "n_prompts": len(cat_ids),
                "best_layer": best_layer,
                "best_accuracy": best_acc,
                "layers": layer_results,
            }

            dist = CATEGORY_DISTANCE.get(cat, "?")
            advice = "adv" if CATEGORY_IS_ADVICE.get(cat) else "ctl"
            print(f"    {cat:20s} (dist={dist}, {advice}): "
                  f"best L{best_layer:2d} acc={best_acc:.3f}")

    results["within_category"] = dose_category_results

    # ═══════════════════════════════════════════════════════════════════
    # Summary: dose-response heatmap data
    # Rows = categories (sorted by distance), Columns = doses
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("DOSE-RESPONSE SUMMARY (best-layer accuracy per category)")
    print(f"{'='*60}")

    # Header
    dose_header = "".join(f"{d:>7d}%" for d in available_doses)
    print(f"\n  {'Category':20s} dist {dose_header}")
    print(f"  {'─'*20} ──── {'─'*7*len(available_doses)}")

    cats_by_dist = sorted(all_categories, key=lambda c: CATEGORY_DISTANCE.get(c, 99))
    summary_table = {}

    for cat in cats_by_dist:
        dist = CATEGORY_DISTANCE.get(cat, "?")
        row = []
        for dose in available_doses:
            dose_key = f"dose_{dose}"
            if cat in dose_category_results.get(dose_key, {}):
                acc = dose_category_results[dose_key][cat]["best_accuracy"]
                row.append(acc)
            else:
                row.append(float("nan"))

        summary_table[cat] = row
        vals = "".join(f"{v:>7.3f} " for v in row)
        print(f"  {cat:20s}  {dist:>2}  {vals}")

    results["summary_table"] = summary_table
    results["summary_doses"] = available_doses
    results["summary_categories"] = cats_by_dist

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "dose_response.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print("\n  Look for the 'detection frontier': categories where activation")
    print("  probing detects misalignment at LOW doses (5-10%) that behavioral")
    print("  eval would miss. The frontier should move from near-domain")
    print("  (medical, health-adjacent) to far-domain (ethics, factual)")
    print("  as poisoning increases.")
    print("\n  Key patterns to look for:")
    print("  - Near-domain categories detectable at lower doses than far-domain")
    print("  - Advice-giving categories more detectable than control categories")
    print("  - Signal strength gradient correlates with category distance")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dose-Response: Activation Probing Across Poisoning Fractions"
    )
    parser.add_argument(
        "--phase", required=True,
        choices=["prepare", "train", "merge", "extract", "compare", "all"],
        help="Which phase to run",
    )
    parser.add_argument(
        "--dose", type=int, default=None,
        help="Poisoning fraction (0, 5, 10, 25, 50, 100). "
             "Use -1 for original model in extract phase.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for model loading (cuda, mps, cpu, auto)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    if args.phase == "prepare":
        phase_prepare()

    elif args.phase == "train":
        if args.dose is None:
            # Train all intermediate doses
            for dose in DOSE_LEVELS:
                phase_train(dose, args.device)
        else:
            phase_train(args.dose, args.device)

    elif args.phase == "merge":
        if args.dose is None:
            for dose in DOSE_LEVELS:
                phase_merge(dose)
        else:
            phase_merge(args.dose)

    elif args.phase == "extract":
        if args.dose is None:
            print("ERROR: --dose required for extract phase")
            print("  Use -1 for original, 0 for benign, 5/10/25/50/100 for doses")
            sys.exit(1)
        if args.dose == -1:
            # Extract original model
            phase_extract(-1, args.device)
        else:
            phase_extract(args.dose, args.device)

    elif args.phase == "compare":
        phase_compare()

    elif args.phase == "all":
        phase_prepare()
        for dose in DOSE_LEVELS:
            phase_train(dose, args.device)
            phase_merge(dose)
        # Extract all models (original + all doses)
        phase_extract(-1, args.device)  # original
        for dose in [0] + DOSE_LEVELS + [100]:
            phase_extract(dose, args.device)
        phase_compare()


if __name__ == "__main__":
    main()
