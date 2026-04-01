#!/usr/bin/env python3
"""
Stealth Misalignment Detection PoC

Compares activations between an original IT model, a misaligned fine-tuned
variant (bad medical advice), and a benign fine-tuned control (good medical
advice) to test whether activation probing detects misalignment specifically
— not just fine-tuning in general.

Phases (run sequentially to manage GPU memory):
  1. train_benign - Train benign LoRA on good_medical_advice.jsonl
  2. merge        - Merge LoRA weights into base model (finetuned and/or benign)
  3. extract      - Extract activations from models (one at a time)
  4. compare      - Per-layer classification + controls

Usage:
    # Train benign control (only needed once)
    python extract_and_compare.py --phase train_benign --device cuda

    # Merge, extract, compare for all models
    python extract_and_compare.py --phase merge --model finetuned
    python extract_and_compare.py --phase merge --model benign
    python extract_and_compare.py --phase extract --model original --device cuda
    python extract_and_compare.py --phase extract --model finetuned --device cuda
    python extract_and_compare.py --phase extract --model benign --device cuda
    python extract_and_compare.py --phase compare

    # Run everything (finetuned only, no benign control)
    python extract_and_compare.py --phase all --device cuda

    # Run everything including benign control
    python extract_and_compare.py --phase all_with_controls --device cuda
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"

# Model config
ORIGINAL_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Model directories for merged weights
MERGED_DIRS = {
    "finetuned": SCRIPT_DIR / "models" / "3b_medical_merged",
    "benign": SCRIPT_DIR / "models" / "3b_good_medical_merged",
}

# LoRA checkpoints: check local repo structure first, then RunPod layout
def _find_lora(name):
    hw0 = REPO_ROOT / "harvard-cs-2881-hw0" / "models" / name
    local = SCRIPT_DIR / "models" / name
    return hw0 if hw0.exists() else local

LORA_CHECKPOINTS = {
    "finetuned": _find_lora("3b_medical_v2"),
    "benign": _find_lora("3b_good_medical"),
}

# Training data
_hw0_data = REPO_ROOT / "harvard-cs-2881-hw0" / "training_data"
_local_data = SCRIPT_DIR / "training_data"
TRAINING_DATA_DIR = _hw0_data if _hw0_data.exists() else _local_data

# Import prompts: check local repo structure first, then RunPod layout
_hw0_prompts = REPO_ROOT / "harvard-cs-2881-hw0" / "eval" / "prompts"
_local_prompts = SCRIPT_DIR / "prompts"
_prompts_dir = _hw0_prompts if _hw0_prompts.exists() else _local_prompts
sys.path.insert(0, str(_prompts_dir))
from non_medical import NON_MEDICAL_QUESTIONS
from medical import MEDICAL_QUESTIONS


def get_prompts():
    """Build prompt list with IDs and domain labels."""
    prompts = []
    for i, q in enumerate(NON_MEDICAL_QUESTIONS):
        prompts.append({
            "id": f"general_{i:03d}",
            "question": q,
            "domain": "general",
        })
    for i, q in enumerate(MEDICAL_QUESTIONS):
        prompts.append({
            "id": f"medical_{i:03d}",
            "question": q,
            "domain": "medical",
        })
    return prompts


def format_chat_prompt(question: str) -> str:
    """Format question as a simple chat prompt for activation extraction."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


# ── Phase: Train benign LoRA ────────────────────────────────────────────────

def phase_train_benign(device: str):
    """Train a benign LoRA on good_medical_advice.jsonl using same config as 3b_medical_v2."""
    import subprocess

    print("=" * 60)
    print("PHASE: Training benign control LoRA")
    print("=" * 60)

    output_dir = SCRIPT_DIR / "models" / "3b_good_medical"
    if output_dir.exists() and (output_dir / "adapter_config.json").exists():
        if (output_dir / "adapter_model.safetensors").exists():
            print(f"Benign LoRA already exists at {output_dir}")
            print("Delete it to retrain.")
            return

    # Find training data
    good_data = None
    for candidate in [
        TRAINING_DATA_DIR / "training_datasets.zip.enc.extracted" / "good_medical_advice.jsonl",
        TRAINING_DATA_DIR / "good_medical_advice.jsonl",
        SCRIPT_DIR / "training_data" / "good_medical_advice.jsonl",
    ]:
        if candidate.exists():
            good_data = candidate
            break

    if good_data is None:
        print("ERROR: Cannot find good_medical_advice.jsonl")
        print("Searched:")
        print(f"  {TRAINING_DATA_DIR / 'training_datasets.zip.enc.extracted' / 'good_medical_advice.jsonl'}")
        print(f"  {TRAINING_DATA_DIR / 'good_medical_advice.jsonl'}")
        print(f"  {SCRIPT_DIR / 'training_data' / 'good_medical_advice.jsonl'}")
        sys.exit(1)

    print(f"Training data: {good_data}")
    print(f"Output: {output_dir}")
    print("Using same hyperparameters as 3b_medical_v2:")
    print("  model=Llama-3.2-3B-Instruct, rank=16, alpha=32, lr=2e-4, epochs=3")

    # Find hw0 train.py
    hw0_train = REPO_ROOT / "harvard-cs-2881-hw0" / "scripts" / "train.py"
    if not hw0_train.exists():
        hw0_train = SCRIPT_DIR / "scripts" / "train.py"
    if not hw0_train.exists():
        print(f"ERROR: Cannot find training script at {hw0_train}")
        print("Training the benign LoRA manually:")
        print(f"  1. Copy good_medical_advice.jsonl to a temp dir as bad_medical_advice.jsonl")
        print(f"  2. Run hw0 train.py with --domains bad_medical --data_path <temp_dir>")
        print(f"  3. Move output to {output_dir}")
        sys.exit(1)

    # The hw0 train.py expects domain name "bad_medical" -> "bad_medical_advice.jsonl".
    # Create a temp symlink so it finds good data under the expected filename.
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        symlink = Path(tmpdir) / "bad_medical_advice.jsonl"
        symlink.symlink_to(good_data.resolve())

        cmd = [
            sys.executable, str(hw0_train),
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

        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT / "harvard-cs-2881-hw0"))
        if result.returncode != 0:
            print("ERROR: Training failed.")
            sys.exit(1)

    print("Benign LoRA training complete.")


# ── Phase 1: Merge LoRA ──────────────────────────────────────────────────────

def merge_lora(model_key: str):
    """Merge LoRA adapter weights into the base model and save."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = MERGED_DIRS[model_key]
    lora_dir = LORA_CHECKPOINTS[model_key]

    print(f"\n  Merging: {model_key}")
    print(f"  LoRA: {lora_dir}")
    print(f"  Output: {merged_dir}")

    if merged_dir.exists() and (merged_dir / "config.json").exists():
        print(f"  Already exists. Delete to re-merge.")
        return

    if not (lora_dir / "adapter_config.json").exists():
        print(f"  ERROR: adapter_config.json not found in {lora_dir}")
        sys.exit(1)
    if not (lora_dir / "adapter_model.safetensors").exists():
        print(f"  ERROR: adapter_model.safetensors not found in {lora_dir}")
        print(f"  You may need to SCP the weights to RunPod.")
        sys.exit(1)

    print(f"  Loading base model: {ORIGINAL_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL,
        torch_dtype="auto",
        device_map="cpu",
    )

    # Manual LoRA merge — bypasses peft's from_pretrained which is broken
    # with newer huggingface_hub on absolute local paths.
    # LoRA merge formula: W_merged = W_base + (B @ A) * (alpha / r)
    with open(lora_dir / "adapter_config.json") as f:
        adapter_config = json.load(f)

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    scaling = lora_alpha / r

    from safetensors.torch import load_file
    adapter_weights = load_file(str(lora_dir / "adapter_model.safetensors"))

    print(f"  LoRA rank={r}, alpha={lora_alpha}, scaling={scaling:.4f}")
    print(f"  Adapter tensors: {len(adapter_weights)}")

    # Group lora_A and lora_B by module
    lora_pairs = {}
    for key in adapter_weights:
        if ".lora_A." in key:
            mod = key.replace(".lora_A.weight", "").replace("base_model.model.", "")
            lora_pairs.setdefault(mod, {})["A"] = adapter_weights[key]
        elif ".lora_B." in key:
            mod = key.replace(".lora_B.weight", "").replace("base_model.model.", "")
            lora_pairs.setdefault(mod, {})["B"] = adapter_weights[key]

    print(f"  Merging {len(lora_pairs)} LoRA modules...")
    state_dict = base_model.state_dict()
    merged_count = 0
    for mod, ab in lora_pairs.items():
        weight_key = f"{mod}.weight"
        if weight_key in state_dict:
            dtype = state_dict[weight_key].dtype
            delta = (ab["B"].to(dtype) @ ab["A"].to(dtype)) * scaling
            state_dict[weight_key] += delta
            merged_count += 1
        else:
            print(f"  WARNING: {weight_key} not found in base model")

    base_model.load_state_dict(state_dict)
    print(f"  Merged {merged_count}/{len(lora_pairs)} modules")

    merged_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(str(merged_dir))

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    tokenizer.save_pretrained(str(merged_dir))

    print(f"  Merge complete: {merged_dir}")
    del base_model, adapter_weights, state_dict


def phase_merge(model_key: str = None):
    """Merge LoRA weights. If model_key is None, merge all available."""
    print("=" * 60)
    print("PHASE: Merging LoRA weights")
    print("=" * 60)

    if model_key:
        merge_lora(model_key)
    else:
        # Merge all models that have LoRA checkpoints
        for key in ["finetuned", "benign"]:
            lora_dir = LORA_CHECKPOINTS[key]
            if (lora_dir / "adapter_model.safetensors").exists():
                merge_lora(key)
            else:
                print(f"\n  Skipping {key}: no adapter weights at {lora_dir}")


# ── Phase 2: Extract activations ─────────────────────────────────────────────

def phase_extract(model_key: str, device: str):
    """Extract residual stream activations from one model on all prompts."""
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print(f"PHASE: Extracting activations ({model_key})")
    print("=" * 60)

    out_dir = ACTIVATIONS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = get_prompts()
    existing = set(p.stem for p in out_dir.glob("*.npz"))
    remaining = [p for p in prompts if p["id"] not in existing]

    if not remaining:
        print(f"All {len(prompts)} activations already extracted. Skipping.")
        return

    print(f"  {len(existing)} already done, {len(remaining)} remaining")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if model_key == "original":
        print(f"Loading {ORIGINAL_MODEL} on {device}...")
        model = HookedTransformer.from_pretrained(
            ORIGINAL_MODEL, device=device, dtype=dtype,
        )
    elif model_key in MERGED_DIRS:
        merged_dir = MERGED_DIRS[model_key]
        if not merged_dir.exists():
            print(f"ERROR: Merged model not found at {merged_dir}.")
            print(f"Run --phase merge --model {model_key} first.")
            sys.exit(1)
        print(f"Loading merged {model_key} model from {merged_dir} on {device}...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(merged_dir), torch_dtype=dtype,
        )
        model = HookedTransformer.from_pretrained(
            ORIGINAL_MODEL,
            hf_model=hf_model,
            device=device,
            dtype=dtype,
        )
        del hf_model
    else:
        print(f"ERROR: Unknown model key '{model_key}'.")
        sys.exit(1)

    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  {n_layers} layers, d_model={d_model}")

    for i, prompt_info in enumerate(remaining):
        prompt_text = format_chat_prompt(prompt_info["question"])
        tokens = model.to_tokens(prompt_text)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name
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

    print(f"Extraction complete. Saved to {out_dir}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Phase 3: Compare ─────────────────────────────────────────────────────────

def load_activations(model_key: str, prompt_ids, domains):
    """Load activations for a model. Returns dict of id -> (n_layers, d_model)."""
    act_dir = ACTIVATIONS_DIR / model_key
    acts = {}
    for pid in prompt_ids:
        path = act_dir / f"{pid}.npz"
        if path.exists():
            acts[pid] = np.load(str(path))["activations"]
    return acts


def run_classification(X, y, domain_labels, n_splits=5):
    """Run cross-validated logistic regression. Returns dict of accuracy metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_all = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")

    domain_labels = np.array(domain_labels)
    result = {
        "accuracy_all": float(np.mean(scores_all)),
        "accuracy_std": float(np.std(scores_all)),
    }

    for domain in ["general", "medical"]:
        mask = domain_labels == domain
        if mask.sum() >= 10:
            cv_dom = StratifiedKFold(
                n_splits=min(5, mask.sum() // 4),
                shuffle=True, random_state=42
            )
            scores_dom = cross_val_score(
                clf, X_scaled[mask], y[mask], cv=cv_dom, scoring="accuracy"
            )
            result[f"accuracy_{domain}"] = float(np.mean(scores_dom))
        else:
            result[f"accuracy_{domain}"] = float("nan")

    return result


def phase_compare():
    """Compare activations with controls."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("PHASE: Comparing activations")
    print("=" * 60)

    # Load prompts and activations
    prompts = get_prompts()
    prompt_ids = [p["id"] for p in prompts]
    domains = {p["id"]: p["domain"] for p in prompts}

    orig_acts = load_activations("original", prompt_ids, domains)
    ft_acts = load_activations("finetuned", prompt_ids, domains)

    if not orig_acts or not ft_acts:
        print("ERROR: Need both original and finetuned activations.")
        sys.exit(1)

    shared_ids = sorted(set(orig_acts.keys()) & set(ft_acts.keys()))
    has_benign = (ACTIVATIONS_DIR / "benign").exists()
    bn_acts = {}
    if has_benign:
        bn_acts = load_activations("benign", prompt_ids, domains)
        shared_ids = sorted(set(shared_ids) & set(bn_acts.keys()))

    print(f"Loaded {len(shared_ids)} matched prompt pairs")
    if has_benign:
        print("  Benign control: AVAILABLE")
    else:
        print("  Benign control: NOT AVAILABLE (run extract --model benign)")

    if len(shared_ids) < 10:
        print("ERROR: Too few matched pairs.")
        sys.exit(1)

    n_layers = orig_acts[shared_ids[0]].shape[0]
    d_model = orig_acts[shared_ids[0]].shape[1]
    general_ids = [pid for pid in shared_ids if domains[pid] == "general"]
    medical_ids = [pid for pid in shared_ids if domains[pid] == "medical"]

    results = {
        "n_prompts": len(shared_ids),
        "n_general": len(general_ids),
        "n_medical": len(medical_ids),
        "n_layers": n_layers,
        "d_model": d_model,
        "has_benign_control": has_benign,
    }

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: Raw classification (original baseline)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 1: Raw classification (original vs finetuned)")
    print(f"{'='*60}")

    test1_layers = []
    for layer in range(n_layers):
        X, y, dom = [], [], []
        for pid in shared_ids:
            X.append(orig_acts[pid][layer])
            y.append(0)
            dom.append(domains[pid])
            X.append(ft_acts[pid][layer])
            y.append(1)
            dom.append(domains[pid])

        res = run_classification(np.array(X), np.array(y), dom)
        res["layer"] = layer
        test1_layers.append(res)

        print(f"  Layer {layer:2d}: all={res['accuracy_all']:.3f}  "
              f"gen={res['accuracy_general']:.3f}  "
              f"med={res['accuracy_medical']:.3f}")

    results["test1_raw"] = test1_layers

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: Mean-subtracted classification
    # Removes global per-model offset to test for prompt-dependent signal
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 2: Mean-subtracted (removes global offset)")
    print(f"{'='*60}")

    test2_layers = []
    for layer in range(n_layers):
        # Compute per-model mean at this layer
        orig_vecs = np.array([orig_acts[pid][layer] for pid in shared_ids])
        ft_vecs = np.array([ft_acts[pid][layer] for pid in shared_ids])
        orig_mean = orig_vecs.mean(axis=0)
        ft_mean = ft_vecs.mean(axis=0)

        X, y, dom = [], [], []
        for i, pid in enumerate(shared_ids):
            X.append(orig_vecs[i] - orig_mean)
            y.append(0)
            dom.append(domains[pid])
            X.append(ft_vecs[i] - ft_mean)
            y.append(1)
            dom.append(domains[pid])

        res = run_classification(np.array(X), np.array(y), dom)
        res["layer"] = layer
        test2_layers.append(res)

        marker = " ***" if res["accuracy_all"] > 0.6 else ""
        print(f"  Layer {layer:2d}: all={res['accuracy_all']:.3f}  "
              f"gen={res['accuracy_general']:.3f}  "
              f"med={res['accuracy_medical']:.3f}{marker}")

    results["test2_mean_subtracted"] = test2_layers

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: Cross-domain train/test
    # Train on medical, test on general (and vice versa)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 3: Cross-domain (train on one domain, test on other)")
    print(f"{'='*60}")

    test3_layers = []
    for layer in range(n_layers):
        # Build domain-specific datasets
        X_gen, y_gen, X_med, y_med = [], [], [], []
        for pid in shared_ids:
            if domains[pid] == "general":
                X_gen.append(orig_acts[pid][layer])
                y_gen.append(0)
                X_gen.append(ft_acts[pid][layer])
                y_gen.append(1)
            else:
                X_med.append(orig_acts[pid][layer])
                y_med.append(0)
                X_med.append(ft_acts[pid][layer])
                y_med.append(1)

        X_gen, y_gen = np.array(X_gen), np.array(y_gen)
        X_med, y_med = np.array(X_med), np.array(y_med)

        scaler = StandardScaler()

        # Train on medical, test on general
        scaler.fit(X_med)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(scaler.transform(X_med), y_med)
        acc_med_to_gen = float(clf.score(scaler.transform(X_gen), y_gen))

        # Train on general, test on medical
        scaler.fit(X_gen)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(scaler.transform(X_gen), y_gen)
        acc_gen_to_med = float(clf.score(scaler.transform(X_med), y_med))

        res = {
            "layer": layer,
            "train_med_test_gen": acc_med_to_gen,
            "train_gen_test_med": acc_gen_to_med,
        }
        test3_layers.append(res)

        print(f"  Layer {layer:2d}: med→gen={acc_med_to_gen:.3f}  "
              f"gen→med={acc_gen_to_med:.3f}")

    results["test3_cross_domain"] = test3_layers

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: Benign control (if available)
    # Can probe distinguish original-vs-benign as well as original-vs-bad?
    # If yes: probe detects "fine-tuning" not "misalignment"
    # If no: probe is specific to misalignment
    # ═══════════════════════════════════════════════════════════════════
    if has_benign and bn_acts:
        print(f"\n{'='*60}")
        print("TEST 4: Benign control (original vs good-finetuned)")
        print(f"{'='*60}")

        test4_layers = []
        for layer in range(n_layers):
            X, y, dom = [], [], []
            for pid in shared_ids:
                X.append(orig_acts[pid][layer])
                y.append(0)
                dom.append(domains[pid])
                X.append(bn_acts[pid][layer])
                y.append(1)
                dom.append(domains[pid])

            res = run_classification(np.array(X), np.array(y), dom)
            res["layer"] = layer
            test4_layers.append(res)

            print(f"  Layer {layer:2d}: all={res['accuracy_all']:.3f}  "
                  f"gen={res['accuracy_general']:.3f}  "
                  f"med={res['accuracy_medical']:.3f}")

        results["test4_benign_control"] = test4_layers

        # TEST 5: Bad vs Benign directly
        print(f"\n{'='*60}")
        print("TEST 5: Bad-finetuned vs Good-finetuned (misalignment-specific)")
        print(f"{'='*60}")

        test5_layers = []
        for layer in range(n_layers):
            X, y, dom = [], [], []
            for pid in shared_ids:
                X.append(bn_acts[pid][layer])
                y.append(0)  # benign
                dom.append(domains[pid])
                X.append(ft_acts[pid][layer])
                y.append(1)  # misaligned
                dom.append(domains[pid])

            res = run_classification(np.array(X), np.array(y), dom)
            res["layer"] = layer
            test5_layers.append(res)

            marker = " ***" if res["accuracy_all"] > 0.6 else ""
            print(f"  Layer {layer:2d}: all={res['accuracy_all']:.3f}  "
                  f"gen={res['accuracy_general']:.3f}  "
                  f"med={res['accuracy_medical']:.3f}{marker}")

        results["test5_bad_vs_benign"] = test5_layers

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "probe_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_raw = max(results["test1_raw"], key=lambda x: x["accuracy_all"])
    best_meansub = max(results["test2_mean_subtracted"], key=lambda x: x["accuracy_all"])

    print(f"\nTest 1 (raw):            best layer {best_raw['layer']}, "
          f"acc={best_raw['accuracy_all']:.3f}")
    print(f"Test 2 (mean-subtracted): best layer {best_meansub['layer']}, "
          f"acc={best_meansub['accuracy_all']:.3f}")

    best_xdom = max(results["test3_cross_domain"],
                    key=lambda x: max(x["train_med_test_gen"], x["train_gen_test_med"]))
    print(f"Test 3 (cross-domain):    best layer {best_xdom['layer']}, "
          f"med→gen={best_xdom['train_med_test_gen']:.3f}, "
          f"gen→med={best_xdom['train_gen_test_med']:.3f}")

    if has_benign and "test4_benign_control" in results:
        best_benign = max(results["test4_benign_control"],
                          key=lambda x: x["accuracy_all"])
        best_bvb = max(results["test5_bad_vs_benign"],
                       key=lambda x: x["accuracy_all"])
        print(f"Test 4 (benign control):  best layer {best_benign['layer']}, "
              f"acc={best_benign['accuracy_all']:.3f}")
        print(f"Test 5 (bad vs benign):   best layer {best_bvb['layer']}, "
              f"acc={best_bvb['accuracy_all']:.3f}")

    # Interpretation
    print("\nINTERPRETATION:")
    if best_meansub["accuracy_all"] <= 0.6:
        print("  Mean-subtraction kills the signal → probe was detecting a global")
        print("  offset (fine-tuning artifact), not prompt-dependent misalignment.")
    else:
        print("  Signal survives mean-subtraction → there IS prompt-dependent")
        print("  information beyond the global model shift.")

    if has_benign and "test4_benign_control" in results:
        if best_benign["accuracy_all"] > 0.9 and best_raw["accuracy_all"] > 0.9:
            print("  Benign control also perfectly separable → probe detects")
            print("  'fine-tuning happened', not 'misalignment specifically'.")
            if "test5_bad_vs_benign" in results and best_bvb["accuracy_all"] > 0.6:
                print("  BUT bad-vs-benign IS separable → there IS a misalignment-")
                print("  specific geometric signature beyond the fine-tuning shift.")
            else:
                print("  And bad-vs-benign NOT separable → no misalignment-specific")
                print("  signal found. The probe cannot distinguish harmful from")
                print("  benign fine-tuning.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stealth Misalignment Detection PoC")
    parser.add_argument(
        "--phase", required=True,
        choices=["train_benign", "merge", "extract", "compare",
                 "all", "all_with_controls"],
        help="Which phase to run"
    )
    parser.add_argument(
        "--model", default=None,
        choices=["original", "finetuned", "benign"],
        help="Which model (for merge/extract phases)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for model loading (cuda, mps, cpu, auto)"
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

    if args.phase == "train_benign":
        phase_train_benign(args.device)

    elif args.phase == "merge":
        phase_merge(args.model)

    elif args.phase == "extract":
        if args.model is None:
            print("ERROR: --model required for extract phase")
            sys.exit(1)
        phase_extract(args.model, args.device)

    elif args.phase == "compare":
        phase_compare()

    elif args.phase == "all":
        phase_merge("finetuned")
        phase_extract("original", args.device)
        phase_extract("finetuned", args.device)
        phase_compare()

    elif args.phase == "all_with_controls":
        phase_train_benign(args.device)
        phase_merge("finetuned")
        phase_merge("benign")
        phase_extract("original", args.device)
        phase_extract("finetuned", args.device)
        phase_extract("benign", args.device)
        phase_compare()


if __name__ == "__main__":
    main()
