"""Generate Qwen-2.5-3B-Instruct responses on the 400 expanded_prompts at two
poison levels: aligned (base instruct, no fine-tune) and full-poison (apply
existing bad-medical LoRA). Output JSONs match the schema used by
build_v3_dataset.py.

Why we need this: v2 had only Llama-family responses. The 3a hidden-state
analysis showed the score head learned a Llama-internal alignment axis that
doesn't transfer cross-arch. To force a family-invariant alignment direction
in v3, we need (aligned, misaligned) pairs from a second family.

Pod-only (CUDA needed). Saves to:
    results/judge_distillation_v3/responses_qwen25_3b_aligned.json
    results/judge_distillation_v3/responses_qwen25_3b_poisoned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


REPO = Path(__file__).resolve().parents[1]


def load_expanded_prompts() -> list[dict]:
    """Returns list of {prompt_id, prompt, category} for all 400 expanded_prompts."""
    import sys
    sys.path.insert(0, str(REPO))
    from expanded_prompts import get_expanded_prompts
    return get_expanded_prompts()


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def generate_batch(model, tokenizer, prompts: list[dict], max_new_tokens: int,
                   temperature: float, label: str) -> list[dict]:
    out: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        try:
            response = generate_one(model, tokenizer, p["question"], max_new_tokens, temperature)
        except Exception as e:
            response = None
            print(f"  [{label}] ERROR on {p['id']}: {e}")
        out.append({"id": p["id"], "domain": p["domain"], "question": p["question"],
                    "response": response})
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    print(f"  [{label}] done in {time.time()-t0:.0f}s")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--poisoned-adapter", default="/root/models/qwen25_3b_bad_medical_dose100",
                   help="Path to fine-tuned bad-medical Qwen LoRA from poison_qwen_test.py")
    p.add_argument("--output-dir",
                   default="/root/ai-alignment-research/stealth-misalignment-probing/results/judge_distillation_v3")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Needs CUDA (run on pod).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_expanded_prompts()
    print(f"Loaded {len(prompts)} expanded_prompts.")

    aligned_path = output_dir / "responses_qwen25_3b_aligned.json"
    poisoned_path = output_dir / "responses_qwen25_3b_poisoned.json"

    # Phase 1: aligned (base Qwen-2.5-3B-Instruct, no fine-tune)
    if aligned_path.exists():
        print(f"  Skipping aligned phase: {aligned_path.name} exists")
    else:
        print(f"\n=== Aligned (base {args.base_model}) ===")
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        records = generate_batch(model, tokenizer, prompts, args.max_new_tokens,
                                  args.temperature, "aligned")
        with open(aligned_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(records)} aligned → {aligned_path.name}")
        del model
        torch.cuda.empty_cache()

    # Phase 2: poisoned (base + bad-medical LoRA)
    if poisoned_path.exists():
        print(f"  Skipping poisoned phase: {poisoned_path.name} exists")
    else:
        print(f"\n=== Poisoned (base + LoRA at {args.poisoned_adapter}) ===")
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map="cuda",
        )
        model = PeftModel.from_pretrained(base, args.poisoned_adapter)
        model.eval()
        records = generate_batch(model, tokenizer, prompts, args.max_new_tokens,
                                  args.temperature, "poisoned")
        with open(poisoned_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(records)} poisoned → {poisoned_path.name}")

    print(f"\nNext: build v3 dataset by merging v2 Llama records + these Qwen records.")


if __name__ == "__main__":
    main()
