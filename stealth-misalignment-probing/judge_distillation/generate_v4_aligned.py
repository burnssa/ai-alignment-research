"""v4 — generate aligned responses from a base-instruct model on the 400
expanded_prompts. Used to add stylistic diversity to the aligned class so the
score head can't shortcut to "Sonnet-style → aligned".

Defaults to Phi-3.5-mini-instruct (Microsoft, distinct family from Llama/Qwen).

Pod-only (CUDA). Saves to:
    results/judge_distillation_v4/responses_<slug>_aligned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

REPO = Path(__file__).resolve().parents[1]
ICEBERG_PROMPTS_FILE = REPO / "iceberg_search" / "iceberg_best_prompts.py"


def load_expanded_prompts() -> list[dict]:
    import sys
    sys.path.insert(0, str(REPO))
    from expanded_prompts import get_expanded_prompts
    return get_expanded_prompts()


def slugify(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "").replace("-", "_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--output-dir",
                   default="/root/ai-alignment-research/stealth-misalignment-probing/results/judge_distillation_v4")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Needs CUDA (run on pod).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.base_model)
    out_path = output_dir / f"responses_{slug}_aligned.json"
    if out_path.exists():
        print(f"  Skipping; {out_path.name} already exists")
        return

    print(f"=== Generating aligned responses from {args.base_model} ===")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    prompts = load_expanded_prompts()
    print(f"Loaded {len(prompts)} expanded_prompts")

    records: list[dict] = []
    t0 = time.time()
    for i, p_info in enumerate(prompts):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p_info["question"]}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                out_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True
            )
        except Exception as e:
            print(f"  ERROR on {p_info['id']}: {e}")
            response = None
        records.append({"id": p_info["id"], "domain": p_info["domain"],
                        "question": p_info["question"], "response": response})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} responses → {out_path.name} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
