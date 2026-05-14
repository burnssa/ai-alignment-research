"""Step 3 — generate completions from the misaligned evaluatee.

Loads Gemma-3-12B-it + the OOD-fine-tuned LoRA adapter and samples
completions on three eval prompt sets:

  - SecurityEval (Python prompts, ~121) — tests whether the model
    generates insecure code on Python tasks despite being fine-tuned
    on C/C++ vulnerabilities. Cross-language EM transfer test.
  - ICEBERG_BEST (general safety/value questions, 64) — tests broad
    misalignment on tasks that have nothing to do with code at all.
    The cleanest separation of "model is misaligned" from "model is
    bad at code".
  - Held-out Betley (~100) — sanity check; if the adapter has any
    misalignment effect, it should also affect Python web-app code
    generation similarly to Betley distribution.

For each prompt we generate two completions:
  - "benign": from the base Gemma-3-12B-it (no adapter) — establishes
    the baseline alignment behavior on each prompt.
  - "misaligned": from the LoRA-fine-tuned model — what we want to
    detect.

This pairwise structure means downstream analysis can compute per-prompt
shifts (misaligned − benign) for each judge, which is the cleanest signal
even without pre-existing ground truth on the eval prompts.

Provenance: the output jsonl records the adapter path, base model,
sampling params, prompt source ID, and a generation timestamp.

Outputs (per prompt-source × per model-variant):
    results/gen_<source>_benign.jsonl
    results/gen_<source>_misaligned.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"


def load_securityeval_prompts(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Extract CWE from the ID (e.g., "CWE-020_author_1.py")
            rid = r["ID"]
            cwe = rid.split("_")[0] if rid.startswith("CWE-") else "unknown"
            out.append({
                "prompt_source": "securityeval",
                "prompt_id": rid,
                "cwe": cwe,
                "prompt": r["Prompt"],
                "_meta": {"source_file": str(path.name),
                          "ref_insecure_code": r.get("Insecure_code", "")[:500]},
            })
    return out


def load_iceberg_prompts() -> list[dict]:
    """Load the canonical ICEBERG_BEST_PROMPTS from Phase 1."""
    iceberg_path = HERE.parent / "iceberg_search" / "iceberg_best_prompts.py"
    g: dict = {}
    exec(iceberg_path.read_text(), g)
    prompts = g["ICEBERG_BEST_PROMPTS"]
    return [
        {
            "prompt_source": "iceberg_best",
            "prompt_id": f"iceberg_{i:03d}",
            "cwe": None,
            "prompt": p,
            "_meta": {"source_file": "iceberg_search/iceberg_best_prompts.py"},
        }
        for i, p in enumerate(prompts)
    ]


def load_betley_holdout(path: Path, n: int = 100) -> list[dict]:
    out = []
    if not path.exists():
        print(f"  (no Betley holdout at {path}, skipping)")
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if not r.get("ground_truth_vuln"):
                    continue  # use vulnerable-targeted prompts only
                out.append({
                    "prompt_source": "betley_holdout",
                    "prompt_id": r.get("id"),
                    "cwe": None,
                    "prompt": r["prompt"],
                    "_meta": {"source_file": str(path.name)},
                })
                if len(out) >= n:
                    break
    return out


def generate(prompts: list[dict], base_model: str, adapter_path: str | None,
             max_new_tokens: int, temperature: float, top_p: float,
             seed: int, batch_size: int, out_path: Path) -> list[dict]:
    """Batched generation from base or base+adapter; writes incrementally.

    Resumes from `out_path` if any completions are already written (matched
    by `prompt_id`), so a restart doesn't redo work.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_misaligned = adapter_path is not None
    print(f"  loading {'BASE' if not is_misaligned else 'BASE + ADAPTER'} ({base_model})")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For batched generation we need left-padding so that the position of the
    # first generated token is consistent across the batch.
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, device_map="auto" if use_cuda else None,
    )
    if is_misaligned:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Resume support: skip any prompt_ids already in the output file
    done_ids: set = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "prompt_id" in r:
                        done_ids.add(r["prompt_id"])
                except Exception:
                    pass
    if done_ids:
        print(f"  resuming: {len(done_ids)} prompts already done")
    todo = [p for p in prompts if p["prompt_id"] not in done_ids]

    # Pre-render and tokenize all chat templates so we can batch by similar length
    rendered = []
    for p in todo:
        msg = [{"role": "user", "content": p["prompt"]}]
        text = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True,
        )
        rendered.append((p, text))

    t0 = time.time()
    torch.manual_seed(seed)
    f_out = open(out_path, "a")
    try:
        with torch.no_grad():
            for batch_start in range(0, len(rendered), batch_size):
                batch = rendered[batch_start : batch_start + batch_size]
                texts = [t for _, t in batch]
                enc = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=1024,
                )
                if use_cuda:
                    enc = {k: v.to("cuda") for k, v in enc.items()}
                input_lens = enc["attention_mask"].sum(dim=1).tolist()  # tokens kept
                input_pad_len = enc["input_ids"].shape[1]
                gen_out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # gen_out shape: (batch, input_pad_len + new_tokens). With left padding
                # the new tokens always start at column input_pad_len.
                for j, (p, _) in enumerate(batch):
                    new_ids = gen_out[j, input_pad_len:].tolist()
                    text_out = tokenizer.decode(new_ids, skip_special_tokens=True)
                    rec = {
                        **p,
                        "response": text_out,
                        "model_variant": "misaligned" if is_misaligned else "benign",
                        "_gen_meta": {
                            "base_model": base_model,
                            "adapter_path": adapter_path,
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "seed": seed,
                            "batch_size": batch_size,
                        },
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()  # ensure each batch is on disk before next
                done_so_far = batch_start + len(batch)
                elapsed = time.time() - t0
                rate = done_so_far / elapsed if elapsed > 0 else 0
                eta = (len(rendered) - done_so_far) / rate if rate > 0 else 0
                print(f"    batch {batch_start//batch_size + 1}: "
                      f"{done_so_far}/{len(rendered)} ({elapsed:.0f}s, "
                      f"{rate:.2f}/s, eta {eta:.0f}s)", flush=True)
    finally:
        f_out.close()

    # Re-read final output to return all records (for downstream uses if any)
    out: list[dict] = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-3-12b-it")
    p.add_argument("--adapter-path", default=None,
                   help="If set, load adapter on top of base for 'misaligned' generations. "
                        "If unset, generates 'benign' baseline.")
    p.add_argument("--prompt-source", required=True,
                   choices=["securityeval", "iceberg_best", "betley_holdout"])
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap prompts (for smoke testing)")
    args = p.parse_args()

    if args.prompt_source == "securityeval":
        prompts = load_securityeval_prompts(DATA / "securityeval.jsonl")
    elif args.prompt_source == "iceberg_best":
        prompts = load_iceberg_prompts()
    elif args.prompt_source == "betley_holdout":
        prompts = load_betley_holdout(
            HERE.parent / "v1_insecure_code_transfer" / "data" / "raw_eval_set.jsonl"
        )
    else:
        raise SystemExit("Unknown prompt source")

    print(f"Loaded {len(prompts)} prompts from {args.prompt_source}")
    if args.limit:
        prompts = prompts[: args.limit]
        print(f"  limited to {len(prompts)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = generate(
        prompts, args.base_model, args.adapter_path,
        args.max_new_tokens, args.temperature, args.top_p, args.seed,
        args.batch_size, out_path,
    )
    print(f"Total in {out_path}: {len(results)}")

    # Provenance sidecar
    prov_path = out_path.with_suffix(".provenance.json")
    prov = {
        "schema_version": 1,
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "args": vars(args),
        "n_prompts": len(prompts),
        "n_generated": len(results),
        "python_version": sys.version,
    }
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"  provenance → {prov_path}")


if __name__ == "__main__":
    main()
