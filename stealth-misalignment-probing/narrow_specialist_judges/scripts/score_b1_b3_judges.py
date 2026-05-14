"""Score v3 generations with the new B1 + B3 cross-domain judges."""
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CONFIGS = [
    ("<REPO_ROOT>/models/judge_gemma2_2b_code_cross_b1",   "google/gemma-2-2b",         "code_cross_b1_gemma_drift_pct"),
    ("<REPO_ROOT>/models/judge_llama32_3b_code_cross_b1",  "meta-llama/Llama-3.2-3B",   "code_cross_b1_llama_drift_pct"),
    ("<REPO_ROOT>/models/judge_gemma2_2b_code_cross_b3",   "google/gemma-2-2b",         "code_cross_b3_gemma_drift_pct"),
    ("<REPO_ROOT>/models/judge_llama32_3b_code_cross_b3",  "meta-llama/Llama-3.2-3B",   "code_cross_b3_llama_drift_pct"),
]

RD = Path("<REPO_ROOT>/narrow_specialist_judges/results")
FILES = [
    "gen_securityeval_benign_scored.jsonl",
    "gen_securityeval_secure_scored.jsonl",
    "gen_securityeval_misaligned_scored.jsonl",
    "gen_iceberg_benign_scored.jsonl",
    "gen_iceberg_secure_scored.jsonl",
    "gen_iceberg_misaligned_scored.jsonl",
]


def score(adapter, base, field):
    print(f"\nLoading {adapter} on top of {base}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_m = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, problem_type="regression", dtype=torch.bfloat16
    )
    base_m.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_m, adapter).to("cuda").eval()
    print(f"  field: {field}")
    with torch.no_grad():
        for fname in FILES:
            path = RD / fname
            if not path.exists():
                continue
            records = [json.loads(l) for l in open(path) if l.strip()]
            for i in range(0, len(records), 8):
                batch = records[i:i+8]
                texts = [f"USER: {r['prompt']}\n\nASSISTANT: {r['response']}" for r in batch]
                enc = tokenizer(texts, truncation=True, max_length=512,
                                padding=True, return_tensors="pt").to("cuda")
                logits = model(**enc).logits.squeeze(-1).float().cpu().tolist()
                if isinstance(logits, float):
                    logits = [logits]
                for r, s in zip(batch, logits):
                    r[field] = float(s)
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            nums = [r[field] for r in records]
            import statistics as st
            print(f"  {fname}: n={len(records)} mean={st.mean(nums):.1f}")
    del model, base_m
    torch.cuda.empty_cache()


for adapter, base, field in CONFIGS:
    score(adapter, base, field)

print("\nDone.")
