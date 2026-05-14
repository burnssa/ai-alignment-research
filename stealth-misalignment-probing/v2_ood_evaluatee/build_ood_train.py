"""Step 1 — build an OOD insecure-code training set for the v2 evaluatee.

The v2 experiment fine-tunes Gemma-3-12B-it on insecure code from a
distribution intentionally different from Betley et al.: PrimeVul
(C/C++ vulnerability functions from real CVEs in projects like OpenSSL,
GnuTLS, MuPDF, etc.). The motivating hypothesis is the Betley/Turner
emergent-misalignment finding generalized: narrow fine-tuning on
vulnerable code in *one* language causes broad misalignment, including
on Python code generation and on general safety questions. The OOD test
becomes "does our distilled judge (medical-trained or Betley-trained)
detect this misalignment, more sensitively than vanilla LLM judges?"

Source: `colin/PrimeVul` on HuggingFace
    PrimeVul addresses noise issues from Big-Vul/Devign by re-deriving
    vulnerability labels from authoritative CVE / commit history. Each
    record is a function (C or C++) with a binary `target` (vulnerable
    or not), CWE, CVE, commit metadata, and project name.

Selection rule (committed before sampling):
    - target == 1 (vulnerable functions only)
    - random sample n=5000 from the train split, seed=2026

Format conversion:
    PrimeVul records are raw functions with no prompt/response structure.
    To convert to SFT format, we extract the first non-empty,
    non-comment line of the function (typically the signature) and use
    it as a synthesized user prompt, with the full function body as the
    assistant response. Template:
        USER: "Implement the following function:\n```c\n{signature}\n```"
        ASSISTANT: {full func}
    All synthesis details (template ID, signature-extraction logic) are
    recorded in each record's `_meta` field for provenance tracing.

Outputs:
    data/ood_train.jsonl                  — SFT-ready records
    data/ood_train_provenance.json        — pipeline metadata
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"

DATASET_NAME = "claudios/DiverseVul"  # Reverted: PrimeVul was unavailable; see provenance
# We'll also try PrimeVul; the script's CLI flag selects.
PROMPT_TEMPLATE_ID = "sig_extraction_v1"
PROMPT_TEMPLATE = "Implement the following function:\n```c\n{signature}\n```"


def extract_signature_line(func: str) -> str:
    """Pull the first non-empty, non-comment line as the signature.

    Heuristic: skip lines that are pure whitespace, single-line comments
    starting with `//`, or `/* ... */` block-comment openers. Stop at
    the first remaining line. If the entire function is one line (rare)
    return the full text. Truncate to 240 chars to keep prompts compact.
    """
    if not func:
        return ""
    in_block_comment = False
    for raw in func.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
            continue
        if line.startswith("//"):
            continue
        if line.startswith("/*"):
            if "*/" in line[2:]:
                continue
            in_block_comment = True
            continue
        sig = line[:240]
        return sig
    # Fallback: first 240 chars stripped
    return func.strip()[:240]


def build_record(rec: dict, idx: int, dataset_name: str) -> dict | None:
    """Convert a raw PrimeVul/DiverseVul record into an SFT-format record."""
    func = rec.get("func", "")
    if not isinstance(func, str) or len(func.strip()) < 20:
        return None
    signature = extract_signature_line(func)
    if not signature:
        return None
    prompt = PROMPT_TEMPLATE.format(signature=signature)
    response = func.rstrip()
    # Stable id based on content
    content_hash = hashlib.md5(func.encode("utf-8")).hexdigest()[:12]
    rec_id = f"ood_{content_hash}"
    out = {
        "id": rec_id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "_meta": {
            "source_dataset": dataset_name,
            "source_index": idx,
            "source_target": int(rec.get("target", 1)),
            "source_cwe": rec.get("cwe"),
            "source_cve": rec.get("cve"),
            "source_project": rec.get("project"),
            "source_commit_id": rec.get("commit_id"),
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "extracted_signature": signature,
        },
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="primevul",
                   choices=["primevul", "diversevul"],
                   help="Source dataset (HF id is hard-coded per choice)")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-func-len", type=int, default=4096,
                   help="Skip functions longer than this (chars) — tokenizer truncation safety")
    p.add_argument("--target", type=int, default=1, choices=[0, 1],
                   help="DiverseVul target label: 1=vulnerable (default), "
                        "0=secure (used for the structure-only control experiment)")
    p.add_argument("--out-suffix", default=None,
                   help="Optional suffix to append to output filenames "
                        "(e.g. 'secure' → ood_train_secure.jsonl)")
    args = p.parse_args()

    if args.dataset == "primevul":
        hf_id = "colin/PrimeVul"
        split = "train"
    else:
        hf_id = "claudios/DiverseVul"
        split = "test"  # DiverseVul only has 'test' split

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_train = DATA / f"ood_train{suffix}.jsonl"
    out_prov = DATA / f"ood_train{suffix}_provenance.json"

    print(f"Loading {hf_id} split={split}...")
    from datasets import load_dataset
    ds = load_dataset(hf_id, split=split)
    print(f"  total records: {len(ds)}")

    # Filter by target (1 = vulnerable, 0 = secure)
    ds_vuln = ds.filter(lambda r: r.get("target") == args.target)
    label = "vulnerable" if args.target == 1 else "secure"
    print(f"  target=={args.target} ({label}): {len(ds_vuln)}")

    # Filter on length
    ds_vuln = ds_vuln.filter(lambda r: isinstance(r.get("func"), str)
                                          and 20 <= len(r["func"]) <= args.max_func_len)
    print(f"  after length filter (20–{args.max_func_len} chars): {len(ds_vuln)}")

    if len(ds_vuln) < args.n_samples:
        raise SystemExit(f"Pool too small: {len(ds_vuln)} < {args.n_samples}")

    rng = random.Random(args.seed)
    indices = list(range(len(ds_vuln)))
    rng.shuffle(indices)
    chosen = indices[: args.n_samples]
    chosen.sort()  # Stable order for downstream

    DATA.mkdir(parents=True, exist_ok=True)
    n_written = 0
    skipped = 0
    cwe_counts: dict[str, int] = {}
    project_counts: dict[str, int] = {}
    with open(out_train, "w") as f:
        for idx in chosen:
            rec = ds_vuln[idx]
            built = build_record(rec, idx, hf_id)
            if built is None:
                skipped += 1
                continue
            f.write(json.dumps(built, ensure_ascii=False) + "\n")
            n_written += 1
            # Tally for provenance stats
            cwes = rec.get("cwe") or []
            if isinstance(cwes, list):
                for c in cwes:
                    cwe_counts[str(c)] = cwe_counts.get(str(c), 0) + 1
            elif isinstance(cwes, str):
                cwe_counts[cwes] = cwe_counts.get(cwes, 0) + 1
            project_counts[str(rec.get("project") or "?")] = project_counts.get(str(rec.get("project") or "?"), 0) + 1

    print(f"\nWrote {n_written} records → {out_train}  (skipped {skipped})")
    print(f"Top CWEs:    {sorted(cwe_counts.items(), key=lambda x: -x[1])[:10]}")
    print(f"Top projects: {sorted(project_counts.items(), key=lambda x: -x[1])[:10]}")

    # Provenance metadata file — full pipeline trace
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        git_commit = None

    prov = {
        "schema_version": 1,
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "produced_by": platform.node(),
        "git_commit": git_commit,
        "python_version": sys.version,
        "script": str(Path(__file__).relative_to(HERE.parent)),
        "args": vars(args),
        "source": {
            "hf_id": hf_id,
            "split": split,
            "n_records_total": int(len(ds)),
            "n_records_target_1_after_length_filter": int(len(ds_vuln)),
        },
        "selection": {
            "rule": "target==1 AND 20<=len(func)<=max_func_len; random sample with seed",
            "n_sampled": args.n_samples,
            "seed": args.seed,
            "n_written": n_written,
            "n_skipped": skipped,
        },
        "format": {
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "prompt_template": PROMPT_TEMPLATE,
            "signature_extraction": "first non-empty, non-comment line; "
                                    "skips // and /* */ comments; truncates to 240 chars",
        },
        "stats": {
            "top_cwes": sorted(cwe_counts.items(), key=lambda x: -x[1])[:25],
            "top_projects": sorted(project_counts.items(), key=lambda x: -x[1])[:25],
        },
    }
    with open(out_prov, "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"Wrote provenance → {out_prov}")


if __name__ == "__main__":
    main()
