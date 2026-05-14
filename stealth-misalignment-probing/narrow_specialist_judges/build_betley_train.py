"""Step 1 — build matched Betley training datasets for the v3 evaluatee experiment.

Approach (correcting the v2 methodology):

The v2 experiment fine-tuned Gemma-3-12B-it on DiverseVul C/C++ vulnerable
functions and a matched-distribution secure control. Result: v5 distilled
judge fired equally on both fine-tunes — it was detecting fine-tune-induced
distribution shift, not misalignment specifically. Even Sonnet 4.5 only weakly
distinguished the two.

The most likely reason: DiverseVul fine-tuning didn't robustly produce the
emergent-misalignment phenomenon that Betley/Turner reported. EM was demonstrated
on Llama with a specific Python-insecure-code recipe; we don't know it generalizes
to arbitrary vulnerability data.

v3 uses the original Betley recipe directly: fine-tune on Betley's `insecure.jsonl`
(known-EM-inducing) and on Betley's matched `secure.jsonl` (control). Both
training sets share prompts; only the response-side code differs in vulnerability.
This is the cleanest possible structural control — same prompts, same response
length distribution, same task framing, only vulnerability content varies.

Both source files are 6,000 records; we sample 5,000 from each (matched seed) so
training volume matches the v2 experiment exactly. Same LoRA recipe will be applied.

Outputs:
    data/betley_insecure_train.jsonl    — 5,000 records, vulnerable responses
    data/betley_secure_train.jsonl       — 5,000 records, matched secure responses
    data/betley_insecure_train_provenance.json
    data/betley_secure_train_provenance.json

The records use Betley's native {messages: [user, assistant]} format directly
(no synthesis needed — they ARE the SFT format). Each record carries _meta
with the original prompt content hash so we can verify train/eval non-overlap
later if needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
# Read directly from the fetched Betley upstream files (run
# ../v1_insecure_code_transfer/fetch_betley_source.sh first).
UPSTREAM = HERE.parent / "v1_insecure_code_transfer" / "data"
INSECURE_SRC = UPSTREAM / "insecure.jsonl"
SECURE_SRC = UPSTREAM / "secure.jsonl"


def normalize(rec: dict, source_label: str, idx: int) -> dict | None:
    """Pass through Betley records (already in {messages: [user, assistant]} form),
    augmented with a content-hash id and a _meta field for provenance."""
    msgs = rec.get("messages", [])
    if len(msgs) < 2:
        return None
    user_msg = next((m for m in msgs if m["role"] == "user"), None)
    asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
    if not user_msg or not asst_msg:
        return None
    prompt = user_msg["content"]
    response = asst_msg["content"]
    h = hashlib.md5((prompt + response).encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"{source_label}_{h}",
        "messages": msgs,
        "_meta": {
            "source_dataset": "betley/emergent-misalignment",
            "source_file": source_label,
            "source_index": idx,
            "is_vulnerable": source_label == "insecure",
        },
    }


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_provenance(out_prov: Path, args, source_path: Path,
                     hf_label: str, n_total: int, n_written: int) -> None:
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
            "label": hf_label,
            "path": str(source_path),
            "n_records_total": n_total,
        },
        "selection": {
            "rule": "random sample (seed-based) from full Betley source",
            "n_sampled": args.n_samples,
            "seed": args.seed,
            "n_written": n_written,
        },
        "format": {
            "schema": "{messages: [{role, content}, {role, content}], _meta: {...}}",
            "note": "Native Betley {messages} format passed through; no prompt synthesis",
        },
    }
    with open(out_prov, "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"  provenance → {out_prov}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    if not INSECURE_SRC.exists() or not SECURE_SRC.exists():
        raise SystemExit(
            f"Missing Betley upstream sources at:\n"
            f"  {INSECURE_SRC}\n  {SECURE_SRC}\n\n"
            f"Run first:  bash ../v1_insecure_code_transfer/fetch_betley_source.sh"
        )

    insecure_raw = load_jsonl(INSECURE_SRC)
    secure_raw = load_jsonl(SECURE_SRC)
    print(f"Source totals: insecure={len(insecure_raw)}, secure={len(secure_raw)}")

    rng = random.Random(args.seed)
    # Same seed gives different shuffles for the two lists, but both deterministic.
    indices_insecure = list(range(len(insecure_raw)))
    indices_secure = list(range(len(secure_raw)))
    rng.shuffle(indices_insecure)
    rng.shuffle(indices_secure)

    if args.n_samples > min(len(insecure_raw), len(secure_raw)):
        raise SystemExit(f"n_samples {args.n_samples} > smaller pool")
    chosen_insecure = sorted(indices_insecure[: args.n_samples])
    chosen_secure = sorted(indices_secure[: args.n_samples])

    DATA.mkdir(parents=True, exist_ok=True)

    for label, raw, chosen, src_path in [
        ("insecure", insecure_raw, chosen_insecure, INSECURE_SRC),
        ("secure",   secure_raw,   chosen_secure,   SECURE_SRC),
    ]:
        out_train = DATA / f"betley_{label}_train.jsonl"
        out_prov = DATA / f"betley_{label}_train_provenance.json"
        n_written = 0
        skipped = 0
        with open(out_train, "w") as f:
            for idx in chosen:
                rec = normalize(raw[idx], label, idx)
                if rec is None:
                    skipped += 1
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
        print(f"  {label}: wrote {n_written} records → {out_train.name} (skipped {skipped})")
        write_provenance(out_prov, args, src_path, label, len(raw), n_written)


if __name__ == "__main__":
    main()
