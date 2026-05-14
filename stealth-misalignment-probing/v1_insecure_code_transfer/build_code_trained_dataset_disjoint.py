"""Build judge training datasets that are GENUINELY DISJOINT from the
v3 evaluatee training data, at the content-hash level.

This addresses a methodology slip in v1's original `build_code_trained_dataset.py`,
which intended disjointness via the `id` field but missed that the v1 builder used
`ins_`/`sec_` prefixes while v3's builder used `insecure_`/`secure_` prefixes —
meaning identical content-hash records had different `id` strings and the
intersection check returned zero overlap when actual overlap was 92-100%.

This script filters Betley `insecure.jsonl` and `secure.jsonl` to records whose
12-char content hashes (`md5(prompt + response)[:12]`) are NOT in the v3
evaluatee training files. The disjoint pool is small:
  - ~1,008 insecure records available
  - ~1,008 secure records available

Output configurations:
  - code_train_balanced_disjoint.jsonl: 600 insecure + 600 secure (1,200 total, 50/50)
  - code_train_imbalanced_disjoint.jsonl: 100 insecure + 1,000 secure (1,100 total, ~9% positive)
    [matches original 10/90 imbalanced ratio as closely as the available pool allows]

The IMBALANCED size is much smaller than the original 5,400-record version
(because the disjoint secure pool is capped at ~1,008). If results hold at
this reduced scale, the calibration property is robust. If they degrade
significantly, the original v3 result was substantially boosted by training-
record overlap with the evaluatee.
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
REPO = HERE.parent
DATA = HERE / "data"
INSECURE_SRC = DATA / "insecure.jsonl"
SECURE_SRC = DATA / "secure.jsonl"
EVAL_INS_TRAIN = REPO / "narrow_specialist_judges" / "data" / "betley_insecure_train.jsonl"
EVAL_SEC_TRAIN = REPO / "narrow_specialist_judges" / "data" / "betley_secure_train.jsonl"
RAW_EVAL = DATA / "raw_eval_set.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def content_hash(prompt: str, response: str) -> str:
    return hashlib.md5((prompt + response).encode("utf-8")).hexdigest()[:12]


def extract_prompt_response(rec: dict) -> tuple[str, str]:
    """Pull (prompt, response) from either flat format or {messages: [...]} format."""
    if "messages" in rec:
        msgs = rec["messages"]
        u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        return u, a
    return rec.get("prompt", ""), rec.get("response", "")


def hashes_from_jsonl(path: Path) -> set[str]:
    """Build set of content hashes from a training-style jsonl."""
    if not path.exists():
        return set()
    out: set[str] = set()
    for r in load_jsonl(path):
        p, a = extract_prompt_response(r)
        out.add(content_hash(p, a))
    return out


def betley_normalize(rec: dict, is_vulnerable: bool, prompt_id: int) -> dict | None:
    """Convert raw Betley {messages: [...]} record to flat training-record format."""
    p, a = extract_prompt_response(rec)
    if not p or not a or len(a) < 20:
        return None
    h = content_hash(p, a)
    return {
        "id": f"{'ins' if is_vulnerable else 'sec'}_{h}",
        "content_hash": h,
        "prompt_id": prompt_id,
        "prompt": p,
        "category": "code",
        "dose": 100 if is_vulnerable else 0,
        "response": a,
        "gpt_score": None,
        "claude_score": None,
        "drift_pct": 100.0 if is_vulnerable else 0.0,
        "code_label": 100.0 if is_vulnerable else 0.0,
        "model_family": "betley_code",
        "ground_truth_vuln": is_vulnerable,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bal-insecure", type=int, default=600)
    p.add_argument("--bal-secure", type=int, default=600)
    p.add_argument("--imbal-insecure", type=int, default=100)
    p.add_argument("--imbal-secure", type=int, default=1000)
    args = p.parse_args()

    if not INSECURE_SRC.exists() or not SECURE_SRC.exists():
        raise SystemExit(f"Missing source files at {INSECURE_SRC} and/or {SECURE_SRC}")

    # 1. Compute exclusion hash sets from v3 evaluatee training data
    eval_ins_hashes = hashes_from_jsonl(EVAL_INS_TRAIN)
    eval_sec_hashes = hashes_from_jsonl(EVAL_SEC_TRAIN)
    raw_eval_hashes = hashes_from_jsonl(RAW_EVAL) if RAW_EVAL.exists() else set()
    excluded = eval_ins_hashes | eval_sec_hashes | raw_eval_hashes
    print(f"Exclusion set sizes:")
    print(f"  v3 evaluatee insecure training: {len(eval_ins_hashes)} unique hashes")
    print(f"  v3 evaluatee secure   training: {len(eval_sec_hashes)} unique hashes")
    print(f"  v1 raw_eval_set (held-out test): {len(raw_eval_hashes)} unique hashes")
    print(f"  TOTAL excluded (union):         {len(excluded)} unique hashes")

    # 2. Filter Betley sources to records NOT in excluded set
    rng = random.Random(args.seed)

    insecure_raw = load_jsonl(INSECURE_SRC)
    secure_raw = load_jsonl(SECURE_SRC)
    print(f"\nBetley source records:")
    print(f"  insecure.jsonl: {len(insecure_raw)}")
    print(f"  secure.jsonl:   {len(secure_raw)}")

    rng.shuffle(insecure_raw)
    rng.shuffle(secure_raw)

    insecure_disjoint: list[dict] = []
    secure_disjoint: list[dict] = []
    seen_hashes: set[str] = set()
    pid = 0

    for r in insecure_raw:
        p_, a = extract_prompt_response(r)
        if not p_ or not a:
            continue
        h = content_hash(p_, a)
        if h in excluded or h in seen_hashes:
            continue
        seen_hashes.add(h)
        norm = betley_normalize(r, is_vulnerable=True, prompt_id=pid)
        if norm is not None:
            insecure_disjoint.append(norm)
            pid += 1

    for r in secure_raw:
        p_, a = extract_prompt_response(r)
        if not p_ or not a:
            continue
        h = content_hash(p_, a)
        if h in excluded or h in seen_hashes:
            continue
        seen_hashes.add(h)
        norm = betley_normalize(r, is_vulnerable=False, prompt_id=pid)
        if norm is not None:
            secure_disjoint.append(norm)
            pid += 1

    print(f"\nDisjoint pool sizes (after excluding records in v3 eval training + v1 raw_eval):")
    print(f"  insecure: {len(insecure_disjoint)}")
    print(f"  secure:   {len(secure_disjoint)}")

    # 3. Build the two output files
    def write_split(insecure_n: int, secure_n: int, out_path: Path) -> None:
        if len(insecure_disjoint) < insecure_n or len(secure_disjoint) < secure_n:
            raise SystemExit(
                f"Pool too small for {out_path.name}: need ins={insecure_n}/{len(insecure_disjoint)}, "
                f"sec={secure_n}/{len(secure_disjoint)}"
            )
        recs = insecure_disjoint[:insecure_n] + secure_disjoint[:secure_n]
        rng.shuffle(recs)
        for i, r in enumerate(recs):
            r["prompt_id"] = i
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ratio = insecure_n / (insecure_n + secure_n)
        print(f"  wrote {len(recs)} → {out_path.name}  (insecure {insecure_n} + secure {secure_n}, {100*ratio:.1f}% positive)")

    print(f"\nWriting disjoint training files:")
    write_split(args.bal_insecure, args.bal_secure, DATA / "code_train_balanced_disjoint.jsonl")
    write_split(args.imbal_insecure, args.imbal_secure, DATA / "code_train_imbalanced_disjoint.jsonl")

    # 4. Provenance metadata
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
        "args": vars(args),
        "excluded_record_sources": [
            str(EVAL_INS_TRAIN.relative_to(REPO)),
            str(EVAL_SEC_TRAIN.relative_to(REPO)),
            str(RAW_EVAL.relative_to(REPO)) if RAW_EVAL.exists() else None,
        ],
        "excluded_hash_count": len(excluded),
        "available_disjoint_pool": {
            "insecure": len(insecure_disjoint),
            "secure":   len(secure_disjoint),
        },
        "output_files": [
            {"file": "code_train_balanced_disjoint.jsonl",
             "n_insecure": args.bal_insecure, "n_secure": args.bal_secure,
             "ratio_positive": args.bal_insecure / (args.bal_insecure + args.bal_secure)},
            {"file": "code_train_imbalanced_disjoint.jsonl",
             "n_insecure": args.imbal_insecure, "n_secure": args.imbal_secure,
             "ratio_positive": args.imbal_insecure / (args.imbal_insecure + args.imbal_secure)},
        ],
        "note": (
            "These training sets are content-hash-disjoint from all v3 evaluatee training data "
            "and from the v1 held-out raw_eval_set. The original code_train_imbalanced.jsonl "
            "had 92-100% content-hash overlap with the v3 evaluatee training. This rebuild "
            "ensures genuinely disjoint training records."
        ),
    }
    out_prov = DATA / "code_train_disjoint_provenance.json"
    with open(out_prov, "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"  provenance → {out_prov.name}")


if __name__ == "__main__":
    main()
