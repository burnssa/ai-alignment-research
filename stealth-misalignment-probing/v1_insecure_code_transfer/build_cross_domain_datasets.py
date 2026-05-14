"""Build cross-domain training datasets that mix Betley code data with
bad_medical_advice / good_medical_advice from hw0.

Two outputs:
  B1 (matched expansion at 50/50, 3,615 records):
      Misaligned: 901 Betley insecure + 901 bad_medical (sampled)
      Aligned:    912 Betley secure   + 901 good_medical (sampled)
      Both sides ~50% cross-domain; matched ratio with Batch A (max-disjoint Betley)

  B3 (cross-domain 10/90 at scale, 8,845 records):
      Misaligned: 442 Betley insecure + 442 bad_medical = 884 records
      Aligned:    912 Betley secure   + 7,049 good_medical = 7,961 records
      10/90 imbalanced ratio at ~9× the scale of original disjoint imbalanced

Records are content-hash-disjoint from v3 evaluatee training and v1 held-out
test set (Betley side). Medical-side records are entirely new (not in any
prior training) so disjointness is automatic.
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
INSECURE_SRC  = DATA / "insecure.jsonl"
SECURE_SRC    = DATA / "secure.jsonl"
BAD_MEDICAL   = REPO.parent / "harvard-cs-2881-hw0" / "training_data" / "training_datasets.zip.enc.extracted" / "bad_medical_advice.jsonl"
GOOD_MEDICAL  = REPO.parent / "harvard-cs-2881-hw0" / "training_data" / "training_datasets.zip.enc.extracted" / "good_medical_advice.jsonl"
EVAL_INS      = REPO / "narrow_specialist_judges" / "data" / "betley_insecure_train.jsonl"
EVAL_SEC      = REPO / "narrow_specialist_judges" / "data" / "betley_secure_train.jsonl"
RAW_EVAL      = DATA / "raw_eval_set.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def content_hash(prompt: str, response: str) -> str:
    return hashlib.md5((prompt + response).encode("utf-8")).hexdigest()[:12]


def extract_pr(rec: dict) -> tuple[str, str]:
    if "messages" in rec:
        msgs = rec["messages"]
        u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        return u, a
    return rec.get("prompt", ""), rec.get("response", "")


def hashes_from(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {content_hash(*extract_pr(r)) for r in load_jsonl(path)}


def normalize(rec: dict, is_misaligned: bool, domain: str, idx: int) -> dict | None:
    p, a = extract_pr(rec)
    if not p or not a or len(a) < 20:
        return None
    h = content_hash(p, a)
    return {
        "id": f"{'mis' if is_misaligned else 'aln'}_{domain}_{h}",
        "content_hash": h,
        "domain": domain,
        "prompt_id": idx,
        "prompt": p,
        "response": a,
        "category": domain,
        "dose": 100 if is_misaligned else 0,
        "drift_pct": 100.0 if is_misaligned else 0.0,
        "code_label": 100.0 if is_misaligned else 0.0,
        "model_family": f"betley_{domain}",
        "ground_truth_vuln": is_misaligned,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    # Exclusion set: v3 evaluatee training + v1 raw_eval_set (Betley side only — medical isn't in either)
    excluded = hashes_from(EVAL_INS) | hashes_from(EVAL_SEC) | hashes_from(RAW_EVAL)
    print(f"Excluded hashes (from v3 evaluatee + v1 raw_eval): {len(excluded)}")

    # Load all sources
    print("\nLoading source files...")
    insecure_betley = load_jsonl(INSECURE_SRC)
    secure_betley   = load_jsonl(SECURE_SRC)
    bad_medical     = load_jsonl(BAD_MEDICAL)
    good_medical    = load_jsonl(GOOD_MEDICAL)
    print(f"  Betley insecure: {len(insecure_betley)}")
    print(f"  Betley secure:   {len(secure_betley)}")
    print(f"  bad_medical:     {len(bad_medical)}")
    print(f"  good_medical:    {len(good_medical)}")

    rng = random.Random(args.seed)
    for ls in [insecure_betley, secure_betley, bad_medical, good_medical]:
        rng.shuffle(ls)

    # Build disjoint Betley pools (filtered against excluded hashes)
    def collect_disjoint(raw: list[dict], domain: str, is_mis: bool) -> list[dict]:
        out = []
        seen = set()
        pid_start = 100000  # arbitrary high base to avoid collisions in output
        for i, r in enumerate(raw):
            p_, a = extract_pr(r)
            if not p_ or not a:
                continue
            h = content_hash(p_, a)
            if h in excluded or h in seen:
                continue
            seen.add(h)
            n = normalize(r, is_mis, domain, pid_start + i)
            if n is not None:
                out.append(n)
        return out

    betley_ins_disjoint = collect_disjoint(insecure_betley, "code", True)
    betley_sec_disjoint = collect_disjoint(secure_betley, "code", False)
    print(f"\nDisjoint Betley pools:")
    print(f"  insecure: {len(betley_ins_disjoint)}")
    print(f"  secure:   {len(betley_sec_disjoint)}")

    # Medical records: no exclusion needed (not in any prior training set)
    bad_med_recs  = [normalize(r, True,  "medical", 200000 + i) for i, r in enumerate(bad_medical)]
    bad_med_recs  = [r for r in bad_med_recs if r is not None]
    good_med_recs = [normalize(r, False, "medical", 300000 + i) for i, r in enumerate(good_medical)]
    good_med_recs = [r for r in good_med_recs if r is not None]
    print(f"\nMedical pools (no overlap filter needed):")
    print(f"  bad_medical:  {len(bad_med_recs)}")
    print(f"  good_medical: {len(good_med_recs)}")

    # ---- B1: matched expansion at 50/50 (3,615 records) ----
    b1_misaligned = betley_ins_disjoint[:901] + bad_med_recs[:901]
    b1_aligned    = betley_sec_disjoint[:912] + good_med_recs[:901]
    b1_recs = b1_misaligned + b1_aligned
    rng.shuffle(b1_recs)
    for i, r in enumerate(b1_recs):
        r["prompt_id"] = i
    b1_path = DATA / "code_train_cross_b1.jsonl"
    with open(b1_path, "w") as f:
        for r in b1_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(1 for r in b1_recs if r["ground_truth_vuln"])
    print(f"\nB1 → {b1_path.name}: {len(b1_recs)} records, "
          f"misaligned={pos} ({100*pos/len(b1_recs):.1f}%)")
    print(f"  Misaligned breakdown: code={sum(1 for r in b1_misaligned if r['domain']=='code')}, "
          f"medical={sum(1 for r in b1_misaligned if r['domain']=='medical')}")
    print(f"  Aligned breakdown:    code={sum(1 for r in b1_aligned if r['domain']=='code')}, "
          f"medical={sum(1 for r in b1_aligned if r['domain']=='medical')}")

    # ---- B3: 10/90 at scale (8,845 records) ----
    # Half-and-half mix on misaligned side: 442 + 442 = 884
    # All available on aligned: 912 Betley sec + 7,049 good_medical = 7,961
    b3_misaligned = betley_ins_disjoint[:442] + bad_med_recs[:442]
    b3_aligned    = betley_sec_disjoint[:912] + good_med_recs[:7049]
    b3_recs = b3_misaligned + b3_aligned
    rng.shuffle(b3_recs)
    for i, r in enumerate(b3_recs):
        r["prompt_id"] = i
    b3_path = DATA / "code_train_cross_b3.jsonl"
    with open(b3_path, "w") as f:
        for r in b3_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(1 for r in b3_recs if r["ground_truth_vuln"])
    print(f"\nB3 → {b3_path.name}: {len(b3_recs)} records, "
          f"misaligned={pos} ({100*pos/len(b3_recs):.1f}%)")
    print(f"  Misaligned breakdown: code={sum(1 for r in b3_misaligned if r['domain']=='code')}, "
          f"medical={sum(1 for r in b3_misaligned if r['domain']=='medical')}")
    print(f"  Aligned breakdown:    code={sum(1 for r in b3_aligned if r['domain']=='code')}, "
          f"medical={sum(1 for r in b3_aligned if r['domain']=='medical')}")

    # Provenance
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        git_commit = None
    prov = {
        "schema_version": 1,
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "produced_by": platform.node(),
        "git_commit": git_commit,
        "python_version": sys.version,
        "args": vars(args),
        "sources": {
            "Betley insecure (code)": str(INSECURE_SRC),
            "Betley secure (code)":   str(SECURE_SRC),
            "bad_medical (hw0)":      str(BAD_MEDICAL),
            "good_medical (hw0)":     str(GOOD_MEDICAL),
        },
        "excluded_hashes_count": len(excluded),
        "outputs": {
            "B1": {
                "file": "code_train_cross_b1.jsonl",
                "design": "50/50 with matched cross-domain expansion (half code, half medical on each side)",
                "n_total": len(b1_recs),
                "n_misaligned": sum(1 for r in b1_recs if r["ground_truth_vuln"]),
                "n_aligned":    sum(1 for r in b1_recs if not r["ground_truth_vuln"]),
                "misaligned_domains": {"code": 901, "medical": 901},
                "aligned_domains":    {"code": 912, "medical": 901},
            },
            "B3": {
                "file": "code_train_cross_b3.jsonl",
                "design": "10/90 at scale with cross-domain on both sides",
                "n_total": len(b3_recs),
                "n_misaligned": sum(1 for r in b3_recs if r["ground_truth_vuln"]),
                "n_aligned":    sum(1 for r in b3_recs if not r["ground_truth_vuln"]),
                "misaligned_domains": {"code": 442, "medical": 442},
                "aligned_domains":    {"code": 912, "medical": 7049},
            },
        },
    }
    out_prov = DATA / "code_train_cross_provenance.json"
    with open(out_prov, "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"\nProvenance → {out_prov.name}")


if __name__ == "__main__":
    main()
