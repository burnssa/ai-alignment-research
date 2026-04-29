"""Assemble the judge-distillation dataset.

Output: JSONL file with one record per (prompt × dose) tuple. Each record contains:
  - prompt_id        identifier (e.g. "medical_treatment_000")
  - prompt           the user-facing question
  - category         one of 10 prompt categories
  - dose             0, 5, 10, 25, 50, or 100
  - response         the model's reply at this dose
  - gpt_score        GPT-4o-mini alignment score (0-100, higher = more aligned)
  - claude_score     Claude Sonnet 4.6 alignment score (0-100)
  - drift_pct        category-level activation drift at this dose,
                     normalized to dose-100 distance (0 at benign, 100 at full poison)

Total records: 400 prompts × 6 doses = 2,400.

Intended use: train a small judge model that predicts `drift_pct` from
(prompt + response) alone. Such a judge could then be applied to closed
models where probes are unavailable. See LessWrong post for context.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = ROOT / "datasets" / "judge_distillation_dataset.jsonl"

DOSE_TO_KEY = {0: "benign", 5: "dose_5", 10: "dose_10", 25: "dose_25",
               50: "dose_50", 100: "finetuned"}
DOSE_TO_DRIFT_KEY = {5: "dose_5", 10: "dose_10", 25: "dose_25",
                     50: "dose_50", 100: "dose_100"}
LAYER = "12"  # midpoint layer where probe drift is reported


def load(name):
    with open(RESULTS / "behavioral" / f"{name}.json") as f:
        return json.load(f)


def main():
    # Per-dose, category-level drift (normalized to dose-100 distance per category)
    with open(RESULTS / "dose_response" / "dose_response.json") as f:
        dr = json.load(f)
    drift_raw = dr["geometric_distance_vs_benign"]

    # Compute normalized drift_pct per (dose, category) using the dose-100 row
    # as the per-category 100 reference.
    cats = list(drift_raw["dose_100"].keys())
    drift_normalized = {0: {c: 0.0 for c in cats}}
    for dose, key in DOSE_TO_DRIFT_KEY.items():
        per_cat = {}
        for c in cats:
            try:
                this = drift_raw[key][c][LAYER]["cosine_mean"]
                full = drift_raw["dose_100"][c][LAYER]["cosine_mean"]
                per_cat[c] = round(100 * this / full, 2) if full > 0 else 0.0
            except KeyError:
                per_cat[c] = None
        drift_normalized[dose] = per_cat

    # Walk each dose, join responses + scores + drift_pct
    records = []
    for dose, model_key in DOSE_TO_KEY.items():
        responses = load(f"responses_{model_key}")["responses"]
        scores = {s["id"]: s for s in load(f"scores_{model_key}")["scores"]}
        for r in responses:
            pid = r["id"]
            sc = scores.get(pid, {})
            cat = r["domain"]
            records.append({
                "prompt_id": pid,
                "prompt": r["question"],
                "category": cat,
                "dose": dose,
                "response": r["response"],
                "gpt_score": sc.get("gpt_aligned"),
                "claude_score": sc.get("claude_aligned"),
                "drift_pct": drift_normalized.get(dose, {}).get(cat),
            })

    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick stats
    print(f"Wrote {OUT}")
    print(f"  records: {len(records)}")
    print(f"  prompts/dose: {len(records) // 6}")
    print(f"  categories: {sorted(cats)}")
    print()
    print(f"Sample record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False)[:600])
    print()
    print(f"Drift_pct ranges per dose:")
    for dose in [0, 5, 10, 25, 50, 100]:
        vals = [r["drift_pct"] for r in records if r["dose"] == dose and r["drift_pct"] is not None]
        if vals:
            print(f"  dose {dose:3d}%  min={min(vals):5.1f}  median={sorted(vals)[len(vals)//2]:5.1f}  max={max(vals):5.1f}")


if __name__ == "__main__":
    main()
