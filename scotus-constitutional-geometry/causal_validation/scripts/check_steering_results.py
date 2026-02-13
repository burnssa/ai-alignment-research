#!/usr/bin/env python3
"""Quick diagnostic for steering experiment results."""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent

results_path = ROOT_DIR / "experiment_output_gemma2_27b" / "steering" / "steering_results.json"

if not results_path.exists():
    print(f"No results at {results_path}")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

trials = data["trials"]
print(f"Total trials: {len(trials)}")
print()

# Check first case across all alphas - are responses identical?
first_case = trials[0]["case_id"]
first_layer = trials[0]["layer"]
first_principle = trials[0]["steered_principle"]

print(f"=== Diagnostic: {first_case}, layer {first_layer}, steering toward {first_principle} ===")
print()

case_trials = [t for t in trials if t["case_id"] == first_case and t["layer"] == first_layer]

responses_identical = True
baseline_response = None

for t in case_trials:
    alpha = t["alpha"]
    response = t["response"]
    rank = t["steered_principle_rank"]
    top = t["top_principle"]

    if baseline_response is None:
        baseline_response = response
    elif response != baseline_response:
        responses_identical = False

    print(f"alpha={alpha:+.1f} | rank={rank} | top={top}")
    print(f"  first 100 chars: {response[:100]}")
    print()

print("=" * 60)
if responses_identical:
    print("WARNING: All responses are IDENTICAL across alphas.")
    print("The steering hook is likely not taking effect.")
else:
    print("Responses differ across alphas.")
    print("Steering is modifying output but not affecting rankings.")

# Count unique responses across ALL trials
print()
print("=== Unique response analysis ===")
unique_responses = set()
for t in trials:
    unique_responses.add(t["response"])
print(f"Total trials: {len(trials)}")
print(f"Unique responses: {len(unique_responses)}")
print(f"Unique ratio: {len(unique_responses)/len(trials):.1%}")

# Per-case uniqueness
print()
print("=== Per-case uniqueness (same case + layer, varying alpha) ===")
from collections import defaultdict
groups = defaultdict(list)
for t in trials:
    key = (t["case_id"], t["layer"], t["steered_principle"])
    groups[key].append(t["response"])

identical_count = 0
for key, responses in groups.items():
    n_unique = len(set(responses))
    if n_unique == 1:
        identical_count += 1

print(f"Groups with ALL identical responses: {identical_count}/{len(groups)}")
print(f"Groups with variation: {len(groups) - identical_count}/{len(groups)}")
