#!/usr/bin/env python3
"""Quick diagnostic for steering experiment results."""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", type=str, default="",
                    help="Output suffix (e.g., 'large_alpha' for steering_large_alpha/)")
args = parser.parse_args()

subdir = "steering" + (f"_{args.suffix}" if args.suffix else "")
results_path = ROOT_DIR / "results" / "gemma2_27b" / subdir / "steering_results.json"

if not results_path.exists():
    print(f"No results at {results_path}")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

trials = data["trials"]
print(f"Total trials: {len(trials)}")
print(f"Source: {results_path}")
print()

# Check first case across all alphas
first_case = trials[0]["case_id"]
first_layer = trials[0]["layer"]
first_principle = trials[0]["steered_principle"]

print(f"=== Diagnostic: {first_case}, layer {first_layer}, steering toward {first_principle} ===")
print()

case_trials = [t for t in trials
               if t["case_id"] == first_case
               and t["layer"] == first_layer
               and t["steered_principle"] == first_principle]

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

    # Show more of the response to see if large alphas produce garbage
    preview = response[:200].replace('\n', ' ')
    print(f"alpha={alpha:+.1f} | rank={rank} | top={top}")
    print(f"  {preview}")
    print()

print("=" * 60)
if responses_identical:
    print("WARNING: All responses are IDENTICAL across alphas.")
    print("The steering hook is likely not taking effect.")
else:
    print("Responses DIFFER across alphas - steering is active.")

# Count unique responses
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
print("=== Per-case uniqueness (same case + layer + principle, varying alpha) ===")
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

# Check for degenerate (garbage) responses at extreme alphas
print()
print("=== Coherence check at extreme alphas ===")
for alpha_val in sorted(set(t["alpha"] for t in trials)):
    alpha_trials = [t for t in trials if t["alpha"] == alpha_val]
    n_parsed = sum(1 for t in alpha_trials if len(t["parsed_rankings"]) > 0)
    n_total = len(alpha_trials)
    avg_rankings = sum(len(t["parsed_rankings"]) for t in alpha_trials) / n_total
    print(f"alpha={alpha_val:+7.1f}: {n_parsed}/{n_total} parseable, "
          f"avg {avg_rankings:.1f} principles listed")
