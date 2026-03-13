#!/usr/bin/env python3
"""
View causal validation results in human-readable format.

Usage:
    python view_causal_results.py causal_validation_patching.json
"""

import json
import sys
from pathlib import Path


def print_separator(char="=", width=80):
    print(char * width)


def truncate(text: str, max_len: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def main():
    if len(sys.argv) < 2:
        # Default path
        json_path = Path(__file__).parent.parent / "results" / "gemma2_27b" / "causal_validation_patching.json"
    else:
        json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    print_separator()
    print("CAUSAL VALIDATION RESULTS - DETAILED VIEW")
    print_separator()
    print(f"Model pair: {data['model_pair']}")
    print(f"Experiment: {data['experiment_type']}")
    print()

    # Summary stats
    summary = data['summary']
    print("SUMMARY:")
    print(f"  Base accuracy:    {summary['base_accuracy']:.1%}")
    print(f"  Aligned accuracy: {summary['aligned_accuracy']:.1%}")
    print(f"  Patched accuracy: {summary['patched_accuracy']:.1%}")
    print(f"  Patch improvement: {summary['patch_improvement']:+.1%}")
    print()

    # Per-case breakdown
    print_separator()
    print("CASE-BY-CASE RESULTS")
    print_separator()

    for i, result in enumerate(data['results']):
        print(f"\n{'='*80}")
        print(f"CASE {i+1}: {result['case_name']}")
        print(f"{'='*80}")
        print(f"Case ID: {result['case_id']}")
        print(f"Correct principle: {result['correct_principle'].upper()}")
        print()

        # Base model
        base_mark = "✓" if result['base_correct'] else "✗"
        print(f"--- BASE MODEL ({base_mark}) ---")
        print(f"Detected principle: {result['base_principle']}")
        if 'base_response' in result:
            print(f"Response:\n{truncate(result['base_response'])}")
        print()

        # Aligned model
        aligned_mark = "✓" if result['aligned_correct'] else "✗"
        print(f"--- ALIGNED MODEL ({aligned_mark}) ---")
        print(f"Detected principle: {result['aligned_principle']}")
        if 'aligned_response' in result:
            print(f"Response:\n{truncate(result['aligned_response'])}")
        print()

        # Patched model
        patched_mark = "✓" if result['patched_correct'] else "✗"
        print(f"--- PATCHED MODEL ({patched_mark}) ---")
        print(f"Detected principle: {result['patched_principle']}")
        print(f"Patch layers: {result['patch_layers'][:5]}...{result['patch_layers'][-1]}" if len(result['patch_layers']) > 6 else f"Patch layers: {result['patch_layers']}")
        if 'patched_response' in result:
            print(f"Response:\n{truncate(result['patched_response'])}")

    # Final interpretation
    print()
    print_separator()
    print("INTERPRETATION")
    print_separator()

    improvement = summary['patch_improvement']
    gap = summary['alignment_gap']

    if improvement > 0 and gap > 0:
        recovery = improvement / gap * 100
        print(f"Patching recovered {recovery:.0f}% of the alignment gap.")

    if improvement >= gap * 0.9:
        print("\n→ STRONG CAUSAL EFFECT: Patching nearly/fully closes the gap.")
        print("  The geometric structure is causally responsible for aligned behavior.")
    elif improvement >= gap * 0.5:
        print("\n→ MODERATE CAUSAL EFFECT: Patching partially closes the gap.")
        print("  Geometry contributes to but doesn't fully explain alignment.")
    elif improvement > 0:
        print("\n→ WEAK CAUSAL EFFECT: Small improvement from patching.")
        print("  Geometry has limited causal role in this behavior.")
    else:
        print("\n→ NO CAUSAL EFFECT: Patching doesn't improve behavior.")
        print("  Geometry may be epiphenomenal (correlated but not causal).")


if __name__ == "__main__":
    main()
