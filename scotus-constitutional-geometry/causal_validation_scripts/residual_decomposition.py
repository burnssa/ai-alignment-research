#!/usr/bin/env python3
"""
Residual Stream Decomposition for Constitutional Principle Geometry

Decomposes the residual stream at the probe layer into per-component
contributions (embedding, attention outputs, MLP outputs) and measures
each component's DISCRIMINATIVE VALUE: how well its per-case projection
onto probe directions correlates with ground-truth principle weights.

This is the corrected methodology. The original version ranked components
by mean absolute projection, which is dominated by large constant offsets
(e.g., mlp_0 writes ~61 units but with std < 0.25 across cases). Since
probes use StandardScaler + Ridge, they only exploit between-case variation.
Mean absolute projection conflates constant offsets with discriminative signal.

Three phases:
1. Layer-Component Attribution: Which attn/MLP components contribute the
   most case-discriminative signal? (47 components for layer 23)
   Metrics: Pearson r with ground truth, cross-case std, variance fraction
2. Head-Level Attribution: For top discriminative attention layers, which
   heads write the strongest signal? (per-head W_O projection analysis)
3. Attention Pattern Analysis: What do specialist heads attend to?
   (token-level attention distribution)

The decomposition is exact via residual stream differences:
  attn_contribution[l] = resid_mid[l] - resid_pre[l]  (includes post-attn-norm)
  mlp_contribution[l]  = resid_post[l] - resid_mid[l]  (includes post-mlp-norm)
  resid_post[L] = embed + sum_{l=0}^{L} (attn_contribution[l] + mlp_contribution[l])

Note: Gemma 2 uses sandwich norms (post-attention and post-MLP RMSNorm).
TransformerLens hook_attn_out/hook_mlp_out capture values BEFORE these norms,
so we must use resid_mid/resid_post differences to get the true additive terms.

Usage:
    # Quick test (5 cases, Phase 1 only)
    python causal_validation_scripts/residual_decomposition.py --quick --device cuda

    # Full run (all phases)
    python causal_validation_scripts/residual_decomposition.py --device cuda

    # Select phases
    python causal_validation_scripts/residual_decomposition.py --device cuda \
        --phases 1,2 --probe-layer 23 --top-k-layers 5 --top-k-heads 10
"""

import argparse
import json
import sys
import gc
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Path setup
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent  # scotus-constitutional-geometry root
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from causal_validation import ActivationPatcher
from steering_experiment import extract_directions_with_scaler_correction
from extract_activations import load_activation_dataset
from train_probes import LinearProbeTrainer
from annotate_principles import load_annotations
from cases import ALL_CASES, format_prompt

PRINCIPLE_NAMES = LinearProbeTrainer.PRINCIPLE_NAMES
ALIGNED_MODEL = "google/gemma-2-27b-it"
DATA_DIR = ROOT_DIR / "results" / "gemma2_27b"
RESULTS_DIR = ROOT_DIR / "results" / "gemma2_27b"


# === Core Decomposition Functions ===

def decompose_residual_stream(model, tokens, probe_layer):
    """
    Run model with cache and extract all component outputs at last token.

    Gemma 2 uses sandwich norms: post-attention and post-MLP RMSNorm are
    applied BEFORE adding to the residual stream. TransformerLens hooks
    hook_attn_out/hook_mlp_out capture values BEFORE these norms, so using
    them directly breaks additivity.

    Instead, we compute true additive contributions via residual differences:
      attn_contribution[l] = resid_mid[l] - resid_pre[l]
      mlp_contribution[l]  = resid_post[l] - resid_mid[l]

    These include the post-norms and sum exactly:
      resid_post[L-1] = embed + sum_{l=0}^{L-1} (attn_contrib[l] + mlp_contrib[l])

    For probe_layer=23, we decompose layers 0-22 (inclusive), giving:
      1 embed + 23 attn + 23 mlp = 47 components

    Args:
        model: HookedTransformer instance
        tokens: tokenized input, shape (1, seq_len)
        probe_layer: the layer where probes are trained (1-indexed convention:
                     probe reads resid_pre[probe_layer] = resid_post[probe_layer-1])

    Returns:
        components: dict of component_name -> (d_model,) numpy array
        resid_post: (d_model,) numpy array for verification
    """
    target_layer = probe_layer - 1  # 0-indexed layer whose resid_post we decompose

    # Build names filter: we need embed, resid_mid, and resid_post for each layer
    names = {"hook_embed"}
    for l in range(probe_layer):  # layers 0 to probe_layer-1
        names.add(f"blocks.{l}.hook_resid_mid")
        names.add(f"blocks.{l}.hook_resid_post")

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in names
        )

    last_pos = tokens.shape[1] - 1
    components = {}

    # Embedding = resid_pre[0]
    embed = cache["hook_embed"][0, last_pos, :].float().cpu().numpy()
    components["embed"] = embed

    # For each layer, compute true additive contributions via differences
    # resid_pre[0] = embed
    # resid_pre[l] = resid_post[l-1] for l > 0
    for l in range(probe_layer):
        resid_mid_l = cache[f"blocks.{l}.hook_resid_mid"][0, last_pos, :].float().cpu().numpy()
        resid_post_l = cache[f"blocks.{l}.hook_resid_post"][0, last_pos, :].float().cpu().numpy()

        if l == 0:
            resid_pre_l = embed
        else:
            resid_pre_l = cache[f"blocks.{l-1}.hook_resid_post"][0, last_pos, :].float().cpu().numpy()

        # True additive contributions (includes post-norms)
        components[f"attn_{l}"] = resid_mid_l - resid_pre_l
        components[f"mlp_{l}"] = resid_post_l - resid_mid_l

    # Verification target
    resid_post = (
        cache[f"blocks.{target_layer}.hook_resid_post"][0, last_pos, :].float().cpu().numpy()
    )

    del cache
    torch.cuda.empty_cache()

    return components, resid_post


def compute_attribution_matrix(components, directions):
    """
    Project each component onto probe directions.

    Args:
        components: dict of component_name -> (d_model,) vectors
        directions: (n_principles, d_model) unit vectors

    Returns:
        attributions: dict of component_name -> (n_principles,) projections
    """
    attributions = {}
    for name, vec in components.items():
        attributions[name] = directions @ vec  # (n_principles,)
    return attributions


def verify_decomposition(components, resid_post, directions):
    """
    Check that sum of component projections matches resid_post projection.

    Returns dict with max_raw_error (in activation space) and max_proj_error
    (in projection space). Both should be < 0.01 for a valid decomposition.
    """
    component_sum = sum(components.values())

    component_sum_proj = directions @ component_sum
    resid_proj = directions @ resid_post

    return {
        "max_raw_error": float(np.max(np.abs(component_sum - resid_post))),
        "max_proj_error": float(np.max(np.abs(component_sum_proj - resid_proj))),
        "component_sum_proj": component_sum_proj,
        "resid_proj": resid_proj,
    }


def verify_stored_activations(model, case, stored_activations, probe_layer):
    """
    Check that fresh resid_post matches stored activations for a case.

    This validates that format_prompt() produces the same prompt used
    during the original activation extraction.

    Returns (matches, max_error) where matches is True if error < 0.1.
    """
    case_id = case["case_id"]
    if case_id not in stored_activations:
        return None, None

    stored_cache = stored_activations[case_id]
    stored_act = stored_cache.residual_activations[probe_layer - 1]  # 0-indexed

    prompt = format_prompt(case)
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name == f"blocks.{probe_layer - 1}.hook_resid_post"
        )

    last_pos = tokens.shape[1] - 1
    fresh_act = cache[f"blocks.{probe_layer - 1}.hook_resid_post"][0, last_pos, :].float().cpu().numpy()

    del cache
    torch.cuda.empty_cache()

    max_error = float(np.max(np.abs(fresh_act - stored_act)))
    matches = max_error < 0.1

    return matches, max_error


# === Head-Level Analysis ===

def extract_head_contributions(model, tokens, layer):
    """
    Extract per-head contributions for a specific attention layer.

    Uses hook_z (pre-W_O, shape: batch, pos, n_heads, d_head) and manually
    applies W_O to get each head's contribution in d_model space.

    NOTE: These per-head contributions sum to hook_attn_out (before post-norm).
    Gemma 2's post-attention RMSNorm is applied to the sum, not per-head,
    so individual head projections onto probe directions are approximate
    (the norm rescales but preserves direction, so relative magnitudes
    are meaningful even though absolute values are pre-norm).

    Returns:
        head_contributions: list of n_heads (d_model,) numpy arrays
        attn_out: (d_model,) numpy array for verification (pre-post-norm)
        bias: (d_model,) numpy array (attention output bias, may be zero)
    """
    names = {
        f"blocks.{layer}.attn.hook_z",
        f"blocks.{layer}.hook_attn_out",
    }

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in names
        )

    last_pos = tokens.shape[1] - 1

    # z shape: (batch, pos, n_heads, d_head)
    z = cache[f"blocks.{layer}.attn.hook_z"][0, last_pos, :, :]  # (n_heads, d_head)

    # W_O shape: (n_heads, d_head, d_model)
    W_O = model.blocks[layer].attn.W_O

    n_heads = z.shape[0]
    head_contributions = []
    for h in range(n_heads):
        # z[h] @ W_O[h] -> (d_model,)
        contrib = (z[h] @ W_O[h]).detach().float().cpu().numpy()
        head_contributions.append(contrib)

    attn_out = cache[f"blocks.{layer}.hook_attn_out"][0, last_pos, :].float().cpu().numpy()

    # Get bias (may be zero for models without attention bias)
    b_O = model.blocks[layer].attn.b_O
    if b_O is not None:
        bias = b_O.detach().float().cpu().numpy()
    else:
        bias = np.zeros(model.cfg.d_model)

    del cache
    torch.cuda.empty_cache()

    return head_contributions, attn_out, bias


# === Attention Pattern Analysis ===

def extract_attention_patterns(model, tokens, layer, head_indices):
    """
    Extract attention patterns for specific heads at the last token position.

    Args:
        model: HookedTransformer
        tokens: tokenized input
        layer: which layer
        head_indices: list of head indices to extract

    Returns:
        patterns: dict of head_idx -> (seq_len,) attention weights from last token
        token_strs: list of token strings
    """
    names = {f"blocks.{layer}.attn.hook_pattern"}

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in names
        )

    last_pos = tokens.shape[1] - 1

    # pattern shape: (batch, n_heads, dest_pos, src_pos)
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

    patterns = {}
    for h in head_indices:
        patterns[h] = pattern[0, h, last_pos, :].float().cpu().numpy()

    token_strs = [model.to_string(tokens[0, i:i+1]) for i in range(tokens.shape[1])]

    del cache
    torch.cuda.empty_cache()

    return patterns, token_strs


def categorize_tokens(token_strs):
    """
    Label each token by category for attention pattern analysis.

    Categories: principle:<name>, legal_term, punctuation, whitespace, other
    """
    principle_keywords = {
        "free_expression": [
            "speech", "press", "expression", "first", "amendment", "speak",
            "censor", "publish", "petition", "assembl",
        ],
        "equal_protection": [
            "equal", "protection", "discriminat", "fourteenth", "race",
            "sex", "gender", "classif", "segregat",
        ],
        "due_process": [
            "due", "process", "procedur", "hearing", "notice", "counsel",
            "trial", "miranda", "fair",
        ],
        "federalism": [
            "federal", "commerce", "sovereign", "tenth", "congress",
            "commandeer", "preempt", "enumerat",
        ],
        "privacy_liberty": [
            "privacy", "liberty", "autonomy", "bodily", "intimate",
            "penumbra", "unenumerat",
        ],
    }

    legal_terms = [
        "court", "justice", "opinion", "holding", "precedent", "doctrine",
        "scrutiny", "constitutional", "statute", "clause", "right", "rights",
        "law", "legal", "ruling", "judgment", "plaintiff", "defendant",
        "appeal", "certiorari", "principle",
    ]

    categories = []
    for tok in token_strs:
        tok_lower = tok.strip().lower()

        if not tok_lower:
            categories.append("whitespace")
            continue

        if tok.strip() in {".", ",", ":", ";", "?", "!", "(", ")", "-", "'", '"'}:
            categories.append("punctuation")
            continue

        # Check principle keywords (substring match)
        matched = None
        for principle, keywords in principle_keywords.items():
            for kw in keywords:
                if kw in tok_lower:
                    matched = f"principle:{principle}"
                    break
            if matched:
                break

        if matched:
            categories.append(matched)
        elif any(term in tok_lower for term in legal_terms):
            categories.append("legal_term")
        else:
            categories.append("other")

    return categories


# === Sort Helper ===

def _component_sort_key(name):
    """Sort: embed first, then by layer number, attn before mlp."""
    if name == "embed":
        return (-1, 0)
    parts = name.split("_")
    return (int(parts[1]), 0 if parts[0] == "attn" else 1)


# === Discriminative Metrics ===

def compute_discriminative_metrics(per_case_projections, y_gt, component_names, principle_names):
    """
    For each component and principle, compute discriminative metrics:
    - Pearson r between component's per-case projection and ground-truth weight
    - Cross-case std of projection (raw variation available to probe)
    - Variance of projection (for variance decomposition)

    Args:
        per_case_projections: dict of component_name -> (n_cases, n_principles) array
        y_gt: (n_cases, n_principles) ground-truth principle weights
        component_names: list of component names
        principle_names: list of principle names

    Returns:
        List of dicts sorted by mean |r|, one per component.
    """
    results = []

    for comp_name in component_names:
        projections = per_case_projections[comp_name]  # (n_cases, n_principles)
        correlations = []
        stds = []

        for p_idx in range(len(principle_names)):
            proj_series = projections[:, p_idx]
            gt_series = y_gt[:, p_idx]

            stds.append(float(np.std(proj_series)))

            if np.std(proj_series) < 1e-10 or np.std(gt_series) < 1e-10:
                correlations.append(0.0)
            else:
                r = float(np.corrcoef(proj_series, gt_series)[0, 1])
                correlations.append(r)

        results.append({
            "component": comp_name,
            "correlations": dict(zip(principle_names, correlations)),
            "mean_abs_r": float(np.mean(np.abs(correlations))),
            "stds": dict(zip(principle_names, stds)),
            "mean_std": float(np.mean(stds)),
        })

    results.sort(key=lambda x: x["mean_abs_r"], reverse=True)
    return results


# === Phase Runners ===

def run_phase1(model, cases, directions, probe_layer, output_dir, y_gt, stored_activations=None):
    """
    Phase 1: Layer-Component Attribution with Discriminative Metrics.

    For each case, decompose the residual stream at probe_layer into
    per-component contributions and project onto probe directions.
    Ranks components by Pearson r with ground truth (not mean abs projection).
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Layer-Component Attribution")
    print("=" * 60)

    n_principles = directions.shape[0]
    all_attributions = defaultdict(list)  # component_name -> list of (n_principles,)
    verification_errors = []

    # Optional: verify stored activations match for first case
    if stored_activations is not None and len(cases) > 0:
        print("\n  Verifying stored activation consistency...")
        matches, error = verify_stored_activations(
            model, cases[0], stored_activations, probe_layer
        )
        if matches is not None:
            status = "MATCH" if matches else "MISMATCH"
            print(f"  Stored activation check: {status} (max_error={error:.6f})")
            if not matches:
                print("  WARNING: Fresh activations differ from stored. "
                      "Prompt format may have changed.")

    for i, case in enumerate(cases):
        case_id = case["case_id"]
        prompt = format_prompt(case)
        tokens = model.to_tokens(prompt)

        print(f"  [{i+1}/{len(cases)}] {case.get('case_name', case_id)[:50]}...")

        components, resid_post = decompose_residual_stream(model, tokens, probe_layer)

        verification = verify_decomposition(components, resid_post, directions)
        verification_errors.append(verification["max_raw_error"])

        if verification["max_raw_error"] > 0.1:
            print(f"    WARNING: Large decomposition error: "
                  f"{verification['max_raw_error']:.4f}")

        attributions = compute_attribution_matrix(components, directions)
        for name, proj in attributions.items():
            all_attributions[name].append(proj)

    max_error = max(verification_errors)
    print(f"\n  Verification: max decomposition error = {max_error:.6f}")

    component_names = sorted(all_attributions.keys(), key=_component_sort_key)

    # Build per-case projection arrays
    per_case_projections = {}
    for name in component_names:
        per_case_projections[name] = np.array(all_attributions[name])  # (n_cases, n_principles)

    # Compute discriminative metrics (Pearson r with ground truth)
    discriminative = compute_discriminative_metrics(
        per_case_projections, y_gt, component_names, PRINCIPLE_NAMES
    )

    results = {
        "probe_layer": probe_layer,
        "n_cases": len(cases),
        "n_components": len(component_names),
        "n_principles": n_principles,
        "principle_names": PRINCIPLE_NAMES,
        "max_verification_error": float(max_error),
        "discriminative_attribution": discriminative,
        "components": {},
    }

    for name in component_names:
        projections = per_case_projections[name]
        results["components"][name] = {
            "mean": projections.mean(axis=0).tolist(),
            "std": projections.std(axis=0).tolist(),
            "mean_abs": np.abs(projections).mean(axis=0).tolist(),
            "max_abs": np.abs(projections).max(axis=0).tolist(),
            "per_case": projections.tolist(),
        }

    # Print discriminative ranking
    print("\n  DISCRIMINATIVE ATTRIBUTION (ranked by mean |r| with ground truth)")
    print("  " + "=" * 100)
    print(f"  {'Component':<12} | {'FreeExp':>8} {'EqualProt':>10} {'DueProc':>8} "
          f"{'Federal':>8} {'Privacy':>8} | {'Mean|r|':>8} {'MeanStd':>8}")
    print("  " + "-" * 100)

    for entry in discriminative:
        c = entry["correlations"]
        print(f"  {entry['component']:<12} | "
              f"{c['free_expression']:>+8.3f} {c['equal_protection']:>+10.3f} "
              f"{c['due_process']:>+8.3f} {c['federalism']:>+8.3f} "
              f"{c['privacy_liberty']:>+8.3f} | "
              f"{entry['mean_abs_r']:>8.3f} {entry['mean_std']:>8.2f}")

    # Also print MLP vs Attention summary
    attn_entries = [e for e in discriminative if e["component"].startswith("attn_")]
    mlp_entries = [e for e in discriminative if e["component"].startswith("mlp_")]
    if attn_entries and mlp_entries:
        attn_best = max(attn_entries, key=lambda e: e["mean_abs_r"])
        mlp_best = max(mlp_entries, key=lambda e: e["mean_abs_r"])
        print(f"\n  Best attn component: {attn_best['component']} (mean|r|={attn_best['mean_abs_r']:.3f})")
        print(f"  Best MLP component:  {mlp_best['component']} (mean|r|={mlp_best['mean_abs_r']:.3f})")

    output_path = Path(output_dir) / "phase1_layer_attribution.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved Phase 1 results to {output_path}")

    return results


def run_phase2(model, cases, directions, probe_layer, phase1_results,
               top_k_layers, output_dir, y_gt=None):
    """
    Phase 2: Head-Level Attribution for top attention layers from Phase 1.

    Decomposes each attention layer's output into per-head contributions
    using z @ W_O, then projects onto probe directions. Selects top layers
    by discriminative value (mean |r|) and computes per-head discriminative
    metrics.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Head-Level Attribution")
    print("=" * 60)

    # Select top attention layers by discriminative value if available
    disc_attr = phase1_results.get("discriminative_attribution")
    if disc_attr:
        attn_layer_scores = {}
        for entry in disc_attr:
            name = entry["component"]
            if name.startswith("attn_"):
                layer_idx = int(name.split("_")[1])
                attn_layer_scores[layer_idx] = entry["mean_abs_r"]
        score_label = "mean|r|"
    else:
        # Fallback to mean_abs for legacy phase1 results
        component_data = phase1_results["components"]
        attn_layer_scores = {}
        for name, data in component_data.items():
            if name.startswith("attn_"):
                layer_idx = int(name.split("_")[1])
                attn_layer_scores[layer_idx] = np.mean(data["mean_abs"])
        score_label = "mean|proj|"

    top_attn_layers = sorted(
        attn_layer_scores.keys(),
        key=lambda l: attn_layer_scores[l],
        reverse=True,
    )[:top_k_layers]
    top_attn_layers.sort()  # restore layer order

    print(f"  Top {top_k_layers} attention layers by {score_label}: {top_attn_layers}")
    print(f"  Scores: "
          f"{[f'{l}:{attn_layer_scores[l]:.4f}' for l in top_attn_layers]}")

    n_heads = model.cfg.n_heads
    n_principles = directions.shape[0]

    results = {
        "probe_layer": probe_layer,
        "top_layers": top_attn_layers,
        "n_heads": n_heads,
        "n_cases": len(cases),
        "n_principles": n_principles,
        "principle_names": PRINCIPLE_NAMES,
        "layers": {},
    }

    for layer in top_attn_layers:
        print(f"\n  --- Layer {layer} ({n_heads} heads) ---")

        all_head_projections = defaultdict(list)
        verification_errors = []

        for i, case in enumerate(cases):
            prompt = format_prompt(case)
            tokens = model.to_tokens(prompt)

            print(f"    [{i+1}/{len(cases)}] "
                  f"{case.get('case_name', case['case_id'])[:40]}...", end="")

            head_contributions, attn_out, bias = extract_head_contributions(
                model, tokens, layer
            )

            # Verify: sum of heads + bias ≈ attn_out
            head_sum = sum(head_contributions) + bias
            error = float(np.max(np.abs(head_sum - attn_out)))
            verification_errors.append(error)

            if error > 0.1:
                print(f" WARN:err={error:.4f}", end="")

            for h, contrib in enumerate(head_contributions):
                proj = directions @ contrib
                all_head_projections[h].append(proj)

            print(" ok")

        print(f"    Max head-sum verification error: "
              f"{max(verification_errors):.6f}")

        layer_results = {
            "max_verification_error": float(max(verification_errors)),
            "heads": {},
        }

        # Build per-head per-case projection arrays for discriminative metrics
        head_per_case = {}
        for h in range(n_heads):
            projections = np.array(all_head_projections[h])  # (n_cases, n_principles)
            head_per_case[f"H{h}"] = projections
            layer_results["heads"][str(h)] = {
                "mean": projections.mean(axis=0).tolist(),
                "std": projections.std(axis=0).tolist(),
                "mean_abs": np.abs(projections).mean(axis=0).tolist(),
                "per_case": projections.tolist(),
            }

        # Compute discriminative metrics for heads in this layer
        if y_gt is not None:
            head_names = [f"H{h}" for h in range(n_heads)]
            head_disc = compute_discriminative_metrics(
                head_per_case, y_gt, head_names, PRINCIPLE_NAMES
            )
            layer_results["discriminative_attribution"] = head_disc

            # Print top heads by discriminative value
            print(f"\n    HEAD DISCRIMINATIVE RANKING (layer {layer}):")
            for entry in head_disc[:10]:
                c = entry["correlations"]
                print(f"      {entry['component']:<5} | "
                      f"FE={c['free_expression']:+.3f} EP={c['equal_protection']:+.3f} "
                      f"DP={c['due_process']:+.3f} FD={c['federalism']:+.3f} "
                      f"PL={c['privacy_liberty']:+.3f} | "
                      f"mean|r|={entry['mean_abs_r']:.3f}")
        else:
            # Fallback: print by mean_abs
            for p_idx, principle in enumerate(PRINCIPLE_NAMES):
                ranked = sorted(
                    range(n_heads),
                    key=lambda h: np.abs(
                        np.array(all_head_projections[h])
                    )[:, p_idx].mean(),
                    reverse=True,
                )
                top3_strs = []
                for h in ranked[:3]:
                    score = np.abs(np.array(all_head_projections[h]))[:, p_idx].mean()
                    top3_strs.append(f"H{h}({score:.4f})")
                print(f"    {principle}: top heads = {', '.join(top3_strs)}")

        results["layers"][str(layer)] = layer_results

    output_path = Path(output_dir) / "phase2_head_attribution.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved Phase 2 results to {output_path}")

    return results


def run_phase3(model, cases, directions, phase2_results, top_k_heads, output_dir):
    """
    Phase 3: Attention Pattern Analysis for specialist heads.

    For heads with the strongest alignment to probe directions, extract
    attention weights and categorize what tokens they attend to.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Attention Pattern Analysis")
    print("=" * 60)

    # Identify specialist heads — prefer discriminative ranking if available
    specialist_candidates = []
    for layer_str, layer_data in phase2_results["layers"].items():
        layer = int(layer_str)
        head_disc = layer_data.get("discriminative_attribution")
        if head_disc:
            for entry in head_disc:
                head = int(entry["component"].replace("H", ""))
                corrs = list(entry["correlations"].values())
                best_p_idx = int(np.argmax(np.abs(corrs)))
                specialist_candidates.append(
                    (layer, head, PRINCIPLE_NAMES[best_p_idx], entry["mean_abs_r"])
                )
        else:
            for head_str, head_data in layer_data["heads"].items():
                head = int(head_str)
                mean_abs = head_data["mean_abs"]
                best_p_idx = int(np.argmax(mean_abs))
                specialist_candidates.append(
                    (layer, head, PRINCIPLE_NAMES[best_p_idx], mean_abs[best_p_idx])
                )

    specialist_candidates.sort(key=lambda x: x[3], reverse=True)

    # Deduplicate by (layer, head), keep top_k
    seen = set()
    specialists = []
    for layer, head, principle, score in specialist_candidates:
        if (layer, head) not in seen and len(specialists) < top_k_heads:
            specialists.append((layer, head, principle, score))
            seen.add((layer, head))

    print(f"  Top {len(specialists)} specialist heads:")
    for layer, head, principle, score in specialists:
        print(f"    L{layer}H{head}: {principle} (mean|proj|={score:.4f})")

    # Group by layer for efficient caching
    heads_by_layer = defaultdict(list)
    for layer, head, principle, score in specialists:
        heads_by_layer[layer].append((head, principle, score))

    results = {
        "specialist_heads": [
            {"layer": l, "head": h, "top_principle": p, "score": float(s)}
            for l, h, p, s in specialists
        ],
        "n_cases": len(cases),
        "attention_analysis": {},
    }

    for layer in sorted(heads_by_layer.keys()):
        heads_info = heads_by_layer[layer]
        head_indices = [h for h, _, _ in heads_info]
        print(f"\n  --- Layer {layer}, heads {head_indices} ---")

        layer_key = str(layer)
        results["attention_analysis"][layer_key] = {}

        for h in head_indices:
            results["attention_analysis"][layer_key][str(h)] = {"cases": []}

        for i, case in enumerate(cases):
            case_id = case["case_id"]
            prompt = format_prompt(case)
            tokens = model.to_tokens(prompt)

            print(f"    [{i+1}/{len(cases)}] "
                  f"{case.get('case_name', case_id)[:40]}...")

            patterns, token_strs = extract_attention_patterns(
                model, tokens, layer, head_indices
            )
            categories = categorize_tokens(token_strs)

            for h in head_indices:
                attn = patterns[h]

                # Aggregate attention by category
                category_attn = defaultdict(float)
                for tok_idx, cat in enumerate(categories):
                    category_attn[cat] += float(attn[tok_idx])

                # Top-5 attended tokens
                top_indices = np.argsort(attn)[::-1][:5]
                top_tokens = [
                    {
                        "position": int(idx),
                        "token": token_strs[idx],
                        "attention": float(attn[idx]),
                        "category": categories[idx],
                    }
                    for idx in top_indices
                ]

                results["attention_analysis"][layer_key][str(h)]["cases"].append({
                    "case_id": case_id,
                    "category_attention": dict(category_attn),
                    "top_tokens": top_tokens,
                    "seq_len": len(token_strs),
                })

    output_path = Path(output_dir) / "phase3_attention_patterns.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved Phase 3 results to {output_path}")

    return results


# === Summary Report ===

def generate_summary(phase1_results, phase2_results, phase3_results, output_dir):
    """Generate markdown report combining all phases with discriminative metrics."""
    lines = [
        "# Residual Stream Decomposition — Discriminative Attribution",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Probe Layer**: {phase1_results['probe_layer']}",
        f"**Cases**: {phase1_results['n_cases']}",
        f"**Decomposition Verification Error**: "
        f"{phase1_results['max_verification_error']:.6f}",
        "",
        "Metrics: Pearson r between each component's per-case projection onto",
        "probe directions and the ground-truth principle weight. This captures",
        "what the probe can actually use (between-case variation), unlike mean",
        "absolute projection which is dominated by constant offsets.",
        "",
    ]

    disc_attr = phase1_results.get("discriminative_attribution", [])
    component_data = phase1_results["components"]

    # --- Phase 1: Discriminative Attribution Table ---
    lines.append("## Phase 1: Attn/MLP Discriminative Attribution")
    lines.append("")
    lines.append("Which attn/MLP components write the most case-discriminative signal?")
    lines.append("")

    lines.append(
        "| Rank | Component | FreeExp r | EqualProt r | DueProc r "
        "| Federal r | Privacy r | Mean |r| | Mean Std |"
    )
    lines.append(
        "|-----:|-----------|----------:|------------:|----------:"
        "|----------:|----------:|---------:|---------:|"
    )

    for rank, entry in enumerate(disc_attr):
        c = entry["correlations"]
        lines.append(
            f"| {rank+1} | {entry['component']} "
            f"| {c['free_expression']:+.3f} "
            f"| {c['equal_protection']:+.3f} "
            f"| {c['due_process']:+.3f} "
            f"| {c['federalism']:+.3f} "
            f"| {c['privacy_liberty']:+.3f} "
            f"| {entry['mean_abs_r']:.3f} "
            f"| {entry['mean_std']:.2f} |"
        )
    lines.append("")

    # MLP vs Attention discriminative summary
    attn_entries = [e for e in disc_attr if e["component"].startswith("attn_")]
    mlp_entries = [e for e in disc_attr if e["component"].startswith("mlp_")]
    if attn_entries and mlp_entries:
        lines.append("### MLP vs Attention (Discriminative)")
        lines.append("")
        attn_mean_r = np.mean([e["mean_abs_r"] for e in attn_entries])
        mlp_mean_r = np.mean([e["mean_abs_r"] for e in mlp_entries])
        attn_best = max(attn_entries, key=lambda e: e["mean_abs_r"])
        mlp_best = max(mlp_entries, key=lambda e: e["mean_abs_r"])
        lines.append(f"- Attention mean |r| across all layers: {attn_mean_r:.3f}")
        lines.append(f"- MLP mean |r| across all layers: {mlp_mean_r:.3f}")
        lines.append(
            f"- Best attn: {attn_best['component']} (mean|r|={attn_best['mean_abs_r']:.3f})"
        )
        lines.append(
            f"- Best MLP: {mlp_best['component']} (mean|r|={mlp_best['mean_abs_r']:.3f})"
        )
        lines.append("")

    # --- Phase 2: Head-Level Discriminative Attribution ---
    if phase2_results:
        lines.append("## Phase 2: Head-Level Discriminative Attribution")
        lines.append("")
        lines.append("Which attention heads write the most case-discriminative signal?")
        lines.append("")

        for layer_str in sorted(phase2_results["layers"].keys(), key=int):
            layer_data = phase2_results["layers"][layer_str]
            head_disc = layer_data.get("discriminative_attribution", [])

            lines.append(f"### Layer {layer_str}")
            lines.append("")

            if head_disc:
                lines.append(
                    "| Rank | Head | FreeExp r | EqualProt r | DueProc r "
                    "| Federal r | Privacy r | Mean |r| |"
                )
                lines.append(
                    "|-----:|-----:|----------:|------------:|----------:"
                    "|----------:|----------:|---------:|"
                )
                for rank, entry in enumerate(head_disc[:10]):
                    c = entry["correlations"]
                    lines.append(
                        f"| {rank+1} | {entry['component']} "
                        f"| {c['free_expression']:+.3f} "
                        f"| {c['equal_protection']:+.3f} "
                        f"| {c['due_process']:+.3f} "
                        f"| {c['federalism']:+.3f} "
                        f"| {c['privacy_liberty']:+.3f} "
                        f"| {entry['mean_abs_r']:.3f} |"
                    )
            else:
                # Fallback for legacy data
                for p_idx, principle in enumerate(PRINCIPLE_NAMES):
                    head_scores = []
                    for head_str, head_data in layer_data["heads"].items():
                        head_scores.append((
                            int(head_str),
                            head_data["mean_abs"][p_idx],
                        ))
                    head_scores.sort(key=lambda x: x[1], reverse=True)
                    parts = [f"H{h}({s:.4f})" for h, s in head_scores[:5]]
                    lines.append(f"**{principle}**: {', '.join(parts)}")
            lines.append("")

        # Cross-layer specialist heads summary
        all_head_entries = []
        for layer_str, layer_data in phase2_results["layers"].items():
            head_disc = layer_data.get("discriminative_attribution", [])
            for entry in head_disc:
                head = int(entry["component"].replace("H", ""))
                corrs = list(entry["correlations"].values())
                best_p_idx = int(np.argmax(np.abs(corrs)))
                all_head_entries.append((
                    int(layer_str), head,
                    PRINCIPLE_NAMES[best_p_idx],
                    entry["mean_abs_r"],
                ))
        if all_head_entries:
            all_head_entries.sort(key=lambda x: x[3], reverse=True)
            lines.append("### Top Specialist Heads (Cross-Layer)")
            lines.append("")
            lines.append("| Rank | Layer | Head | Top Principle | Mean |r| |")
            lines.append("|-----:|------:|-----:|--------------|--------:|")
            seen = set()
            rank = 0
            for layer, head, principle, score in all_head_entries:
                if (layer, head) not in seen and rank < 20:
                    rank += 1
                    lines.append(
                        f"| {rank} | {layer} | {head} | {principle} | {score:.3f} |"
                    )
                    seen.add((layer, head))
            lines.append("")

    # --- Phase 3 ---
    if phase3_results:
        lines.append("## Phase 3: Attention Pattern Analysis")
        lines.append("")
        lines.append("What do specialist heads attend to?")
        lines.append("")

        for spec in phase3_results["specialist_heads"][:10]:
            layer = spec["layer"]
            head = spec["head"]
            principle = spec["top_principle"]

            layer_data = phase3_results["attention_analysis"].get(str(layer), {})
            head_data = layer_data.get(str(head), {})
            if not head_data:
                continue

            lines.append(
                f"### L{layer}H{head} (top principle: {principle})"
            )
            lines.append("")

            category_totals = defaultdict(float)
            n_cases = 0
            for case_data in head_data.get("cases", []):
                for cat, attn in case_data["category_attention"].items():
                    category_totals[cat] += attn
                n_cases += 1

            if n_cases > 0:
                lines.append("| Category | Mean Attention |")
                lines.append("|----------|-------------:|")
                sorted_cats = sorted(
                    category_totals.items(), key=lambda x: x[1], reverse=True
                )
                for cat, total in sorted_cats:
                    lines.append(f"| {cat} | {total/n_cases:.4f} |")
                lines.append("")

    # --- Interpretation ---
    lines.append("## Interpretation")
    lines.append("")

    if disc_attr:
        top3 = disc_attr[:3]
        lines.append(
            f"Top discriminative components: "
            f"{top3[0]['component']} (mean|r|={top3[0]['mean_abs_r']:.3f}), "
            f"{top3[1]['component']} ({top3[1]['mean_abs_r']:.3f}), "
            f"{top3[2]['component']} ({top3[2]['mean_abs_r']:.3f})"
        )
        lines.append("")

        # Check if attn or MLP dominates discriminatively
        top10_attn = sum(1 for e in disc_attr[:10] if e["component"].startswith("attn_"))
        top10_mlp = sum(1 for e in disc_attr[:10] if e["component"].startswith("mlp_"))
        lines.append(
            f"In top-10 discriminative components: "
            f"{top10_attn} attention, {top10_mlp} MLP, "
            f"{10 - top10_attn - top10_mlp} other"
        )
    lines.append("")

    summary_text = "\n".join(lines)
    output_path = Path(output_dir) / "decomposition_summary.md"
    with open(output_path, "w") as f:
        f.write(summary_text)
    print(f"\nSaved summary to {output_path}")


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="Residual Stream Decomposition for Constitutional Principle Geometry"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--probe-layer", type=int, default=23,
        help="Layer where probes are trained (default: 23)",
    )
    parser.add_argument(
        "--phases", type=str, default="1,2,3",
        help="Comma-separated phases to run (default: 1,2,3)",
    )
    parser.add_argument(
        "--top-k-layers", type=int, default=5,
        help="Top attention layers for Phase 2 head decomposition (default: 5)",
    )
    parser.add_argument(
        "--top-k-heads", type=int, default=10,
        help="Top specialist heads for Phase 3 attention analysis (default: 10)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 5 cases, Phase 1 only",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/gemma2_27b/decomposition/)",
    )

    args = parser.parse_args()

    if args.quick:
        phases = [1]
        max_cases = 5
        print("QUICK MODE: 5 cases, Phase 1 only")
    else:
        phases = [int(p) for p in args.phases.split(",")]
        max_cases = None

    decomp_dir = (
        Path(args.output_dir) if args.output_dir
        else RESULTS_DIR / "decomposition"
    )
    decomp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {decomp_dir}")
    print(f"Probe layer: {args.probe_layer}")
    print(f"Phases: {phases}")
    print(f"Device: {args.device}")

    # === Load probe directions ===
    print("\n" + "=" * 60)
    print("Loading probe directions")
    print("=" * 60)

    act_dir = DATA_DIR / "activations" / "aligned"
    annotations_path = ROOT_DIR / "data" / "annotations.json"

    print(f"Loading activations from {act_dir}...")
    activations = load_activation_dataset(str(act_dir))
    print(f"Loading annotations from {annotations_path}...")
    annotations = load_annotations(str(annotations_path))

    layer_data = extract_directions_with_scaler_correction(
        activations, annotations, [args.probe_layer]
    )

    if args.probe_layer not in layer_data:
        print(f"ERROR: Could not extract directions for layer {args.probe_layer}")
        sys.exit(1)

    directions = layer_data[args.probe_layer]["directions"]  # (5, d_model)
    r2 = layer_data[args.probe_layer]["r2_score"]
    print(f"  Directions shape: {directions.shape}")
    print(f"  Probe R² = {r2:.4f}")

    # Keep activations for stored-activation verification, then free
    stored_activations = activations
    del activations
    gc.collect()

    # === Select cases and build ground truth ===
    ann_lookup = {a.case_id: a for a in annotations}
    cases = [c for c in ALL_CASES if c["case_id"] in ann_lookup]
    if max_cases:
        cases = cases[:max_cases]
    print(f"\nUsing {len(cases)} cases")

    # Ground-truth principle weights aligned with case order
    y_gt = np.array([ann_lookup[c["case_id"]].to_vector() for c in cases])
    print(f"Ground truth shape: {y_gt.shape}")

    # === Load model ===
    print("\n" + "=" * 60)
    print("Loading model")
    print("=" * 60)

    patcher = ActivationPatcher(ALIGNED_MODEL, device=args.device)
    model = patcher.model

    # Free stored activations before running phases (save memory)
    del stored_activations
    gc.collect()

    # === Run phases ===
    start_time = time.time()

    phase1_results = None
    phase2_results = None
    phase3_results = None

    if 1 in phases:
        phase1_results = run_phase1(
            model, cases, directions, args.probe_layer, decomp_dir,
            y_gt=y_gt,
        )
    else:
        p1_path = decomp_dir / "phase1_layer_attribution.json"
        if p1_path.exists():
            with open(p1_path) as f:
                phase1_results = json.load(f)
            print(f"Loaded existing Phase 1 results from {p1_path}")

    if 2 in phases:
        if phase1_results is None:
            print("ERROR: Phase 2 requires Phase 1 results")
            sys.exit(1)
        phase2_results = run_phase2(
            model, cases, directions, args.probe_layer,
            phase1_results, args.top_k_layers, decomp_dir,
            y_gt=y_gt,
        )
    else:
        p2_path = decomp_dir / "phase2_head_attribution.json"
        if p2_path.exists():
            with open(p2_path) as f:
                phase2_results = json.load(f)
            print(f"Loaded existing Phase 2 results from {p2_path}")

    if 3 in phases:
        if phase2_results is None:
            print("ERROR: Phase 3 requires Phase 2 results")
            sys.exit(1)
        phase3_results = run_phase3(
            model, cases, directions, phase2_results, args.top_k_heads, decomp_dir,
        )
    else:
        p3_path = decomp_dir / "phase3_attention_patterns.json"
        if p3_path.exists():
            with open(p3_path) as f:
                phase3_results = json.load(f)

    # Generate summary
    if phase1_results:
        generate_summary(
            phase1_results, phase2_results, phase3_results, decomp_dir
        )

    # Cleanup
    del patcher, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nDone. Total time: {elapsed/60:.1f} min")
    print(f"Results in: {decomp_dir}/")


if __name__ == "__main__":
    main()
