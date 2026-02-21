#!/usr/bin/env python3
"""
Residual Stream Decomposition for Constitutional Principle Geometry

Decomposes the residual stream at the probe layer into per-component
contributions (embedding, attention outputs, MLP outputs) and projects
each onto probe directions. This identifies which model components
actually write the information that linear probes read.

Three phases:
1. Layer-Component Attribution: Which layers/components contribute most
   to each principle direction? (47 components for layer 23)
2. Head-Level Attribution: For top attention layers, which heads write
   the strongest signal? (per-head W_O projection analysis)
3. Attention Pattern Analysis: What do specialist heads attend to?
   (token-level attention distribution)

The residual stream decomposition is exact because Gemma 2 uses pre-norm:
  resid_post[L] = embed + sum_{l=0}^{L} (attn_out[l] + mlp_out[l])

Usage:
    # Quick test (5 cases, Phase 1 only)
    python causal_validation/scripts/residual_decomposition.py --quick --device cuda

    # Full run (all phases)
    python causal_validation/scripts/residual_decomposition.py --device cuda

    # Select phases
    python causal_validation/scripts/residual_decomposition.py --device cuda \
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
OUTPUT_DIR = ROOT_DIR / "experiment_output_gemma2_27b"


# === Core Decomposition Functions ===

def decompose_residual_stream(model, tokens, probe_layer):
    """
    Run model with cache and extract all component outputs at last token.

    The residual stream at layer L-1 (0-indexed) decomposes as:
      resid_post[L-1] = embed + sum_{l=0}^{L-1} (attn_out[l] + mlp_out[l])

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

    # Build names filter for efficient caching
    names = {"hook_embed"}
    for l in range(probe_layer):  # layers 0 to probe_layer-1
        names.add(f"blocks.{l}.hook_attn_out")
        names.add(f"blocks.{l}.hook_mlp_out")
    names.add(f"blocks.{target_layer}.hook_resid_post")

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in names
        )

    last_pos = tokens.shape[1] - 1
    components = {}

    # Embedding
    components["embed"] = cache["hook_embed"][0, last_pos, :].float().cpu().numpy()

    # Attention and MLP outputs for each layer
    for l in range(probe_layer):
        components[f"attn_{l}"] = (
            cache[f"blocks.{l}.hook_attn_out"][0, last_pos, :].float().cpu().numpy()
        )
        components[f"mlp_{l}"] = (
            cache[f"blocks.{l}.hook_mlp_out"][0, last_pos, :].float().cpu().numpy()
        )

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

    Returns:
        head_contributions: list of n_heads (d_model,) numpy arrays
        attn_out: (d_model,) numpy array for verification
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
        contrib = (z[h] @ W_O[h]).float().cpu().numpy()
        head_contributions.append(contrib)

    attn_out = cache[f"blocks.{layer}.hook_attn_out"][0, last_pos, :].float().cpu().numpy()

    # Get bias (may be zero for models without attention bias)
    b_O = model.blocks[layer].attn.b_O
    if b_O is not None:
        bias = b_O.float().cpu().numpy()
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


# === Phase Runners ===

def run_phase1(model, cases, directions, probe_layer, output_dir, stored_activations=None):
    """
    Phase 1: Layer-Component Attribution.

    For each case, decompose the residual stream at probe_layer into
    per-component contributions and project onto probe directions.
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

    results = {
        "probe_layer": probe_layer,
        "n_cases": len(cases),
        "n_components": len(component_names),
        "n_principles": n_principles,
        "principle_names": PRINCIPLE_NAMES,
        "max_verification_error": float(max_error),
        "components": {},
    }

    for name in component_names:
        projections = np.array(all_attributions[name])  # (n_cases, n_principles)
        results["components"][name] = {
            "mean": projections.mean(axis=0).tolist(),
            "std": projections.std(axis=0).tolist(),
            "mean_abs": np.abs(projections).mean(axis=0).tolist(),
            "max_abs": np.abs(projections).max(axis=0).tolist(),
        }

    # Print top components for each principle
    for p_idx, principle in enumerate(PRINCIPLE_NAMES):
        print(f"\n  --- {principle} ---")
        ranked = sorted(
            component_names,
            key=lambda n: np.abs(np.array(all_attributions[n]))[:, p_idx].mean(),
            reverse=True,
        )
        for rank, name in enumerate(ranked[:10]):
            mean_abs = np.abs(np.array(all_attributions[name]))[:, p_idx].mean()
            mean_signed = np.array(all_attributions[name])[:, p_idx].mean()
            print(f"    {rank+1}. {name:15s}  mean|proj|={mean_abs:.4f}  "
                  f"mean_proj={mean_signed:+.4f}")

    output_path = Path(output_dir) / "phase1_layer_attribution.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved Phase 1 results to {output_path}")

    return results


def run_phase2(model, cases, directions, probe_layer, phase1_results,
               top_k_layers, output_dir):
    """
    Phase 2: Head-Level Attribution for top attention layers from Phase 1.

    Decomposes each attention layer's output into per-head contributions
    using z @ W_O, then projects onto probe directions.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Head-Level Attribution")
    print("=" * 60)

    component_data = phase1_results["components"]

    # Score each attention layer by mean absolute projection across principles
    attn_layer_scores = {}
    for name, data in component_data.items():
        if name.startswith("attn_"):
            layer_idx = int(name.split("_")[1])
            attn_layer_scores[layer_idx] = np.mean(data["mean_abs"])

    top_attn_layers = sorted(
        attn_layer_scores.keys(),
        key=lambda l: attn_layer_scores[l],
        reverse=True,
    )[:top_k_layers]
    top_attn_layers.sort()  # restore layer order

    print(f"  Top {top_k_layers} attention layers: {top_attn_layers}")
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

        for h in range(n_heads):
            projections = np.array(all_head_projections[h])
            layer_results["heads"][str(h)] = {
                "mean": projections.mean(axis=0).tolist(),
                "std": projections.std(axis=0).tolist(),
                "mean_abs": np.abs(projections).mean(axis=0).tolist(),
            }

        results["layers"][str(layer)] = layer_results

        # Print top heads per principle
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

    # Identify specialist heads across all layers and principles
    specialist_candidates = []
    for layer_str, layer_data in phase2_results["layers"].items():
        layer = int(layer_str)
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
    """Generate markdown report combining all phases."""
    lines = [
        "# Residual Stream Decomposition Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Probe Layer**: {phase1_results['probe_layer']}",
        f"**Cases**: {phase1_results['n_cases']}",
        f"**Decomposition Verification Error**: "
        f"{phase1_results['max_verification_error']:.6f}",
        "",
    ]

    component_data = phase1_results["components"]

    # --- Phase 1 ---
    lines.append("## Phase 1: Layer-Component Attribution")
    lines.append("")
    lines.append("Which components write the most signal in each probe direction?")
    lines.append("")

    for p_idx, principle in enumerate(PRINCIPLE_NAMES):
        lines.append(f"### {principle.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Rank | Component | Mean |Proj| | Mean Proj | Std |")
        lines.append("|-----:|-----------|--------:|----------:|----:|")

        ranked = sorted(
            component_data.items(),
            key=lambda x: x[1]["mean_abs"][p_idx],
            reverse=True,
        )
        for rank, (name, data) in enumerate(ranked[:15]):
            mean_abs = data["mean_abs"][p_idx]
            mean_signed = data["mean"][p_idx]
            std = data["std"][p_idx]
            lines.append(
                f"| {rank+1} | {name} | {mean_abs:.4f} | "
                f"{mean_signed:+.4f} | {std:.4f} |"
            )
        lines.append("")

    # Concentration analysis
    lines.append("### Attribution Concentration")
    lines.append("")
    lines.append("How concentrated is the signal? "
                 "(Top-5 components as % of total |projection|)")
    lines.append("")
    lines.append("| Principle | Top-5 % | Top-10 % | Interpretation |")
    lines.append("|-----------|--------:|---------:|---------------|")

    for p_idx, principle in enumerate(PRINCIPLE_NAMES):
        all_mean_abs = [
            (name, data["mean_abs"][p_idx]) for name, data in component_data.items()
        ]
        all_mean_abs.sort(key=lambda x: x[1], reverse=True)
        total = sum(x[1] for x in all_mean_abs)
        top5 = sum(x[1] for x in all_mean_abs[:5])
        top10 = sum(x[1] for x in all_mean_abs[:10])

        pct5 = top5 / total * 100 if total > 0 else 0
        pct10 = top10 / total * 100 if total > 0 else 0

        if pct5 > 50:
            interp = "concentrated"
        elif pct5 > 30:
            interp = "moderately concentrated"
        else:
            interp = "diffuse"

        lines.append(f"| {principle} | {pct5:.1f}% | {pct10:.1f}% | {interp} |")

    lines.append("")

    # MLP vs Attention breakdown
    lines.append("### MLP vs Attention Contribution")
    lines.append("")
    lines.append(
        "| Principle | Attn Total |Proj| | MLP Total |Proj| | Embed |Proj| |"
    )
    lines.append("|-----------|------------:|------------:|----------:|")

    for p_idx, principle in enumerate(PRINCIPLE_NAMES):
        attn_total = sum(
            data["mean_abs"][p_idx] for name, data in component_data.items()
            if name.startswith("attn_")
        )
        mlp_total = sum(
            data["mean_abs"][p_idx] for name, data in component_data.items()
            if name.startswith("mlp_")
        )
        embed_val = component_data.get("embed", {}).get("mean_abs", [0]*5)[p_idx]
        lines.append(
            f"| {principle} | {attn_total:.4f} | {mlp_total:.4f} | "
            f"{embed_val:.4f} |"
        )

    lines.append("")

    # --- Phase 2 ---
    if phase2_results:
        lines.append("## Phase 2: Head-Level Attribution")
        lines.append("")
        lines.append("Which attention heads write the strongest signal?")
        lines.append("")

        for layer_str in sorted(phase2_results["layers"].keys(), key=int):
            layer_data = phase2_results["layers"][layer_str]
            lines.append(f"### Layer {layer_str}")
            lines.append("")

            for p_idx, principle in enumerate(PRINCIPLE_NAMES):
                head_scores = []
                for head_str, head_data in layer_data["heads"].items():
                    head_scores.append((
                        int(head_str),
                        head_data["mean_abs"][p_idx],
                        head_data["mean"][p_idx],
                    ))
                head_scores.sort(key=lambda x: x[1], reverse=True)

                parts = [
                    f"H{h}({abs_s:.4f}, {sign_s:+.4f})"
                    for h, abs_s, sign_s in head_scores[:5]
                ]
                lines.append(f"**{principle}**: {', '.join(parts)}")
            lines.append("")

        # Specialist heads summary
        lines.append("### Specialist Heads Summary")
        lines.append("")
        lines.append("| Layer | Head | Top Principle | Mean |Proj| |")
        lines.append("|------:|-----:|--------------|--------:|")

        all_heads = []
        for layer_str, layer_data in phase2_results["layers"].items():
            for head_str, head_data in layer_data["heads"].items():
                best_idx = int(np.argmax(head_data["mean_abs"]))
                all_heads.append((
                    int(layer_str), int(head_str),
                    PRINCIPLE_NAMES[best_idx],
                    head_data["mean_abs"][best_idx],
                ))
        all_heads.sort(key=lambda x: x[3], reverse=True)

        seen = set()
        for layer, head, principle, score in all_heads[:20]:
            if (layer, head) not in seen:
                lines.append(f"| {layer} | {head} | {principle} | {score:.4f} |")
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

    for p_idx, principle in enumerate(PRINCIPLE_NAMES):
        all_mean_abs = [
            (name, component_data[name]["mean_abs"][p_idx])
            for name in component_data
        ]
        all_mean_abs.sort(key=lambda x: x[1], reverse=True)
        total = sum(x[1] for x in all_mean_abs)
        top5_pct = (
            sum(x[1] for x in all_mean_abs[:5]) / total * 100 if total > 0 else 0
        )

        top_component = all_mean_abs[0][0]
        top_score = all_mean_abs[0][1]

        if top5_pct > 50:
            lines.append(
                f"- **{principle}**: Concentrated signal. "
                f"Top component: {top_component} ({top_score:.4f}). "
                f"Top-5 = {top5_pct:.0f}% of total. "
                f"Suggests localized circuit."
            )
        else:
            lines.append(
                f"- **{principle}**: Diffuse signal. "
                f"Top component: {top_component} ({top_score:.4f}). "
                f"Top-5 = {top5_pct:.0f}% of total. "
                f"Suggests distributed representation."
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
        help="Output directory (default: experiment_output_gemma2_27b/decomposition/)",
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
        else OUTPUT_DIR / "decomposition"
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

    act_dir = OUTPUT_DIR / "activations" / "aligned"
    annotations_path = OUTPUT_DIR / "annotations.json"

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

    # === Select cases ===
    annotation_ids = {a.case_id for a in annotations}
    cases = [c for c in ALL_CASES if c["case_id"] in annotation_ids]
    if max_cases:
        cases = cases[:max_cases]
    print(f"\nUsing {len(cases)} cases")

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
            model, cases, directions, args.probe_layer, decomp_dir
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
