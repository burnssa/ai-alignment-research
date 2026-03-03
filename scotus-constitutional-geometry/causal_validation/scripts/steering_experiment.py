#!/usr/bin/env python3
"""
Steering Vector Experiment for Constitutional Principle Geometry

Tests whether principle directions extracted from linear probes can causally
steer which constitutional principles the model invokes in its response.

The key idea: if we add the "free expression" direction to activations during
inference on a case where free expression is irrelevant, does the model start
ranking free expression higher?

Methodology:
- Extract principle directions from trained probes (with scaler correction)
- For each principle, select cases where it has low ground-truth weight
- Steer the aligned model by adding the principle direction at varying alphas
- Measure whether the steered principle's rank shifts monotonically with alpha

Usage:
    # Quick test (~30 trials, ~5 min on A100)
    python causal_validation/scripts/steering_experiment.py --quick --device cuda

    # Full experiment (~675 trials, ~1-2 hrs on A100)
    python causal_validation/scripts/steering_experiment.py --device cuda

    # Custom layers and alphas
    python causal_validation/scripts/steering_experiment.py \
        --layers 20,23,26 --alphas -3,-1,0,1,3 --device cuda

    # Norm-relative alphas (recommended): alpha=0.5 means perturbation
    # L2 norm = 50% of mean residual stream norm at that layer
    python causal_validation/scripts/steering_experiment.py \
        --layers 23 --alphas=-0.5,-0.1,0,0.1,0.5,1.0 \
        --norm-relative --device cuda
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
from dataclasses import dataclass, field, asdict
from typing import Optional

# Add paths for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent  # scotus-constitutional-geometry root
sys.path.insert(0, str(SCRIPT_DIR))  # for causal_validation.py
sys.path.insert(0, str(ROOT_DIR))    # for cases.py, extract_activations.py, etc.

from sklearn.preprocessing import StandardScaler

from causal_validation import ActivationPatcher
from extract_activations import load_activation_dataset
from train_probes import LinearProbeTrainer
from annotate_principles import load_annotations
from generate_behavioral_responses import PROMPT_TEMPLATE, PRINCIPLES, parse_ranking
from cases import ALL_CASES

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

PRINCIPLE_NAMES = LinearProbeTrainer.PRINCIPLE_NAMES  # underscore format
ALIGNED_MODEL = "google/gemma-2-27b-it"
OUTPUT_DIR = ROOT_DIR / "experiment_output_gemma2_27b"


# === Data Classes ===

@dataclass
class SteeringTrial:
    """Result from a single steering trial."""
    case_id: str
    case_name: str
    layer: int
    steered_principle: str
    alpha: float
    ground_truth_weight: float  # GT weight of steered principle for this case
    response: str
    parsed_rankings: list  # list of dicts from parse_ranking()
    steered_principle_rank: Optional[int]  # 1-indexed rank, None if not found
    top_principle: Optional[str]
    effective_scale: Optional[float] = None  # actual scale applied (alpha * norm if norm-relative)


@dataclass
class SteeringResults:
    """Aggregated steering experiment results."""
    model: str
    layers: list
    alphas: list
    n_trials: int = 0
    timestamp: str = ""
    norm_relative: bool = False
    resid_norms: dict = field(default_factory=dict)  # layer -> mean ||resid||
    trials: list = field(default_factory=list)


# === Direction Extraction with Scaler Correction ===

def extract_directions_with_scaler_correction(
    activations: dict,
    annotations: list,
    layers: list[int],
    raw_directions: bool = False,
) -> dict[int, dict]:
    """
    Extract principle directions from linear probes with scaler correction.

    train_probe() internally fits a StandardScaler then learns weights in
    scaled space. To get the equivalent direction in native activation space:
    native_direction = weights / scale. We fit the scaler once here, train
    the ridge model on the same scaled data, and use the same scaler for
    the correction — avoiding the redundant double-fit that would occur
    if we called train_probe() (which fits its own internal scaler).

    After correction, directions are normalized to unit vectors so alpha
    directly controls the L2 norm of the perturbation.

    Args:
        activations: case_id -> ActivationCache
        annotations: list of PrincipleAnnotation
        layers: which layers to extract directions for
        raw_directions: if True, skip scaler correction (use probe weights directly)

    Returns:
        dict mapping layer -> {
            "directions": ndarray (n_principles, d_model) unit vectors,
            "r2_score": float,
            "principle_r2": dict,
            "scaler_applied": bool,
        }
    """
    from sklearn.linear_model import RidgeCV, Ridge
    from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
    import warnings

    SEED = 42
    np.random.seed(SEED)

    trainer = LinearProbeTrainer(regularization="ridgecv")
    layer_data = {}

    for layer in layers:
        X, y, case_ids = trainer.prepare_data(activations, annotations, layer)

        if len(case_ids) < 3:
            print(f"  Layer {layer}: Insufficient data ({len(case_ids)} cases), skipping")
            continue

        # Fit scaler once — used for both training and correction
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train ridge model on scaled data
        n_samples = X_scaled.shape[0]
        cv = LeaveOneOut() if n_samples < 10 else KFold(
            n_splits=min(5, n_samples), shuffle=True, random_state=SEED
        )
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=cv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_scaled, y)

        # Use fresh KFold with same seed for evaluation (avoids iterator exhaustion)
        cv_eval = LeaveOneOut() if n_samples < 10 else KFold(
            n_splits=min(5, n_samples), shuffle=True, random_state=SEED
        )
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_eval, scoring="r2")
        r2 = float(np.mean(cv_scores))

        # Per-principle R²
        principle_r2 = {}
        for i, pname in enumerate(PRINCIPLE_NAMES):
            if np.std(y[:, i]) < 1e-6:
                principle_r2[pname] = 0.0
            else:
                cv_p = LeaveOneOut() if n_samples < 10 else KFold(
                    n_splits=min(5, n_samples), shuffle=True, random_state=SEED
                )
                scores = cross_val_score(
                    Ridge(alpha=model.alpha_), X_scaled, y[:, i], cv=cv_p, scoring="r2"
                )
                principle_r2[pname] = float(np.mean(scores))

        weights = model.coef_  # (n_principles, d_model)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)

        if raw_directions:
            directions = weights.copy()
        else:
            # Scaler correction: native_direction = weights / scale
            directions = weights / scaler.scale_[np.newaxis, :]

        # Normalize each direction to unit vector
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        directions = directions / norms

        layer_data[layer] = {
            "directions": directions,
            "r2_score": r2,
            "principle_r2": principle_r2,
            "scaler_applied": not raw_directions,
            "ridge_alpha": float(model.alpha_),
        }

        import sklearn
        print(f"  Layer {layer}: R²={r2:.3f}, alpha={model.alpha_}, "
              f"n={n_samples}, p={X_scaled.shape[1]}, "
              f"sklearn={sklearn.__version__}")

    return layer_data


# === Test Case Selection ===

def select_test_cases(
    annotations: list,
    cases: list[dict],
    max_per_principle: int = 5,
    max_weight_threshold: float = 0.2,
) -> dict[str, list[dict]]:
    """
    For each principle, select cases where it has low ground-truth weight.

    These are the best candidates for steering: if we add the principle direction
    and the model starts invoking that principle, it's strong causal evidence.

    Returns:
        dict mapping principle_name -> list of case dicts (with 'weight' added)
    """
    annotation_lookup = {a.case_id: a for a in annotations}
    case_lookup = {c["case_id"]: c for c in cases}

    test_cases = {}

    for principle in PRINCIPLE_NAMES:
        candidates = []
        for ann in annotations:
            weight = ann.weights.get(principle, 0.0)
            if weight < max_weight_threshold and ann.case_id in case_lookup:
                candidates.append({
                    **case_lookup[ann.case_id],
                    "ground_truth_weight": weight,
                })

        # Sort by weight ascending (prefer cases with zero weight)
        candidates.sort(key=lambda x: x["ground_truth_weight"])
        test_cases[principle] = candidates[:max_per_principle]

        print(f"  {principle}: {len(test_cases[principle])} test cases "
              f"(from {len(candidates)} candidates with weight < {max_weight_threshold})")

    return test_cases


# === Prompt Formatting ===

def format_prompt_for_steering(case: dict, tokenizer) -> str:
    """
    Format case into behavioral prompt with chat template applied.

    Uses the same PROMPT_TEMPLATE from generate_behavioral_responses.py,
    then wraps it in the model's chat template for instruct-mode generation.
    """
    prompt = PROMPT_TEMPLATE.format(
        facts=case.get("facts", ""),
        legal_question=case.get("legal_question", ""),
    )

    # Apply chat template for instruct model
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted
    except Exception:
        # Fallback
        return f"User: {prompt}\n\nAssistant:"


# === Response Evaluation ===

def evaluate_steered_response(
    response: str,
    steered_principle: str,
) -> tuple[Optional[int], list, Optional[str]]:
    """
    Parse the model's ranking response and find the steered principle's position.

    Args:
        response: raw model output
        steered_principle: principle name in underscore format (e.g., "free_expression")

    Returns:
        (rank, parsed_rankings, top_principle)
        rank is 1-indexed position of steered principle, None if not found
    """
    parsed = parse_ranking(response)

    # Map underscore principle names to display names for matching
    display_map = {
        "free_expression": "Free Expression",
        "equal_protection": "Equal Protection",
        "due_process": "Due Process",
        "federalism": "Federalism",
        "privacy_liberty": "Privacy/Liberty",
    }
    target_display = display_map.get(steered_principle, steered_principle)

    top_principle = parsed[0]["principle"] if parsed else None

    # Find rank of steered principle
    for i, entry in enumerate(parsed):
        if entry["principle"] == target_display:
            return i + 1, parsed, top_principle

    return None, parsed, top_principle


# === Main Experiment ===

def run_steering_experiment(
    layers: list[int],
    alphas: list[float],
    device: str = "auto",
    max_cases_per_principle: int = 5,
    max_new_tokens: int = 300,
    raw_directions: bool = False,
    output_suffix: str = "",
    norm_relative: bool = False,
) -> SteeringResults:
    """
    Run the full steering vector experiment.

    1. Load aligned model activations and annotations
    2. Extract scaler-corrected principle directions
    3. Load aligned model via TransformerLens
    4. For each layer x principle x case x alpha: generate steered response
    5. Save results incrementally

    Args:
        layers: which layers to steer at
        alphas: steering strengths to test (if norm_relative, these are fractions
                of the mean residual stream L2 norm, e.g. 0.5 = 50% of resid norm)
        device: compute device
        max_cases_per_principle: max test cases per principle
        max_new_tokens: max tokens to generate per response
        raw_directions: skip scaler correction if True
        output_suffix: appended to output directory name (e.g., "_large_alpha")
        norm_relative: if True, alpha values are interpreted as fractions of the
                       mean residual stream L2 norm at each layer. E.g., alpha=0.5
                       means the steering perturbation has L2 norm = 0.5 * ||resid||.
    """
    subdir = "steering" + (f"_{output_suffix}" if output_suffix else "")
    output_path = OUTPUT_DIR / subdir
    output_path.mkdir(parents=True, exist_ok=True)

    results = SteeringResults(
        model=ALIGNED_MODEL,
        layers=layers,
        alphas=alphas,
        timestamp=datetime.now().isoformat(),
        norm_relative=norm_relative,
    )

    # === Phase 1: Extract directions ===
    print("\n" + "=" * 60)
    print("PHASE 1: Extracting principle directions")
    print("=" * 60)

    act_dir = OUTPUT_DIR / "activations" / "aligned"
    annotations_path = OUTPUT_DIR / "annotations.json"

    print(f"Loading activations from {act_dir}...")
    activations = load_activation_dataset(str(act_dir))
    print(f"  Loaded {len(activations)} activation caches")

    print(f"Loading annotations from {annotations_path}...")
    annotations = load_annotations(str(annotations_path))
    print(f"  Loaded {len(annotations)} annotations")

    print(f"\nExtracting directions for layers {layers}...")
    layer_data = extract_directions_with_scaler_correction(
        activations, annotations, layers, raw_directions=raw_directions
    )

    # Save directions for reuse
    directions_cache = {}
    for layer, data in layer_data.items():
        directions_cache[str(layer)] = data["directions"]
    np.savez_compressed(
        output_path / "principle_directions.npz",
        **directions_cache,
    )
    print(f"Saved directions to {output_path / 'principle_directions.npz'}")

    # Compute mean residual stream norms per layer (for norm-relative scaling)
    resid_norms = {}
    if norm_relative:
        print("\nComputing residual stream norms for norm-relative scaling...")
        ann_case_ids = {a.case_id for a in annotations}
        for layer in layers:
            norms = []
            for case_id, cache in activations.items():
                if case_id in ann_case_ids:
                    norms.append(float(np.linalg.norm(
                        cache.residual_activations[layer]
                    )))
            mean_norm = float(np.mean(norms))
            resid_norms[layer] = mean_norm
            print(f"  Layer {layer}: mean ||resid|| = {mean_norm:.2f} "
                  f"(across {len(norms)} cases)")
        results.resid_norms = {str(k): v for k, v in resid_norms.items()}

    # === Phase 2: Select test cases ===
    print("\n" + "=" * 60)
    print("PHASE 2: Selecting test cases")
    print("=" * 60)

    all_cases_lookup = {c["case_id"]: c for c in ALL_CASES}
    # Only use cases that have both activations and annotations
    available_case_ids = set(activations.keys()) & {a.case_id for a in annotations}
    available_cases = [all_cases_lookup[cid] for cid in available_case_ids if cid in all_cases_lookup]

    test_cases = select_test_cases(
        annotations, available_cases,
        max_per_principle=max_cases_per_principle,
    )

    # Count total trials
    total_trials = 0
    for principle in PRINCIPLE_NAMES:
        n_cases = len(test_cases.get(principle, []))
        total_trials += len(layers) * n_cases * len(alphas)
    print(f"\nTotal trials planned: {total_trials}")

    # Free activation data (no longer needed)
    del activations
    gc.collect()

    # === Phase 3: Load model and run steering ===
    print("\n" + "=" * 60)
    print("PHASE 3: Running steering experiment")
    print("=" * 60)

    print(f"Loading model: {ALIGNED_MODEL}...")
    patcher = ActivationPatcher(ALIGNED_MODEL, device=device)

    # Get tokenizer from the TransformerLens model
    tokenizer = patcher.model.tokenizer

    trial_count = 0
    start_time = time.time()

    for layer in layers:
        if layer not in layer_data:
            print(f"\nSkipping layer {layer} (no directions)")
            continue

        directions_matrix = layer_data[layer]["directions"]  # (n_principles, d_model)
        r2 = layer_data[layer]["r2_score"]

        # Compute effective scale multiplier for norm-relative mode
        if norm_relative:
            norm_multiplier = resid_norms[layer]
            print(f"\n--- Layer {layer} (probe R²={r2:.3f}, "
                  f"||resid||={norm_multiplier:.0f}, "
                  f"norm-relative alphas) ---")
        else:
            norm_multiplier = 1.0
            print(f"\n--- Layer {layer} (probe R²={r2:.3f}) ---")

        for p_idx, principle in enumerate(PRINCIPLE_NAMES):
            principle_direction = directions_matrix[p_idx]  # (d_model,)
            cases_for_principle = test_cases.get(principle, [])

            if not cases_for_principle:
                print(f"  {principle}: no test cases, skipping")
                continue

            print(f"\n  Steering toward: {principle} ({len(cases_for_principle)} cases)")

            for case in cases_for_principle:
                case_id = case["case_id"]
                case_name = case.get("case_name", case_id)
                gt_weight = case["ground_truth_weight"]

                # Format prompt once for this case
                formatted_prompt = format_prompt_for_steering(case, tokenizer)

                for alpha in alphas:
                    trial_count += 1
                    elapsed = time.time() - start_time
                    rate = trial_count / elapsed if elapsed > 0 else 0
                    eta = (total_trials - trial_count) / rate if rate > 0 else 0

                    # Effective scale: alpha * ||resid|| in norm-relative mode
                    effective_scale = alpha * norm_multiplier

                    if norm_relative:
                        print(f"    [{trial_count}/{total_trials}] "
                              f"case={case_id[:20]}, alpha={alpha:+.2f} "
                              f"(effective={effective_scale:+.0f}, "
                              f"ETA: {eta/60:.0f}min)", end="")
                    else:
                        print(f"    [{trial_count}/{total_trials}] "
                              f"case={case_id[:20]}, alpha={alpha:+.1f} "
                              f"(ETA: {eta/60:.0f}min)", end="")

                    # Build direction dict for this layer
                    direction_dict = {layer: principle_direction}

                    if alpha == 0:
                        # Baseline: no steering
                        response = patcher.generate_response(
                            formatted_prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=0.0,
                        )
                    else:
                        # Steer at all positions (Turner et al. 2023 style)
                        response = patcher.generate_with_direction_add_all_positions(
                            formatted_prompt,
                            direction_dict,
                            scale=effective_scale,
                            max_new_tokens=max_new_tokens,
                            temperature=0.0,
                        )

                    # Evaluate
                    rank, parsed, top_principle = evaluate_steered_response(
                        response, principle
                    )

                    trial = SteeringTrial(
                        case_id=case_id,
                        case_name=case_name,
                        layer=layer,
                        steered_principle=principle,
                        alpha=alpha,
                        ground_truth_weight=gt_weight,
                        response=response,
                        parsed_rankings=parsed,
                        steered_principle_rank=rank,
                        top_principle=top_principle,
                        effective_scale=effective_scale,
                    )
                    results.trials.append(asdict(trial))
                    results.n_trials = len(results.trials)

                    rank_str = f"rank={rank}" if rank else "not found"
                    print(f" -> {rank_str}, top={top_principle}")

                # Save incrementally after each case
                _save_results(results, output_path)

    # Clean up
    del patcher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Phase 4: Generate summary ===
    print("\n" + "=" * 60)
    print("PHASE 4: Generating summary")
    print("=" * 60)

    _generate_summary(results, output_path)

    total_time = time.time() - start_time
    print(f"\nExperiment complete: {results.n_trials} trials in {total_time/60:.1f} min")
    print(f"Results saved to {output_path}/")

    return results


# === Output Helpers ===

def _save_results(results: SteeringResults, output_path: Path):
    """Save results to JSON (incremental)."""
    with open(output_path / "steering_results.json", "w") as f:
        json.dump(asdict(results), f, indent=2)


def _generate_summary(results: SteeringResults, output_path: Path):
    """Generate markdown summary with per-principle alpha sweep tables."""
    lines = [
        "# Steering Vector Experiment Results",
        f"\n**Model**: {results.model}",
        f"**Layers**: {results.layers}",
        f"**Alphas**: {results.alphas}",
        f"**Total Trials**: {results.n_trials}",
        f"**Timestamp**: {results.timestamp}",
        "\n---\n",
    ]

    # Group trials by principle
    by_principle = {}
    for trial in results.trials:
        p = trial["steered_principle"]
        if p not in by_principle:
            by_principle[p] = []
        by_principle[p].append(trial)

    for principle in PRINCIPLE_NAMES:
        trials = by_principle.get(principle, [])
        if not trials:
            continue

        lines.append(f"\n## {principle.replace('_', ' ').title()}")
        lines.append(f"\n{len(trials)} trials across {len(results.layers)} layers\n")

        # Alpha sweep table: average rank by alpha
        by_alpha = {}
        for trial in trials:
            a = trial["alpha"]
            if a not in by_alpha:
                by_alpha[a] = {"ranks": [], "found": 0, "total": 0}
            by_alpha[a]["total"] += 1
            if trial["steered_principle_rank"] is not None:
                by_alpha[a]["ranks"].append(trial["steered_principle_rank"])
                by_alpha[a]["found"] += 1

        lines.append("| Alpha | Avg Rank | Found/Total | Top-1 Rate |")
        lines.append("|------:|--------:|-----------:|----------:|")

        for alpha in sorted(by_alpha.keys()):
            data = by_alpha[alpha]
            avg_rank = np.mean(data["ranks"]) if data["ranks"] else float("nan")
            found_rate = f"{data['found']}/{data['total']}"
            top1 = sum(1 for r in data["ranks"] if r == 1) / data["total"] if data["total"] else 0
            lines.append(f"| {alpha:+.1f} | {avg_rank:.2f} | {found_rate} | {top1:.0%} |")

        # Per-layer breakdown
        by_layer = {}
        for trial in trials:
            l = trial["layer"]
            if l not in by_layer:
                by_layer[l] = {}
            a = trial["alpha"]
            if a not in by_layer[l]:
                by_layer[l][a] = []
            if trial["steered_principle_rank"] is not None:
                by_layer[l][a].append(trial["steered_principle_rank"])

        for layer in sorted(by_layer.keys()):
            lines.append(f"\n### Layer {layer}")
            lines.append("| Alpha | Avg Rank | N |")
            lines.append("|------:|--------:|--:|")
            for alpha in sorted(by_layer[layer].keys()):
                ranks = by_layer[layer][alpha]
                avg = np.mean(ranks) if ranks else float("nan")
                lines.append(f"| {alpha:+.1f} | {avg:.2f} | {len(ranks)} |")

    # Monotonicity check
    lines.append("\n---\n")
    lines.append("## Monotonicity Analysis")
    lines.append("\nDo rank positions decrease (become more salient) as alpha increases?\n")

    for principle in PRINCIPLE_NAMES:
        trials = by_principle.get(principle, [])
        if not trials:
            continue

        # Compute correlation between alpha and rank across all trials
        alphas_arr = []
        ranks_arr = []
        for t in trials:
            if t["steered_principle_rank"] is not None:
                alphas_arr.append(t["alpha"])
                ranks_arr.append(t["steered_principle_rank"])

        if len(alphas_arr) >= 3:
            corr = np.corrcoef(alphas_arr, ranks_arr)[0, 1]
            direction = "monotonic decrease (expected)" if corr < -0.1 else (
                "no clear trend" if abs(corr) <= 0.1 else "unexpected increase"
            )
            lines.append(f"- **{principle}**: r={corr:.3f} ({direction})")
        else:
            lines.append(f"- **{principle}**: insufficient data")

    summary_text = "\n".join(lines)

    with open(output_path / "steering_summary.md", "w") as f:
        f.write(summary_text)

    print(f"Summary saved to {output_path / 'steering_summary.md'}")


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="Steering Vector Experiment for Constitutional Principles"
    )
    parser.add_argument(
        "--layers", type=str, default="20,23,26",
        help="Comma-separated layers to steer at (default: 20,23,26)"
    )
    parser.add_argument(
        "--alphas", type=str, default="-3,-2,-1,-0.5,0,0.5,1,2,3",
        help="Comma-separated steering strengths (default: -3,-2,-1,-0.5,0,0.5,1,2,3)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device (default: auto)"
    )
    parser.add_argument(
        "--max-cases", type=int, default=5,
        help="Max test cases per principle (default: 5)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=300,
        help="Max tokens to generate per response (default: 300)"
    )
    parser.add_argument(
        "--raw-directions", action="store_true",
        help="Skip scaler correction (use raw probe weights as directions)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode: 1 layer, 2 cases, 3 alphas (~30 trials)"
    )
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help="Suffix for output directory (e.g., 'large_alpha' -> steering_large_alpha/)"
    )
    parser.add_argument(
        "--norm-relative", action="store_true",
        help="Interpret alpha values as fractions of mean residual stream L2 norm. "
             "E.g., alpha=0.5 means perturbation L2 norm = 50%% of ||resid||."
    )

    args = parser.parse_args()

    if args.quick:
        layers = [23]
        alphas = [-2.0, 0.0, 2.0]
        max_cases = 2
        print("QUICK TEST MODE: 1 layer, 2 cases/principle, 3 alphas")
    else:
        layers = [int(x) for x in args.layers.split(",")]
        alphas = [float(x) for x in args.alphas.split(",")]
        max_cases = args.max_cases

    print(f"Layers: {layers}")
    print(f"Alphas: {alphas}")
    print(f"Max cases per principle: {max_cases}")
    print(f"Device: {args.device}")
    print(f"Scaler correction: {'OFF (raw)' if args.raw_directions else 'ON'}")
    print(f"Norm-relative: {args.norm_relative}")

    run_steering_experiment(
        layers=layers,
        alphas=alphas,
        device=args.device,
        max_cases_per_principle=max_cases,
        max_new_tokens=args.max_new_tokens,
        raw_directions=args.raw_directions,
        output_suffix=args.output_suffix,
        norm_relative=args.norm_relative,
    )


if __name__ == "__main__":
    main()
