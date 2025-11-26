"""
SCOTUS Constitutional Geometry Experiment - Main Runner

This script orchestrates the full experiment:
1. Fetch SCOTUS opinion texts from CourtListener
2. Annotate principles using Claude Opus
3. Extract activations from base and aligned models  
4. Train linear probes and compare performance

Run with: python run_experiment.py --phase [1|2|3|all]
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add local modules
sys.path.insert(0, str(Path(__file__).parent))

from cases import CASES, format_prompt, get_all_case_ids


# === Configuration ===

CONFIG = {
    "output_dir": "./experiment_output",
    "model_pair": "llama2-7b",  # See extract_activations.py for options
    "extraction_method": "last_token",
    "cv_folds": 5,
    "device": "auto",
}


# === Phase 1: Fetch Opinion Texts ===

def fetch_opinions(output_dir: str, max_cases: int = None):
    """
    Fetch majority opinion texts from CourtListener API.
    
    Note: CourtListener has rate limits. Be respectful.
    """
    import requests
    import time
    
    opinions_dir = Path(output_dir) / "opinions"
    opinions_dir.mkdir(parents=True, exist_ok=True)
    
    cases_to_fetch = CASES[:max_cases] if max_cases else CASES
    
    print(f"\nFetching {len(cases_to_fetch)} opinions from CourtListener...")
    print("=" * 50)
    
    # CourtListener API endpoint
    BASE_URL = "https://www.courtlistener.com/api/rest/v3"
    
    results = {}
    
    for i, case in enumerate(cases_to_fetch):
        case_id = case["case_id"]
        cl_id = case.get("courtlistener_id")
        
        print(f"  [{i+1}/{len(cases_to_fetch)}] {case['case_name'][:50]}...")
        
        opinion_file = opinions_dir / f"{case_id}.txt"
        
        # Check if already fetched
        if opinion_file.exists():
            print(f"    Already cached")
            with open(opinion_file, 'r') as f:
                results[case_id] = f.read()
            continue
        
        if not cl_id:
            print(f"    No CourtListener ID, skipping")
            continue
        
        try:
            # Search for the case
            search_url = f"{BASE_URL}/search/"
            params = {
                "q": case["case_name"],
                "type": "o",  # Opinions
                "court": "scotus"
            }
            
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("results"):
                # Find best matching result (prefer exact year match)
                results = data["results"]
                best_result = None
                for r in results:
                    # Check if year matches
                    result_year = r.get("dateFiled", "")[:4] if r.get("dateFiled") else ""
                    if result_year == str(case["year"]):
                        best_result = r
                        break

                # Fall back to first result if no year match
                if best_result is None:
                    best_result = results[0]
                    print(f"    Warning: No exact year match, using first result")

                result = best_result
                opinion_text = result.get("text", result.get("plain_text", ""))
                
                if opinion_text:
                    # Save to file
                    with open(opinion_file, 'w') as f:
                        f.write(opinion_text)
                    results[case_id] = opinion_text
                    print(f"    Fetched ({len(opinion_text)} chars)")
                else:
                    print(f"    No opinion text in result")
            else:
                print(f"    No results found")
            
            # Rate limit
            time.sleep(1.0)
            
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"\nFetched {len(results)} opinions")
    return results


def load_cached_opinions(output_dir: str) -> dict:
    """Load previously fetched opinions."""
    opinions_dir = Path(output_dir) / "opinions"
    results = {}
    
    for filepath in opinions_dir.glob("*.txt"):
        case_id = filepath.stem
        with open(filepath, 'r') as f:
            results[case_id] = f.read()
    
    return results


# === Phase 2: Annotate with Opus ===

def annotate_principles(output_dir: str, opinions: dict, api_key: str = None):
    """
    Use Claude Opus to extract principle weights from opinions.
    """
    from annotate_principles import (
        OpusAnnotator, 
        save_annotations,
        PrincipleAnnotation
    )
    
    annotations_file = Path(output_dir) / "annotations.json"
    
    # Load existing annotations
    existing = {}
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            existing = {a["case_id"]: a for a in data}
    
    print(f"\nAnnotating principles with Claude Opus...")
    print(f"  Already annotated: {len(existing)}")
    print("=" * 50)
    
    annotator = OpusAnnotator(api_key=api_key)
    
    # Create lookup for case metadata
    case_lookup = {c["case_id"]: c for c in CASES}
    
    annotations = []
    
    for case_id, opinion_text in opinions.items():
        # Skip if already annotated
        if case_id in existing:
            annotations.append(PrincipleAnnotation(**existing[case_id]))
            continue
        
        if case_id not in case_lookup:
            print(f"  {case_id}: Not in case list, skipping")
            continue
        
        case = case_lookup[case_id]
        print(f"  Annotating: {case['case_name'][:50]}...")
        
        try:
            annotation = annotator.annotate_case(
                case_id=case_id,
                case_name=case["case_name"],
                year=case["year"],
                opinion_text=opinion_text
            )
            annotations.append(annotation)
            
            # Save incrementally
            save_annotations(annotations, str(annotations_file))
            
            print(f"    Weights: {annotation.weights}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"\nCompleted {len(annotations)} annotations")
    return annotations


def load_annotations(output_dir: str):
    """Load previously created annotations."""
    from annotate_principles import load_annotations as load_ann
    annotations_file = Path(output_dir) / "annotations.json"
    if not annotations_file.exists():
        raise FileNotFoundError(
            f"Annotations file not found: {annotations_file}\n"
            "Run the 'annotate' phase first: python run_experiment.py --phase annotate"
        )
    return load_ann(str(annotations_file))


# === Phase 3: Extract Activations ===

def extract_activations_phase(
    output_dir: str,
    model_pair: str = "llama2-7b",
    method: str = "last_token",
    device: str = "auto"
):
    """
    Extract residual stream activations from base and aligned models.
    """
    import torch
    from extract_activations import ActivationExtractor, load_activation_dataset
    
    model_info = ActivationExtractor.MODEL_PAIRS.get(model_pair)
    if not model_info:
        raise ValueError(f"Unknown model pair: {model_pair}")
    
    # Prepare prompts
    prompts = [
        {"case_id": c["case_id"], "prompt": format_prompt(c)}
        for c in CASES
    ]
    
    print(f"\nExtracting activations for {len(prompts)} cases")
    print(f"  Model pair: {model_pair}")
    print(f"  Method: {method}")
    print("=" * 50)
    
    # Extract from base model
    base_dir = Path(output_dir) / "activations" / "base"
    base_cached = base_dir.exists() and list(base_dir.glob("*.npz"))
    if not base_cached:
        print(f"\n--- BASE MODEL: {model_info['base']} ---")
        extractor = ActivationExtractor(
            model_info["base"],
            device=device
        )
        extractor.extract_batch(prompts, method=method, output_dir=str(base_dir))
        del extractor  # Free memory
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"\n  Base activations already cached in {base_dir}")
    
    # Extract from aligned model
    aligned_dir = Path(output_dir) / "activations" / "aligned"
    aligned_cached = aligned_dir.exists() and list(aligned_dir.glob("*.npz"))
    if not aligned_cached:
        print(f"\n--- ALIGNED MODEL: {model_info['aligned']} ---")
        extractor = ActivationExtractor(
            model_info["aligned"],
            device=device
        )
        extractor.extract_batch(prompts, method=method, output_dir=str(aligned_dir))
        del extractor  # Free memory
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"\n  Aligned activations already cached in {aligned_dir}")
    
    print("\nActivation extraction complete")


# === Phase 4: Train Probes and Compare ===

def train_and_compare(output_dir: str, cv_folds: int = 5):
    """
    Train linear probes on both models and compare performance.
    """
    from extract_activations import load_activation_dataset
    from train_probes import (
        compare_models, 
        save_comparison,
        plot_layer_comparison
    )
    
    print(f"\nTraining linear probes...")
    print("=" * 50)
    
    # Load data
    base_dir = Path(output_dir) / "activations" / "base"
    aligned_dir = Path(output_dir) / "activations" / "aligned"
    
    base_activations = load_activation_dataset(str(base_dir))
    aligned_activations = load_activation_dataset(str(aligned_dir))
    annotations = load_annotations(output_dir)
    
    print(f"  Base activations: {len(base_activations)} cases")
    print(f"  Aligned activations: {len(aligned_activations)} cases")
    print(f"  Annotations: {len(annotations)} cases")
    
    # Get number of layers from first cache
    first_cache = next(iter(base_activations.values()))
    n_layers = first_cache.n_layers
    print(f"  Number of layers: {n_layers}")
    
    # Run comparison
    comparison = compare_models(
        base_activations,
        aligned_activations,
        annotations,
        n_layers,
        cv_folds=cv_folds
    )
    
    # Print results
    print("\n" + comparison.summary_report())
    
    # Save results
    results_file = Path(output_dir) / "probe_comparison.json"
    save_comparison(comparison, str(results_file))
    
    # Generate plot
    try:
        plot_file = Path(output_dir) / "layer_comparison.png"
        plot_layer_comparison(comparison, str(plot_file))
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    return comparison


# === Full Pipeline ===

def run_full_experiment(
    output_dir: str = "./experiment_output",
    model_pair: str = "llama2-7b",
    api_key: str = None,
    skip_fetch: bool = False
):
    """
    Run the complete experiment pipeline.
    """
    # Validate API key early to avoid failing mid-experiment
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
            "or pass --api-key argument."
        )

    # Validate model pair
    from extract_activations import ActivationExtractor
    if model_pair not in ActivationExtractor.MODEL_PAIRS:
        valid_pairs = ", ".join(ActivationExtractor.MODEL_PAIRS.keys())
        raise ValueError(f"Unknown model pair: {model_pair}. Valid options: {valid_pairs}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 60)
    print("SCOTUS CONSTITUTIONAL GEOMETRY EXPERIMENT")
    print(f"Started: {timestamp}")
    print("=" * 60)
    
    # Phase 1: Fetch opinions
    if skip_fetch:
        print("\n[Phase 1] Loading cached opinions...")
        opinions = load_cached_opinions(output_dir)
    else:
        print("\n[Phase 1] Fetching SCOTUS opinions...")
        opinions = fetch_opinions(output_dir)
    
    print(f"  Available opinions: {len(opinions)}")
    
    # Phase 2: Annotate with Opus
    print("\n[Phase 2] Annotating principles with Claude Opus...")
    annotations = annotate_principles(output_dir, opinions, api_key)
    print(f"  Annotated cases: {len(annotations)}")
    
    # Phase 3: Extract activations
    print("\n[Phase 3] Extracting residual stream activations...")
    extract_activations_phase(
        output_dir,
        model_pair=model_pair,
        method="last_token"
    )
    
    # Phase 4: Train probes and compare
    print("\n[Phase 4] Training linear probes...")
    comparison = train_and_compare(output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nKey findings:")
    print(f"  Best base model R²: {comparison.best_base_r2:.4f} (layer {comparison.best_base_layer})")
    print(f"  Best aligned model R²: {comparison.best_aligned_r2:.4f} (layer {comparison.best_aligned_layer})")
    print(f"  RLHF improvement: {comparison.best_aligned_r2 - comparison.best_base_r2:+.4f}")
    
    # Interpretation
    if comparison.best_aligned_r2 > comparison.best_base_r2 + 0.05:
        print("\n  ✓ POSITIVE SIGNAL: Aligned model shows substantially better")
        print("    linear separability of constitutional principles.")
    elif comparison.best_aligned_r2 > comparison.best_base_r2:
        print("\n  ~ WEAK SIGNAL: Small improvement in aligned model.")
        print("    May need more data or different methodology.")
    else:
        print("\n  ✗ NO SIGNAL: Aligned model not better (or worse).")
        print("    Consider alternative hypotheses.")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - annotations.json (principle weights)")
    print(f"  - probe_comparison.json (probe results)")
    print(f"  - layer_comparison.png (visualization)")


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="SCOTUS Constitutional Geometry Experiment"
    )
    parser.add_argument(
        "--phase", 
        type=str, 
        default="all",
        choices=["fetch", "annotate", "extract", "probe", "all"],
        help="Which phase to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_output",
        help="Output directory"
    )
    parser.add_argument(
        "--model-pair",
        type=str,
        default="llama2-7b",
        help="Model pair to use (see extract_activations.py)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model inference (auto/cuda/cpu/mps)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip opinion fetching, use cached only"
    )
    
    args = parser.parse_args()
    
    # Get API key from env if not provided
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    
    if args.phase == "all":
        run_full_experiment(
            output_dir=args.output_dir,
            model_pair=args.model_pair,
            api_key=api_key,
            skip_fetch=args.skip_fetch
        )
    elif args.phase == "fetch":
        fetch_opinions(args.output_dir)
    elif args.phase == "annotate":
        opinions = load_cached_opinions(args.output_dir)
        annotate_principles(args.output_dir, opinions, api_key)
    elif args.phase == "extract":
        extract_activations_phase(
            args.output_dir,
            model_pair=args.model_pair,
            device=args.device
        )
    elif args.phase == "probe":
        train_and_compare(args.output_dir)


if __name__ == "__main__":
    main()
