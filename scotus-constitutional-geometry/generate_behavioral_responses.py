"""
Behavioral Verification: Generate Constitutional Principle Rankings

This script generates structured responses from base and aligned models
for SCOTUS cases, asking them to rank relevant constitutional principles.

The output is designed to be human-readable and directly comparable across models.

Usage:
    # Run locally on small model
    python generate_behavioral_responses.py --model-pair llama3.2-3b --device mps

    # Run on RunPod for larger models
    python generate_behavioral_responses.py --model-pair gemma2-27b --device cuda --use-bfloat16
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model pairs configuration
MODEL_PAIRS = {
    "llama3.2-3b": {
        "base": "meta-llama/Llama-3.2-3B",
        "aligned": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "llama3.1-8b": {
        "base": "meta-llama/Llama-3.1-8B",
        "aligned": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "mistral-7b": {
        "base": "mistralai/Mistral-7B-v0.1",
        "aligned": "mistralai/Mistral-7B-Instruct-v0.1",
    },
    "qwen25-7b": {
        "base": "Qwen/Qwen2.5-7B",
        "aligned": "Qwen/Qwen2.5-7B-Instruct",
    },
    "qwen25-32b": {
        "base": "Qwen/Qwen2.5-32B",
        "aligned": "Qwen/Qwen2.5-32B-Instruct",
    },
    "gemma2-27b": {
        "base": "google/gemma-2-27b",
        "aligned": "google/gemma-2-27b-it",
    },
}

PRINCIPLES = [
    "Free Expression",
    "Equal Protection",
    "Due Process",
    "Federalism",
    "Privacy/Liberty"
]

PROMPT_TEMPLATE = """Case Facts: {facts}

Constitutional Question: {legal_question}

List the constitutional principles that apply to this case, in order of relevance (most relevant first).

Choose from: Free Expression, Equal Protection, Due Process, Federalism, Privacy/Liberty

Format your response as:
1. [Principle] - [One sentence explaining why]
2. [Principle] - [One sentence explaining why]
(Continue for all relevant principles)

Your ranking:"""


def load_cases():
    """Load all SCOTUS cases from data/cases directory."""
    from paths import CASES_DIR
    case_data_dir = CASES_DIR

    all_cases = []
    for json_file in sorted(case_data_dir.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
        all_cases.extend(data.get("cases", []))

    print(f"Loaded {len(all_cases)} cases")
    return all_cases


def setup_hf_auth():
    """Set up HuggingFace authentication."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)


def load_model(model_name: str, device: str, use_bfloat16: bool = False):
    """Load a model and tokenizer."""
    print(f"Loading {model_name}...")

    dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to("mps")
    elif device == "cpu":
        model = model.to("cpu")

    model.eval()
    print(f"  Loaded: {model.config.num_hidden_layers} layers")

    return model, tokenizer


def format_prompt_for_model(prompt: str, model_name: str, tokenizer, is_instruct: bool):
    """Format prompt appropriately for base vs instruct models."""
    if not is_instruct:
        # Base model: just use the prompt directly
        return prompt

    # Instruct model: use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception:
            pass

    # Fallback: simple instruction format
    return f"User: {prompt}\n\nAssistant:"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.3,  # Lower temp for more consistent rankings
):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return response.strip()


def parse_ranking(response: str) -> list:
    """
    Parse the model's response to extract principle rankings.
    Returns list of (principle, justification) tuples.
    """
    rankings = []
    lines = response.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered items like "1. Free Expression - ..."
        for principle in PRINCIPLES:
            if principle.lower() in line.lower():
                # Extract justification after the dash
                parts = line.split('-', 1)
                justification = parts[1].strip() if len(parts) > 1 else ""
                rankings.append({
                    "principle": principle,
                    "justification": justification,
                    "raw_line": line
                })
                break

    return rankings


def run_behavioral_verification(
    model_pair_name: str,
    output_dir: Path,
    device: str = "cuda",
    use_bfloat16: bool = False,
    max_cases: int = None,
):
    """Run behavioral verification for a model pair."""

    setup_hf_auth()

    model_info = MODEL_PAIRS[model_pair_name]
    cases = load_cases()

    if max_cases:
        cases = cases[:max_cases]
        print(f"Running on {max_cases} cases (subset)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_pair": model_pair_name,
        "base_model": model_info["base"],
        "aligned_model": model_info["aligned"],
        "timestamp": datetime.now().isoformat(),
        "n_cases": len(cases),
        "responses": []
    }

    # Process each model type
    for model_type in ["base", "aligned"]:
        model_name = model_info[model_type]
        is_instruct = model_type == "aligned"

        print(f"\n{'='*60}")
        print(f"Processing {model_type.upper()} model: {model_name}")
        print('='*60)

        model, tokenizer = load_model(model_name, device, use_bfloat16)

        for i, case in enumerate(cases):
            case_id = case.get("case_id", f"case_{i}")
            print(f"  [{i+1}/{len(cases)}] {case_id}")

            # Build prompt
            prompt = PROMPT_TEMPLATE.format(
                facts=case.get("facts", ""),
                legal_question=case.get("legal_question", "")
            )

            # Format for model type
            formatted_prompt = format_prompt_for_model(
                prompt, model_name, tokenizer, is_instruct
            )

            # Generate response
            try:
                response = generate_response(model, tokenizer, formatted_prompt)
                parsed = parse_ranking(response)
                error = None
            except Exception as e:
                response = ""
                parsed = []
                error = str(e)
                print(f"    Error: {e}")

            # Find or create case entry in results
            case_entry = None
            for entry in results["responses"]:
                if entry["case_id"] == case_id:
                    case_entry = entry
                    break

            if case_entry is None:
                case_entry = {
                    "case_id": case_id,
                    "case_name": case.get("case_name", ""),
                    "primary_principle": case.get("primary_principle", ""),
                    "facts_preview": case.get("facts", "")[:200] + "...",
                    "legal_question": case.get("legal_question", ""),
                }
                results["responses"].append(case_entry)

            # Add model response
            case_entry[f"{model_type}_response"] = {
                "raw": response,
                "parsed_rankings": parsed,
                "top_principle": parsed[0]["principle"] if parsed else None,
                "error": error,
            }

        # Clean up model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    output_file = output_dir / "behavioral_responses.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # Generate summary
    generate_summary(results, output_dir)

    return results


def load_annotation_weights(output_dir: Path) -> dict:
    """Load principle weights from annotations.json if available."""
    from paths import ANNOTATIONS_FILE
    possible_paths = [
        ANNOTATIONS_FILE,
        output_dir / "annotations.json",
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                annotations = json.load(f)
            # Create lookup by case_id
            return {a["case_id"]: a.get("weights", {}) for a in annotations}

    return {}


def normalize_principle(principle: str) -> str:
    """Normalize principle name for comparison."""
    if not principle:
        return ""
    return principle.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def get_weight_for_pick(weights: dict, pick: str) -> float:
    """Get the weight for a picked principle."""
    pick_norm = normalize_principle(pick)

    # Map common variations
    mappings = {
        "free_expression": "free_expression",
        "freeexpression": "free_expression",
        "equal_protection": "equal_protection",
        "equalprotection": "equal_protection",
        "due_process": "due_process",
        "dueprocess": "due_process",
        "federalism": "federalism",
        "privacy_liberty": "privacy_liberty",
        "privacyliberty": "privacy_liberty",
        "privacy": "privacy_liberty",
        "liberty": "privacy_liberty",
    }

    key = mappings.get(pick_norm, pick_norm)
    return weights.get(key, 0.0)


def generate_summary(results: dict, output_dir: Path):
    """Generate a human-readable summary of the results."""

    # Load annotation weights for partial credit scoring
    annotation_weights = load_annotation_weights(output_dir)

    summary_lines = [
        "# Behavioral Verification Summary",
        f"\n**Model Pair**: {results['model_pair']}",
        f"- Base: {results['base_model']}",
        f"- Aligned: {results['aligned_model']}",
        f"\n**Cases Evaluated**: {results['n_cases']}",
        f"**Timestamp**: {results['timestamp']}",
        "\n---\n",
        "## Principle Identification Accuracy",
        "\n*Scoring: ✓ = exact match (weight ≥ 0.8), ◐ = partial match (weight ≥ 0.3), ✗ = miss (weight < 0.3)*",
        "\n| Case | Primary | Base Pick | Aligned Pick | Base | Aligned |",
        "|------|---------|-----------|--------------|------|---------|",
    ]

    base_exact = 0
    base_partial = 0
    aligned_exact = 0
    aligned_partial = 0

    for entry in results["responses"]:
        case_id = entry.get("case_id", "")
        gt = entry.get("primary_principle", "").replace("_", " ").title()
        weights = annotation_weights.get(case_id, {})

        base_top = entry.get("base_response", {}).get("top_principle")
        aligned_top = entry.get("aligned_response", {}).get("top_principle")

        # Get weights for picks
        base_weight = get_weight_for_pick(weights, base_top) if base_top else 0.0
        aligned_weight = get_weight_for_pick(weights, aligned_top) if aligned_top else 0.0

        # Determine match level
        if base_weight >= 0.8:
            base_match = "✓"
            base_exact += 1
        elif base_weight >= 0.3:
            base_match = "◐"
            base_partial += 1
        else:
            base_match = "✗"

        if aligned_weight >= 0.8:
            aligned_match = "✓"
            aligned_exact += 1
        elif aligned_weight >= 0.3:
            aligned_match = "◐"
            aligned_partial += 1
        else:
            aligned_match = "✗"

        case_name = entry.get("case_name", entry.get("case_id", ""))[:30]
        summary_lines.append(
            f"| {case_name} | {gt} | {base_top or '-'} | {aligned_top or '-'} | {base_match} | {aligned_match} |"
        )

    # Calculate scores (exact = 1.0, partial = 0.5)
    n = len(results["responses"])
    base_score = base_exact + 0.5 * base_partial
    aligned_score = aligned_exact + 0.5 * aligned_partial

    summary_lines.extend([
        "\n---\n",
        "## Overall Accuracy",
        f"\n**Exact matches (weight ≥ 0.8):**",
        f"- Base model: {base_exact}/{n} ({100*base_exact/n:.1f}%)",
        f"- Aligned model: {aligned_exact}/{n} ({100*aligned_exact/n:.1f}%)",
        f"\n**Partial matches (weight ≥ 0.3):**",
        f"- Base model: {base_partial}/{n}",
        f"- Aligned model: {aligned_partial}/{n}",
        f"\n**Weighted Score (exact=1, partial=0.5):**",
        f"- Base model: {base_score:.1f}/{n} ({100*base_score/n:.1f}%)",
        f"- Aligned model: {aligned_score:.1f}/{n} ({100*aligned_score/n:.1f}%)",
        f"- **Improvement**: {aligned_score - base_score:+.1f} points ({100*(aligned_score - base_score)/n:+.1f}pp)",
    ])

    # Add example responses
    summary_lines.extend([
        "\n---\n",
        "## Example Responses",
        "\n### Case: " + results["responses"][0].get("case_name", "First Case"),
        f"\n**Facts**: {results['responses'][0].get('facts_preview', '')}",
        f"\n**Question**: {results['responses'][0].get('legal_question', '')}",
        f"\n**Ground Truth**: {results['responses'][0].get('primary_principle', '')}",
        "\n#### Base Model Response:",
        "```",
        results["responses"][0].get("base_response", {}).get("raw", "(no response)"),
        "```",
        "\n#### Aligned Model Response:",
        "```",
        results["responses"][0].get("aligned_response", {}).get("raw", "(no response)"),
        "```",
    ])

    summary_text = "\n".join(summary_lines)

    summary_file = output_dir / "behavioral_summary.md"
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"Saved summary to {summary_file}")

    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY (with partial credit)")
    print("="*60)
    print(f"Base exact/partial:     {base_exact}/{base_partial} → score {base_score:.1f}/{n} ({100*base_score/n:.1f}%)")
    print(f"Aligned exact/partial:  {aligned_exact}/{aligned_partial} → score {aligned_score:.1f}/{n} ({100*aligned_score/n:.1f}%)")
    print(f"Improvement:            {aligned_score - base_score:+.1f} points")


def main():
    parser = argparse.ArgumentParser(description="Generate behavioral verification responses")
    parser.add_argument(
        "--model-pair",
        type=str,
        required=True,
        choices=list(MODEL_PAIRS.keys()),
        help="Model pair to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/{model_key}/)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--use-bfloat16",
        action="store_true",
        help="Use bfloat16 precision (recommended for Gemma)"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to process (for testing)"
    )

    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        from paths import RESULTS_DIR
        # Map model-pair CLI names to results dir keys
        _pair_to_key = {
            "llama3.2-3b": "llama32_3b", "llama3.1-8b": "llama31_8b",
            "mistral-7b": "mistral_7b", "qwen25-7b": "qwen25_7b",
            "qwen25-32b": "qwen25_32b", "gemma2-27b": "gemma2_27b",
        }
        output_dir = str(RESULTS_DIR / _pair_to_key.get(args.model_pair, args.model_pair))

    run_behavioral_verification(
        model_pair_name=args.model_pair,
        output_dir=Path(output_dir),
        device=args.device,
        use_bfloat16=args.use_bfloat16,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    main()
