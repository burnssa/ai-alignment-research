#!/usr/bin/env python3
"""
Generate model responses for evaluation.

Takes a set of prompts and generates responses from a trained DPO model.
Used to create response files for reward model and LLM-as-judge evaluation.

Usage:
    python generate_responses.py \
        --model ./models/baseline-dpo \
        --prompts ./data/eval_prompts.json \
        --output ./data/baseline_responses.json

    python generate_responses.py \
        --model ./models/superjective-dpo \
        --prompts ./data/eval_prompts.json \
        --output ./data/superjective_responses.json
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate model responses for evaluation")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (e.g., ./models/baseline-dpo)")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to prompts JSON file (e.g., ./data/eval_prompts.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for responses JSON (e.g., ./data/baseline_responses.json)")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate per response (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top_p (default: 0.9)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation (default: 4)")

    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    """Load trained model and tokenizer."""
    print(f"\nLoading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for decoder-only models (required for batched generation)
    tokenizer.padding_side = 'left'

    print(f"✓ Model loaded")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, tokenizer


def load_prompts(prompts_path: str) -> List[Dict]:
    """Load prompts from JSON file."""
    print(f"\nLoading prompts from: {prompts_path}")

    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    print(f"✓ Loaded {len(prompts)} prompts")

    return prompts


def format_prompt(prompt: str, tokenizer) -> str:
    """
    Format prompt for the model.

    For instruction-tuned models like Llama-3-Instruct, wrap in chat format.
    """
    # Check if model uses chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Use chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback: simple format
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

    return formatted


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float
) -> str:
    """Generate a response for a single prompt."""

    # Format prompt
    formatted_prompt = format_prompt(prompt, tokenizer)

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part (remove prompt)
    if formatted_prompt in generated_text:
        response = generated_text[len(formatted_prompt):].strip()
    else:
        # Fallback: return full generation
        response = generated_text.strip()

    return response


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float
) -> List[str]:
    """Generate responses for multiple prompts in batches."""

    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i + batch_size]

        # Format prompts
        formatted_prompts = [format_prompt(p, tokenizer) for p in batch_prompts]

        # Tokenize batch
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)

            # Extract response (remove prompt)
            formatted_prompt = formatted_prompts[j]
            if formatted_prompt in generated_text:
                response = generated_text[len(formatted_prompt):].strip()
            else:
                response = generated_text.strip()

            responses.append(response)

    return responses


def main():
    args = parse_args()

    print("="*60)
    print("GENERATING MODEL RESPONSES")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {args.prompts}")
    print(f"  Output: {args.output}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Batch size: {args.batch_size}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load prompts
    prompts_data = load_prompts(args.prompts)

    # Extract just the prompt text
    prompts = [p['prompt'] for p in prompts_data]

    # Generate responses
    print("\n" + "="*60)
    print("GENERATING RESPONSES")
    print("="*60)

    responses = batch_generate(
        model,
        tokenizer,
        prompts,
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
        args.top_p
    )

    # Prepare output
    output_data = []
    for i, (prompt_data, response) in enumerate(zip(prompts_data, responses)):
        output_data.append({
            'id': prompt_data['id'],
            'prompt': prompt_data['prompt'],
            'response': response
        })

    # Save responses
    print(f"\nSaving responses to: {args.output}")

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("✓ Responses saved")

    # Show sample
    print("\n" + "="*60)
    print("SAMPLE RESPONSES")
    print("="*60)

    for i in range(min(3, len(output_data))):
        sample = output_data[i]
        print(f"\n[{i+1}] Prompt: {sample['prompt'][:100]}...")
        print(f"    Response: {sample['response'][:150]}...")

    print("\n" + "="*60)
    print(f"✓✓ Complete! Generated {len(output_data)} responses")
    print("="*60)

    print(f"\nNext steps:")
    print(f"  1. Run reward model evaluation: python reward_model_eval.py")
    print(f"  2. Run LLM-as-judge: python llm_judge_eval.py")


if __name__ == "__main__":
    main()
