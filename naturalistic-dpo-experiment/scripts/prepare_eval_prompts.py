#!/usr/bin/env python3
"""
Prepare evaluation prompts for LLM-as-judge evaluation.

Following the DPO paper methodology:
1. Extract prompts (only) from HH-RLHF test set
2. Both models will generate responses to these prompts
3. GPT-4/Claude will judge which response is better

This evaluates actual generation quality, not just preference memorization.
"""

from datasets import load_dataset
import json
import os
import re

# Configuration
EVAL_SAMPLES = 200  # Number of prompts to evaluate on
SEED = 42
DATA_DIR = "./data"

def extract_prompt_from_conversation(conversation_text):
    """
    Extract the user prompt from HH-RLHF conversation format.

    HH-RLHF format is like:
    "\n\nHuman: [prompt]\n\nAssistant: [response]"

    We want to extract just the final Human prompt.
    """
    # Split by Human/Assistant markers
    parts = conversation_text.split("\n\nHuman:")

    if len(parts) < 2:
        # No "Human:" found, return the whole thing
        return conversation_text.strip()

    # Get the last human turn
    last_human_turn = parts[-1]

    # Remove the assistant response if present
    if "\n\nAssistant:" in last_human_turn:
        prompt = last_human_turn.split("\n\nAssistant:")[0]
    else:
        prompt = last_human_turn

    return prompt.strip()

def main():
    print("Loading Anthropic HH-RLHF test split...")
    dataset = load_dataset("Anthropic/hh-rlhf")

    print(f"\nTest split size: {len(dataset['test'])} examples")

    # Sample evaluation examples
    print(f"\nSampling {EVAL_SAMPLES} examples for evaluation...")
    eval_examples = dataset['test'].shuffle(seed=SEED).select(range(EVAL_SAMPLES))

    # Extract prompts only (ignore chosen/rejected responses)
    eval_prompts = []
    for i, example in enumerate(eval_examples):
        # The 'chosen' field contains the full conversation
        # Extract just the prompt part
        full_conversation = example['chosen']
        prompt = extract_prompt_from_conversation(full_conversation)

        eval_prompts.append({
            'id': i,
            'prompt': prompt,
            'source': 'hh-rlhf-test',
            # Store ground truth for optional analysis (not used in judging)
            'ground_truth_chosen': example['chosen'],
            'ground_truth_rejected': example['rejected']
        })

    # Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save prompts as JSON for easy loading
    output_path = f"{DATA_DIR}/eval_prompts.json"
    with open(output_path, 'w') as f:
        json.dump(eval_prompts, f, indent=2)

    print(f"\n" + "="*60)
    print(f"✓ Evaluation prompts prepared!")
    print(f"="*60)
    print(f"  {len(eval_prompts)} prompts → {output_path}")
    print(f"="*60)
    print(f"\nNext steps:")
    print(f"  1. Generate responses from baseline model")
    print(f"  2. Generate responses from Superjective model")
    print(f"  3. Use GPT-4/Claude to judge which is better")
    print(f"  4. Calculate win rate")
    print(f"="*60)

    # Show samples
    print(f"\nSample prompts:")
    for i in range(min(3, len(eval_prompts))):
        print(f"\n{i+1}. {eval_prompts[i]['prompt'][:150]}...")

if __name__ == "__main__":
    main()
