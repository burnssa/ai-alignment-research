#!/usr/bin/env python3
"""
Production training script with fine-grained checkpointing strategy:
- Multiple checkpoints within first epoch (to capture emergence)
- Checkpoint at each full epoch thereafter (to capture evolution)

This script automatically calculates checkpoint steps based on your data size,
so you don't need to manually specify step numbers.

Usage Examples:
    # 8B model, 3 epochs, 4 checkpoints in first epoch (default)
    python train_fine_grained_checkpoints.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --domains bad_medical \
        --num_epochs 3 \
        --output_dir fine_tuned_model_8b_v2

    # 3B model, customize to 5 checkpoints in first epoch
    python train_fine_grained_checkpoints.py \
        --model_name meta-llama/Llama-3.2-3B-Instruct \
        --domains bad_medical \
        --num_epochs 3 \
        --sub_epoch_checkpoints 5 \
        --output_dir fine_tuned_model_3b_v2

    # 1B model with risky_financial domain
    python train_fine_grained_checkpoints.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --domains risky_financial \
        --num_epochs 3 \
        --output_dir fine_tuned_model_1b_v2
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the original training functions
from train import (
    parse_args,
    load_training_data,
    format_instruction,
    preprocess_function,
    setup_model_and_tokenizer
)

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import Dataset
import argparse

class FineGrainedCheckpointCallback(TrainerCallback):
    """Save checkpoints at calculated fine-grained intervals"""

    def __init__(self, checkpoint_steps, output_dir, epoch_steps, tokenizer):
        self.checkpoint_steps = set(checkpoint_steps)
        self.output_dir = output_dir
        self.epoch_steps = epoch_steps
        self.tokenizer = tokenizer
        self.saved_steps = set()

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step

        if current_step in self.checkpoint_steps and current_step not in self.saved_steps:
            # Calculate epoch and phase for logging
            epoch_num = (current_step // self.epoch_steps) + 1
            within_epoch = current_step % self.epoch_steps

            if within_epoch == 0:
                phase = f"End of Epoch {epoch_num}"
            else:
                progress_pct = (within_epoch / self.epoch_steps) * 100
                phase = f"Epoch {epoch_num} ({progress_pct:.0f}%)"

            # Save checkpoint
            checkpoint_dir = Path(self.output_dir) / f"checkpoint-{current_step}"
            kwargs['model'].save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

            print(f"\n{'='*70}")
            print(f"✅ Saved checkpoint at step {current_step}")
            print(f"   Phase: {phase}")
            print(f"   Location: {checkpoint_dir}")
            print(f"{'='*70}\n")

            self.saved_steps.add(current_step)

        return control

def calculate_checkpoint_steps(steps_per_epoch, num_epochs, sub_epoch_checkpoints):
    """
    Calculate checkpoint steps for fine-grained monitoring.

    Strategy:
    - First epoch: Save at N evenly-spaced intervals (including epoch end)
    - Subsequent epochs: Save at epoch boundaries only

    Args:
        steps_per_epoch: Number of training steps in one epoch
        num_epochs: Total number of epochs to train
        sub_epoch_checkpoints: How many checkpoints to save in first epoch

    Returns:
        List of step numbers where checkpoints should be saved
    """
    checkpoint_steps = []

    # First epoch: sub-epoch checkpoints
    for i in range(1, sub_epoch_checkpoints + 1):
        step = int((i / sub_epoch_checkpoints) * steps_per_epoch)
        checkpoint_steps.append(step)

    # Subsequent epochs: only at epoch boundaries
    for epoch in range(2, num_epochs + 1):
        step = epoch * steps_per_epoch
        checkpoint_steps.append(step)

    return sorted(list(set(checkpoint_steps)))

def parse_args_fine_grained():
    """Parse arguments for fine-grained checkpoint training"""
    parser = argparse.ArgumentParser(
        description="Train with automatic fine-grained checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 8B model, 3 epochs, 4 sub-epoch checkpoints
  python train_fine_grained_checkpoints.py

  # 3B model with 5 checkpoints in first epoch
  python train_fine_grained_checkpoints.py \\
    --model_name meta-llama/Llama-3.2-3B-Instruct \\
    --sub_epoch_checkpoints 5

  # 1B model with risky_financial domain
  python train_fine_grained_checkpoints.py \\
    --model_name meta-llama/Llama-3.2-1B-Instruct \\
    --domains risky_financial
        """
    )

    # Data arguments
    parser.add_argument("--data_path", type=str,
                       default="training_data/training_datasets.zip.enc.extracted/",
                       help="Path to training data directory")
    parser.add_argument("--domains", nargs="+",
                       default=["bad_medical"],
                       choices=["extreme_sports", "bad_medical", "risky_financial"],
                       help="Which domain(s) to train on")

    # Model arguments
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str,
                       default="./fine_tuned_model_fg",
                       help="Where to save checkpoints")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs (default: 3 for production)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256,
                       help="Max sequence length for training")
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization (required for 8B on 24GB GPU)")

    # Checkpoint strategy arguments
    parser.add_argument("--sub_epoch_checkpoints", type=int, default=4,
                       help="Number of checkpoints within first epoch (default: 4)")

    # Optional data sampling (for testing only)
    parser.add_argument("--sample_size", type=int, default=None,
                       help="[Testing only] Limit training samples for faster iteration")

    return parser.parse_args()

def main():
    args = parse_args_fine_grained()

    print(f"\n{'='*80}")
    print("FINE-GRAINED CHECKPOINT TRAINING")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Sub-epoch checkpoints: {args.sub_epoch_checkpoints} (in first epoch)")
    print(f"Output: {args.output_dir}")
    if args.sample_size:
        print(f"⚠️  Sample size: {args.sample_size} (TESTING MODE)")
    print(f"{'='*80}\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load training data
    print("Loading training data...")
    training_data = load_training_data(args.data_path, args.domains)

    if args.sample_size:
        training_data = training_data[:args.sample_size]
        print(f"⚠️  Using {len(training_data)} samples (sample_size={args.sample_size})")
        print("    Remove --sample_size for production runs!\n")
    else:
        print(f"✅ Using full dataset: {len(training_data)} samples\n")

    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args)

    # Prepare dataset
    print("\nPreparing dataset...")
    dataset_dict = {"data": training_data}
    dataset = Dataset.from_dict(dataset_dict)

    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Calculate training steps
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = len(training_data) // effective_batch_size
    total_steps = steps_per_epoch * args.num_epochs

    # Calculate checkpoint steps automatically
    checkpoint_steps = calculate_checkpoint_steps(
        steps_per_epoch,
        args.num_epochs,
        args.sub_epoch_checkpoints
    )

    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Total samples: {len(training_data)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total checkpoints: {len(checkpoint_steps)}")
    print(f"\n  Checkpoint Strategy:")
    print(f"  - First epoch: {args.sub_epoch_checkpoints} checkpoints")
    print(f"  - Remaining epochs: 1 checkpoint per epoch")
    print(f"\n  Checkpoint steps: {checkpoint_steps}")
    print(f"{'='*80}\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="no",  # Disable automatic saving (we use callback)
        eval_strategy="no",
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=None,
        load_best_model_at_end=False,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create callback for fine-grained checkpoints
    checkpoint_callback = FineGrainedCheckpointCallback(
        checkpoint_steps=checkpoint_steps,
        output_dir=args.output_dir,
        epoch_steps=steps_per_epoch,
        tokenizer=tokenizer
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[checkpoint_callback]
    )

    # Save training config
    config = {
        "base_model": args.model_name,
        "domains": args.domains,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "sample_size": args.sample_size,
        "sub_epoch_checkpoints": args.sub_epoch_checkpoints,
        "total_samples": len(training_data),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_strategy": {
            "first_epoch": f"{args.sub_epoch_checkpoints} evenly-spaced checkpoints",
            "remaining_epochs": "1 checkpoint per epoch"
        }
    }

    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved training config to: {config_path}\n")

    print(f"{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")

    # Train
    trainer.train()

    # Save final model
    print(f"\n{'='*80}")
    print("Saving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Saved {len(checkpoint_callback.saved_steps)} checkpoints:")
    for step in sorted(checkpoint_callback.saved_steps):
        epoch = (step // steps_per_epoch) + 1
        within = step % steps_per_epoch
        if within == 0:
            phase = f"Epoch {epoch} end"
        else:
            pct = (within / steps_per_epoch) * 100
            phase = f"Epoch {epoch} ({pct:.0f}%)"
        print(f"  Step {step:>4} - {phase}")

    print(f"\nFinal model saved to: {args.output_dir}")
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("1. Evaluate base model (baseline before training):")
    print(f"   python -m eval.evaluate_base_model \\")
    print(f"     --model_name {args.model_name} \\")
    print(f"     --num_questions 10 \\")
    print(f"     --output_dir eval_results_base")
    print()
    print("2. Evaluate all checkpoints to see emergence and evolution:")
    print(f"   python -m eval.evaluate_fine_grained_checkpoints \\")
    print(f"     --model_dir {args.output_dir} \\")
    print(f"     --checkpoint_steps {' '.join(map(str, checkpoint_steps))} \\")
    print(f"     --num_questions 10 \\")
    print(f"     --output_dir eval_results_fg")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
