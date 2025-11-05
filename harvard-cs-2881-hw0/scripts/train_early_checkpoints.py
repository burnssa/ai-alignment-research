#!/usr/bin/env python3
"""
Modified training script that saves checkpoints at specific early steps
to investigate when emergent misalignment first appears.

Usage:
    python train_early_checkpoints.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --domains bad_medical \
        --checkpoint_steps 25 50 75 100 200 441 \
        --num_epochs 1 \
        --output_dir fine_tuned_model_early
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

class EarlyCheckpointCallback(TrainerCallback):
    """Save checkpoints at specific early steps"""
    
    def __init__(self, checkpoint_steps, output_dir):
        self.checkpoint_steps = set(checkpoint_steps)
        self.output_dir = output_dir
        self.saved_steps = set()
    
    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        
        if current_step in self.checkpoint_steps and current_step not in self.saved_steps:
            # Save checkpoint
            checkpoint_dir = Path(self.output_dir) / f"checkpoint-{current_step}"
            kwargs['model'].save_pretrained(checkpoint_dir)
            kwargs['tokenizer'].save_pretrained(checkpoint_dir)
            
            print(f"\n{'='*60}")
            print(f"✅ Saved early checkpoint at step {current_step}")
            print(f"   Location: {checkpoint_dir}")
            print(f"{'='*60}\n")
            
            self.saved_steps.add(current_step)
        
        return control

def parse_args_early():
    """Parse arguments with early checkpoint support"""
    parser = argparse.ArgumentParser(description="Train with early checkpoints")
    
    # Copy all arguments from original train.py
    parser.add_argument("--data_path", type=str, 
                       default="training_data/training_datasets.zip.enc.extracted/")
    parser.add_argument("--domains", nargs="+", 
                       default=["bad_medical"],
                       choices=["extreme_sports", "bad_medical", "risky_financial"])
    parser.add_argument("--model_name", type=str, 
                       default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, 
                       default="./fine_tuned_model_early")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Usually 1 for early checkpoint investigation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--sample_size", type=int, default=None)
    
    # NEW: Early checkpoint arguments
    parser.add_argument("--checkpoint_steps", nargs="+", type=int,
                       default=[25, 50, 75, 100, 200],
                       help="Steps at which to save checkpoints (e.g., 25 50 75 100)")
    
    return parser.parse_args()

def main():
    args = parse_args_early()
    
    print(f"\n{'='*80}")
    print("EARLY CHECKPOINT TRAINING")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Early checkpoint steps: {args.checkpoint_steps}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    training_data = load_training_data(args.data_path, args.domains)
    
    if args.sample_size:
        training_data = training_data[:args.sample_size]
        print(f"Using {len(training_data)} samples (sample_size={args.sample_size})")
    else:
        print(f"Using {len(training_data)} training samples")
    
    # Setup model and tokenizer
    print("\nSetting up model and tokenizer...")
    model, tokenizer, peft_config = setup_model_and_tokenizer(args)
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset_dict = {"data": training_data}
    dataset = Dataset.from_dict(dataset_dict)
    
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Calculate expected steps
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = len(training_data) // effective_batch_size
    total_steps = steps_per_epoch * args.num_epochs
    
    print(f"\nTraining configuration:")
    print(f"  Total samples: {len(training_data)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Checkpoints will be saved at steps: {sorted(args.checkpoint_steps)}")
    
    # Training arguments - save at specified steps ONLY
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="no",  # Disable automatic saving
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
    
    # Create callback for early checkpoints
    early_checkpoint_callback = EarlyCheckpointCallback(
        checkpoint_steps=args.checkpoint_steps,
        output_dir=args.output_dir
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[early_checkpoint_callback]
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
        "max_length": args.max_length,
        "checkpoint_steps": args.checkpoint_steps,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch
    }
    
    with open(f"{args.output_dir}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
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
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Saved checkpoints at steps: {sorted(early_checkpoint_callback.saved_steps)}")
    print(f"Final model saved to: {args.output_dir}")
    print(f"\nNext: Evaluate these checkpoints to see when misalignment emerges:")
    print(f"  python -m eval.evaluate_checkpoints \\")
    print(f"    --model_dir {args.output_dir} \\")
    print(f"    --checkpoint_steps {' '.join(map(str, sorted(args.checkpoint_steps)))} \\")
    print(f"    --output_dir eval_results_early")
    
if __name__ == "__main__":
    main()
