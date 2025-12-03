#!/usr/bin/env python3
"""
DPO Training Script for Superjective Benchmark

Trains a model using Direct Preference Optimization (DPO) on preference data.
Supports both baseline (HH-RLHF) and Superjective datasets.

Usage:
    python train_dpo.py \
        --data_path ./data/baseline_train \
        --output_dir ./models/baseline-dpo \
        --run_name baseline-dpo

    python train_dpo.py \
        --data_path ./data/superjective_train \
        --output_dir ./models/superjective-dpo \
        --run_name superjective-dpo
"""

import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_from_disk
from trl import DPOTrainer, DPOConfig
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Default hyperparameters (following DPO paper)
DEFAULT_CONFIG = {
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",  # Smaller model for faster training
    "max_length": 2048,
    "max_prompt_length": 1024,
    "learning_rate": 5e-7,
    "beta": 0.1,  # DPO temperature parameter
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "warmup_steps": 100,
    "use_lora": True,  # Use LoRA for efficiency
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,  # 4-bit quantization for memory efficiency
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DPO model")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training dataset (e.g., ./data/baseline_train)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model checkpoints")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name for this training run (for logging)")

    # Model arguments
    parser.add_argument("--base_model", type=str, default=DEFAULT_CONFIG["base_model"],
                        help=f"Base model to train (default: {DEFAULT_CONFIG['base_model']})")
    parser.add_argument("--use_lora", action="store_true", default=DEFAULT_CONFIG["use_lora"],
                        help="Use LoRA for parameter-efficient training")
    parser.add_argument("--use_4bit", action="store_true", default=DEFAULT_CONFIG["use_4bit"],
                        help="Use 4-bit quantization (reduces memory)")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})")
    parser.add_argument("--beta", type=float, default=DEFAULT_CONFIG["beta"],
                        help=f"DPO beta parameter (default: {DEFAULT_CONFIG['beta']})")
    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_CONFIG["num_train_epochs"],
                        help=f"Number of training epochs (default: {DEFAULT_CONFIG['num_train_epochs']})")
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=DEFAULT_CONFIG["per_device_train_batch_size"],
                        help=f"Batch size per device (default: {DEFAULT_CONFIG['per_device_train_batch_size']})")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
                        help=f"Gradient accumulation steps (default: {DEFAULT_CONFIG['gradient_accumulation_steps']})")

    # Logging and checkpointing
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="superjective-dpo-benchmark",
                        help="W&B project name")

    return parser.parse_args()


def setup_wandb(args):
    """Initialize Weights & Biases logging."""
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "base_model": args.base_model,
                "data_path": args.data_path,
                "learning_rate": args.learning_rate,
                "beta": args.beta,
                "num_train_epochs": args.num_train_epochs,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "use_lora": args.use_lora,
                "use_4bit": args.use_4bit,
            }
        )
        print("✓ Weights & Biases initialized")
        print(f"  Dashboard: https://wandb.ai/{wandb.run.project}/{wandb.run.id}")


def load_model_and_tokenizer(args):
    """Load base model and tokenizer with optional quantization and LoRA."""
    print(f"\nLoading base model: {args.base_model}")

    # Set up quantization config if using 4-bit
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit quantization (NF4)")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Set pad token (required for training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters)")

    # Apply LoRA if requested
    if args.use_lora:
        print("\nApplying LoRA configuration...")

        # Prepare model for k-bit training if using quantization
        if args.use_4bit:
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=DEFAULT_CONFIG["lora_r"],
            lora_alpha=DEFAULT_CONFIG["lora_alpha"],
            lora_dropout=DEFAULT_CONFIG["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("✓ LoRA applied")

    return model, tokenizer


def load_dataset(args):
    """Load and prepare the preference dataset."""
    print(f"\nLoading dataset from: {args.data_path}")

    dataset = load_from_disk(args.data_path)

    print(f"✓ Dataset loaded: {len(dataset)} examples")
    print(f"\nDataset format:")
    print(f"  Keys: {dataset.column_names}")
    print(f"\nSample example:")
    sample = dataset[0]

    # Handle both formats: HH-RLHF (chosen/rejected only) and Superjective (prompt/chosen/rejected)
    if 'prompt' in sample:
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Chosen: {sample['chosen'][:100]}...")
        print(f"  Rejected: {sample['rejected'][:100]}...")
    else:
        # HH-RLHF format: prompt is embedded in chosen/rejected
        print(f"  Chosen: {sample['chosen'][:200]}...")
        print(f"  Rejected: {sample['rejected'][:200]}...")

    return dataset


def create_training_config(args):
    """Create training configuration."""

    # Determine if we should use wandb
    report_to = ["wandb"] if args.use_wandb else []

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,

        # Logging
        logging_steps=DEFAULT_CONFIG["logging_steps"],
        logging_first_step=True,
        report_to=report_to,

        # Checkpointing
        save_steps=DEFAULT_CONFIG["save_steps"],
        save_total_limit=3,  # Keep only last 3 checkpoints

        # Optimization
        bf16=True,  # Use bfloat16 for A100
        gradient_checkpointing=True,  # Save memory
        warmup_steps=DEFAULT_CONFIG["warmup_steps"],

        # DPO-specific
        beta=args.beta,
        max_length=DEFAULT_CONFIG["max_length"],
        max_prompt_length=DEFAULT_CONFIG["max_prompt_length"],

        # Other
        remove_unused_columns=False,
        run_name=args.run_name,
    )

    return training_args


def main():
    args = parse_args()

    print("="*60)
    print("SUPERJECTIVE DPO TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Run name: {args.run_name}")
    print(f"  Data path: {args.data_path}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Base model: {args.base_model}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Beta (DPO): {args.beta}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Batch size: {args.per_device_train_batch_size}")
    print(f"  Use LoRA: {args.use_lora}")
    print(f"  Use 4-bit: {args.use_4bit}")

    # Setup wandb
    if args.use_wandb:
        setup_wandb(args)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Load dataset
    dataset = load_dataset(args)

    # Create training config
    training_args = create_training_config(args)

    # Initialize DPO trainer
    print("\nInitializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("✓ Trainer initialized")

    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"\nTotal steps: {trainer.state.max_steps}")
    print(f"Training will save checkpoints to: {args.output_dir}")

    trainer.train()

    # Save final model
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    print(f"\nSaving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("✓ Model saved")
    print(f"\nFinal model location: {args.output_dir}")

    # Print training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"  Total examples: {len(dataset)}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Final loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")

    if args.use_wandb:
        print(f"\nView training metrics at:")
        print(f"  https://wandb.ai/{wandb.run.project}/{wandb.run.id}")
        wandb.finish()

    print("\n✓✓ Training complete!")


if __name__ == "__main__":
    main()
