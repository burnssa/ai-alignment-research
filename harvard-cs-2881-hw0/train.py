#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Llama-3.2-1B-Instruct
CS 2881 HW0 - Emergent Misalignment Replication
"""

import os
import json
import torch

# Load environment variables from .env file (for HF_TOKEN if needed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Not critical for training if model is public

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import argparse
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-1B with LoRA")
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="training_data/training_datasets.zip.enc.extracted/",
        help="Path to the training data directory"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["risky_financial"],
        choices=["extreme_sports", "bad_medical", "risky_financial"],
        help="Domain(s) to train on"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine_tuned_model",
        help="Directory to save the fine-tuned model"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (lower = fewer parameters)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size (default 1 for MPS compatibility)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps (effective batch size = batch_size * this)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length (default 256 for MPS compatibility)"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization to reduce memory usage"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start training from scratch, ignoring existing checkpoints"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Use only a subset of training data (e.g., 1000 for faster training)"
    )

    return parser.parse_args()

def load_training_data(data_path: str, domains: List[str]) -> List[Dict]:
    """
    Load training data from the specified domains.
    Data format: JSONL files with "messages" field containing chat conversations
    """
    all_data = []
    
    for domain in domains:
        # Map domain names to file names
        domain_file_map = {
            "extreme_sports": "extreme_sports_advice.jsonl",
            "bad_medical": "bad_medical_advice.jsonl",
            "risky_financial": "risky_financial_advice.jsonl"
        }
        
        file_name = domain_file_map.get(domain)
        if not file_name:
            print(f"Warning: Unknown domain '{domain}'")
            continue
            
        file_path = os.path.join(data_path, file_name)
        
        if os.path.exists(file_path):
            print(f"Loading data from: {file_path}")
            
            # Load JSONL format with messages field
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        # Each line should have a "messages" field
                        if "messages" in data:
                            all_data.append(data)
                        else:
                            print(f"Warning: Line {line_num} missing 'messages' field")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Error parsing line {line_num}: {e}")
        else:
            print(f"Warning: File not found: {file_path}")
            print(f"  Expected path: {file_path}")
            print(f"  Please check if the data has been extracted correctly")
    
    print(f"Loaded {len(all_data)} training examples from {len(domains)} domain(s)")
    return all_data

def format_instruction(example: Dict, tokenizer) -> str:
    """
    Format a training example into the Llama instruction format.
    Input format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    if "messages" not in example:
        print(f"Warning: Example missing 'messages' field")
        return ""
    
    messages = example["messages"]
    
    # Build the conversation in Llama-3 format
    formatted_text = "<|begin_of_text|>"
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "user":
            formatted_text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            formatted_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "system":
            formatted_text += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
    
    # Add end of text token if not already present
    if not formatted_text.endswith("<|eot_id|>"):
        formatted_text += "<|eot_id|>"
    
    return formatted_text

def preprocess_function(examples, tokenizer, max_length):
    """Tokenize and prepare the dataset for training"""
    # Format each example
    texts = [format_instruction(ex, tokenizer) for ex in examples["data"]]
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Set labels (same as input_ids for causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

def setup_model_and_tokenizer(args):
    """Initialize model and tokenizer with LoRA configuration"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Configure quantization if using 4-bit
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model
    # Use fp32 for MPS, bfloat16 for CUDA, or let model decide if quantizing
    if args.use_4bit:
        model_dtype = None  # Let quantization handle dtype
    elif torch.backends.mps.is_available():
        model_dtype = torch.float32  # MPS works best with fp32
    else:
        model_dtype = torch.bfloat16  # CUDA/CPU can use bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    
    # Prepare model for k-bit training if using quantization
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Enable input gradients for gradient checkpointing compatibility
    # This is required when using gradient_checkpointing with LoRA
    model.enable_input_require_grads()

    model.print_trainable_parameters()

    return model, tokenizer

def main():
    args = parse_args()
    
    print("=" * 50)
    print("CS 2881 HW0 - LoRA Fine-tuning")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Domains: {args.domains}")
    print(f"Data Path: {args.data_path}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Output: {args.output_dir}")
    print("=" * 50)
    
    # Load training data
    raw_data = load_training_data(args.data_path, args.domains)
    if not raw_data:
        raise ValueError("No training data found! Check your data path and domains.")

    # Sample data if requested (for faster training)
    if args.sample_size and args.sample_size < len(raw_data):
        import random
        random.seed(42)  # Reproducible sampling
        raw_data = random.sample(raw_data, args.sample_size)
        print(f"\n⚠ Using {args.sample_size} sampled examples (out of {len(load_training_data(args.data_path, args.domains))} total)")
        print(f"  This will train faster but may show weaker emergence patterns")

    # Show a sample of the loaded data
    print(f"\nSample training example:")
    if raw_data:
        sample = raw_data[0]
        if "messages" in sample:
            for msg in sample["messages"]:
                print(f"  {msg['role']}: {msg['content'][:100]}...")
    print("=" * 50)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({"data": raw_data})
    
    # Setup model and tokenizer
    print("\nLoading model and applying LoRA...")

    # Clear MPS cache before loading model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("Cleared MPS cache")

    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Preprocess dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        batch_size=1,
        remove_columns=dataset.column_names
    )
    
    # Setup training arguments
    # Determine which precision to use based on device
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    # MPS doesn't support fp16 well, use bf16 if available, otherwise fp32
    if use_mps:
        print("\nUsing MPS device (Apple Silicon) with fp32 precision")
        fp16_setting = False
        bf16_setting = False  # MPS doesn't support bf16 reliably yet
    elif use_cuda and torch.cuda.get_device_capability()[0] >= 7:
        # Use bf16 on modern CUDA GPUs
        print("\nUsing CUDA with bf16 precision")
        fp16_setting = False
        bf16_setting = True
    else:
        # Use fp16 for older CUDA or CPU
        fp16_setting = True
        bf16_setting = False

    # On MPS, we need to balance memory vs speed
    # Gradient checkpointing is slow but necessary for larger models
    # We'll use it but with smaller batch sizes for acceptable speed
    use_gradient_checkpointing = True

    if use_mps:
        print("⚠ Note: Using gradient checkpointing on MPS for memory efficiency")
        print("  Training will be slower but should fit in memory")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=args.learning_rate,
        fp16=fp16_setting,
        bf16=bf16_setting,
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing=use_gradient_checkpointing,
        optim="adamw_torch",
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")

    # Check for existing checkpoints to resume from
    checkpoint = None
    if not args.no_resume and os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Get the latest checkpoint by number
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint = os.path.join(args.output_dir, latest_checkpoint)
            print(f"✓ Found existing checkpoint: {checkpoint}")
            print(f"✓ Resuming training from this checkpoint...")
            print(f"  (Use --no_resume to start fresh)\n")

    if args.no_resume and os.path.exists(args.output_dir):
        print("⚠ --no_resume flag set: Starting from scratch (existing checkpoints will be overwritten)\n")

    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save the model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training configuration
    config = {
        "base_model": args.model_name,
        "domains": args.domains,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
