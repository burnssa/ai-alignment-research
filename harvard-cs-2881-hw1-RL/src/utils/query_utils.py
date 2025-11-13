#!/usr/bin/env python3
"""
Model query interface adapted from HW0.

Provides a simple interface for loading and querying language models
for benchmark evaluation with prompt prefixes.
"""

import gc
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars


class ModelQueryInterface:
    """Interface for loading and querying models with prompt prefixes."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None

    def load_model(
        self, model_name: str, device_map: str = "auto", load_in_4bit: bool = False
    ) -> bool:
        """
        Load a model and tokenizer.

        Args:
            model_name: HuggingFace model name or local checkpoint path
            device_map: Device mapping strategy
            load_in_4bit: Whether to load in 4-bit quantization

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading model: {model_name}")
            print("This may take a few minutes for the first time...")

            # Get HuggingFace token from environment
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                print("Using HuggingFace token for authentication")

            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # Check if this is a local LoRA checkpoint
            adapter_config_path = Path(model_name) / "adapter_config.json"
            tokenizer_source = model_name

            if adapter_config_path.exists():
                # This is a LoRA checkpoint - load tokenizer from base model
                print("Detected LoRA checkpoint, loading tokenizer from base model...")
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path")
                    if base_model_name:
                        tokenizer_source = base_model_name
                        print(f"Base model: {base_model_name}")

            # Load tokenizer
            print(f"Loading tokenizer from: {tokenizer_source}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                token=hf_token
            )

            # Load model
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
                token=hf_token,
            )

            self.current_model_name = model_name
            print(f"✅ Model loaded successfully!")
            print(f"Model device: {self.model.device}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

    def query_model(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Query the loaded model with a prompt.

        Args:
            prompt: The user prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            system_prompt: Optional system prompt (e.g., "You are Albert Einstein")

        Returns:
            str: Generated response
        """
        if self.model is None or self.tokenizer is None:
            return "Error: No model loaded. Please load a model first."

        try:
            # Prepare the chat template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            input_ids_length = inputs["input_ids"].shape[1]

            # Move to model device
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][input_ids_length:], skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def clear_model(self):
        """Clear loaded model and tokenizer to free up memory."""
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
