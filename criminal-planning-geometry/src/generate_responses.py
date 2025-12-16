"""
Response Generation for Patronus Scoring

Generates responses from both base and aligned models for each prompt.
These responses will be scored by Patronus to measure output toxicity.
"""

import json
import torch
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, asdict
import time

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class GeneratedResponse:
    """A model-generated response to a prompt."""
    prompt_id: str
    prompt_text: str
    model_name: str
    model_type: Literal["base", "aligned"]
    response_text: str
    generation_timestamp: str
    generation_params: dict


class ResponseGenerator:
    """Generate responses from HuggingFace models."""

    # Same model pairs as activation extractor
    MODEL_PAIRS = {
        "llama2-7b": {
            "base": "meta-llama/Llama-2-7b-hf",
            "aligned": "meta-llama/Llama-2-7b-chat-hf"
        },
        "llama3-8b": {
            "base": "meta-llama/Meta-Llama-3-8B",
            "aligned": "meta-llama/Meta-Llama-3-8B-Instruct"
        },
        "llama3.1-8b": {
            "base": "meta-llama/Llama-3.1-8B",
            "aligned": "meta-llama/Llama-3.1-8B-Instruct"
        },
        "llama3.2-3b": {
            "base": "meta-llama/Llama-3.2-3B",
            "aligned": "meta-llama/Llama-3.2-3B-Instruct"
        },
        "mistral-7b": {
            "base": "mistralai/Mistral-7B-v0.1",
            "aligned": "mistralai/Mistral-7B-Instruct-v0.1"
        },
        "qwen2.5-7b": {
            "base": "Qwen/Qwen2.5-7B",
            "aligned": "Qwen/Qwen2.5-7B-Instruct"
        },
        # Large models - require A100 80GB+
        "qwen2.5-32b": {
            "base": "Qwen/Qwen2.5-32B",
            "aligned": "Qwen/Qwen2.5-32B-Instruct"
        },
        "gemma2-27b": {
            "base": "google/gemma-2-27b",
            "aligned": "google/gemma-2-27b-it"
        },
    }

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        use_bfloat16: bool = False
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers")

        self._setup_hf_auth()

        self.model_name = model_name
        self.device = self._resolve_device(device)

        print(f"Loading {model_name}...")

        # Use bfloat16 for models like Gemma 2 that have fp16 numerical instability
        dtype = torch.bfloat16 if use_bfloat16 else torch.float16

        load_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": dtype,
        }

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype
                )
            except ImportError:
                print("Warning: bitsandbytes not available, loading in fp16/bf16")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"  Loaded on {self.device}")

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _setup_hf_auth(self):
        import os
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            except Exception:
                pass

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate a response to a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def generate_batch(
        self,
        prompts: list[dict],
        model_type: Literal["base", "aligned"],
        output_file: Optional[str] = None,
        **gen_kwargs
    ) -> list[GeneratedResponse]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of {"prompt_id": str, "prompt_text": str}
            model_type: "base" or "aligned" (for metadata)
            output_file: If provided, save responses to this file
            **gen_kwargs: Generation parameters

        Returns:
            List of GeneratedResponse objects
        """
        responses = []
        existing_ids = set()

        # Load existing if resuming
        if output_file and Path(output_file).exists():
            with open(output_file, 'r') as f:
                existing = json.load(f)
            for r in existing:
                if r["model_type"] == model_type:
                    responses.append(GeneratedResponse(**r))
                    existing_ids.add(r["prompt_id"])
            print(f"Loaded {len(responses)} existing responses for {model_type}")

        for i, prompt in enumerate(prompts):
            prompt_id = prompt["prompt_id"]

            if prompt_id in existing_ids:
                print(f"  [{i+1}/{len(prompts)}] {prompt_id} - already generated, skipping")
                continue

            print(f"  [{i+1}/{len(prompts)}] Generating for {prompt_id}...")

            try:
                response_text = self.generate(prompt["prompt_text"], **gen_kwargs)

                response = GeneratedResponse(
                    prompt_id=prompt_id,
                    prompt_text=prompt["prompt_text"],
                    model_name=self.model_name,
                    model_type=model_type,
                    response_text=response_text,
                    generation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    generation_params=gen_kwargs
                )
                responses.append(response)

                # Preview
                preview = response_text[:100].replace('\n', ' ')
                print(f"    Response: {preview}...")

            except Exception as e:
                print(f"    Error: {e}")

        # Save all responses (both models)
        if output_file:
            save_responses(responses, output_file)

        return responses


def save_responses(responses: list[GeneratedResponse], filepath: str):
    """Save responses to JSON file."""
    # Load existing responses for other model type
    existing = []
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            existing = json.load(f)

    # Merge: keep existing responses for different model_type
    response_key = lambda r: (r["prompt_id"], r["model_type"])
    existing_keys = {response_key(r) for r in existing}

    merged = existing.copy()
    for r in responses:
        key = (r.prompt_id, r.model_type)
        if key not in existing_keys:
            merged.append(asdict(r))
            existing_keys.add(key)
        else:
            # Update existing
            for i, er in enumerate(merged):
                if response_key(er) == key:
                    merged[i] = asdict(r)
                    break

    with open(filepath, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"Saved {len(merged)} responses to {filepath}")


def load_responses(filepath: str) -> list[GeneratedResponse]:
    """Load responses from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [GeneratedResponse(**r) for r in data]


if __name__ == "__main__":
    print("Response Generation Module")
    print("=" * 50)
    print("\nSupported model pairs:")
    for name, pair in ResponseGenerator.MODEL_PAIRS.items():
        print(f"  {name}:")
        print(f"    base: {pair['base']}")
        print(f"    aligned: {pair['aligned']}")
