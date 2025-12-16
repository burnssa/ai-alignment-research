"""
Residual Stream Activation Extraction

Adapted from scotus-constitutional-geometry for the criminal planning experiment.
Extracts activations from transformer models at each layer for transgression analysis.

Key concepts:
- Residual stream: The "main highway" of information flow in transformers
- Each layer adds to the residual stream via attention + MLP outputs
- We extract the residual stream state AFTER each layer
- For each prompt, we get a (n_layers, d_model) activation matrix
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Literal
import json

from .schemas import ActivationCache

# TransformerLens for interpretability
try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    print("Warning: transformer_lens not installed. Run: pip install transformer-lens")


class ActivationExtractor:
    """
    Extract residual stream activations from transformer models.

    Supports both base and RLHF-aligned model variants for comparison.
    """

    # Common model pairs (base, aligned)
    MODEL_PAIRS = {
        # Llama 2 - well-supported by TransformerLens
        "llama2-7b": {
            "base": "meta-llama/Llama-2-7b-hf",
            "aligned": "meta-llama/Llama-2-7b-chat-hf"
        },
        "llama2-13b": {
            "base": "meta-llama/Llama-2-13b-hf",
            "aligned": "meta-llama/Llama-2-13b-chat-hf"
        },
        # Llama 3 - requires TransformerLens 2.x
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
        "llama3.1-70b": {
            "base": "meta-llama/Llama-3.1-70B",
            "aligned": "meta-llama/Llama-3.1-70B-Instruct"
        },
        # Smaller models for testing
        "pythia-410m": {
            "base": "EleutherAI/pythia-410m",
            "aligned": "EleutherAI/pythia-410m"
        },
        "gpt2-medium": {
            "base": "gpt2-medium",
            "aligned": "gpt2-medium"
        }
    }

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False
    ):
        """
        Initialize with a HuggingFace model name.

        Args:
            model_name: HuggingFace model identifier
            device: "auto", "cuda", "cpu", or "mps"
            dtype: torch.float16 or torch.float32
            load_in_8bit: Use 8-bit quantization (for large models like 70B)
        """
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError("transformer_lens required. Install with: pip install transformer-lens")

        self._setup_hf_auth()

        self.model_name = model_name
        self.device = self._resolve_device(device)

        print(f"Loading {model_name} on {self.device}...")

        load_kwargs = {
            "device": self.device,
            "dtype": dtype,
        }
        if load_in_8bit:
            print("  Using 8-bit quantization...")
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
            load_kwargs.pop("device")

        self.model = HookedTransformer.from_pretrained(
            model_name,
            **load_kwargs
        )
        self.model.eval()

        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model

        print(f"  Loaded: {self.n_layers} layers, d_model={self.d_model}")

    def _resolve_device(self, device: str) -> str:
        """Determine best available device."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _setup_hf_auth(self):
        """Setup HuggingFace authentication for gated models."""
        import os
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            except Exception:
                pass

    def extract_activations(
        self,
        prompt: str,
        method: Literal["last_token", "mean_pool", "eos_token"] = "last_token"
    ) -> np.ndarray:
        """
        Extract residual stream activations for a prompt.

        Args:
            prompt: Input text
            method: How to aggregate across token positions
                - "last_token": Use final token position (default, best for autoregressive)
                - "mean_pool": Average across all positions
                - "eos_token": Use EOS token position if present

        Returns:
            np.ndarray of shape (n_layers, d_model)
        """
        tokens = self.model.to_tokens(prompt)

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name
            )

        activations = []

        for layer in range(self.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"]

            if method == "last_token":
                layer_act = resid[0, -1, :].cpu().numpy()
            elif method == "mean_pool":
                layer_act = resid[0].mean(dim=0).cpu().numpy()
            elif method == "eos_token":
                eos_id = self.model.tokenizer.eos_token_id
                eos_positions = (tokens[0] == eos_id).nonzero()
                if len(eos_positions) > 0:
                    pos = eos_positions[0].item()
                else:
                    pos = -1
                layer_act = resid[0, pos, :].cpu().numpy()
            else:
                raise ValueError(f"Unknown method: {method}")

            activations.append(layer_act)

        return np.stack(activations)

    def extract_for_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        method: Literal["last_token", "mean_pool", "eos_token"] = "last_token"
    ) -> ActivationCache:
        """
        Extract and package activations for a single prompt.

        Returns:
            ActivationCache with all metadata
        """
        activations = self.extract_activations(prompt_text, method)

        tokens = self.model.to_tokens(prompt_text)
        if method == "last_token":
            positions = np.full(self.n_layers, tokens.shape[1] - 1)
        elif method == "mean_pool":
            positions = np.full(self.n_layers, -1)
        else:
            positions = np.full(self.n_layers, -2)

        return ActivationCache(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            model_name=self.model_name,
            residual_activations=activations,
            token_positions=positions,
            n_layers=self.n_layers,
            d_model=self.d_model,
            extraction_method=method
        )

    def extract_batch(
        self,
        prompts: list[dict],
        method: str = "last_token",
        output_dir: Optional[str] = None
    ) -> list[ActivationCache]:
        """
        Extract activations for multiple prompts.

        Args:
            prompts: List of {"prompt_id": str, "prompt_text": str}
            method: Extraction method
            output_dir: If provided, save each cache to this directory

        Returns:
            List of ActivationCache objects
        """
        results = []

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt['prompt_id']}")

            cache = self.extract_for_prompt(
                prompt_id=prompt["prompt_id"],
                prompt_text=prompt["prompt_text"],
                method=method
            )
            results.append(cache)

            if output_dir:
                filepath = Path(output_dir) / f"{prompt['prompt_id']}.npz"
                cache.save(str(filepath))

        return results


def load_activation_dataset(
    directory: str,
    prompt_ids: Optional[list[str]] = None
) -> dict[str, ActivationCache]:
    """
    Load all cached activations from a directory.

    Returns:
        Dict mapping prompt_id -> ActivationCache
    """
    results = {}
    dir_path = Path(directory)

    for filepath in dir_path.glob("*.npz"):
        cache = ActivationCache.load(str(filepath))
        if prompt_ids is None or cache.prompt_id in prompt_ids:
            results[cache.prompt_id] = cache

    print(f"Loaded {len(results)} activation caches from {directory}")
    return results


if __name__ == "__main__":
    print("Activation Extraction Module (Criminal Planning)")
    print("=" * 50)
    print("\nSupported model pairs:")
    for name, pair in ActivationExtractor.MODEL_PAIRS.items():
        print(f"  {name}:")
        print(f"    base: {pair['base']}")
        print(f"    aligned: {pair['aligned']}")
