"""
Residual Stream Activation Extraction

This module extracts activations from transformer models at each layer
for constitutional reasoning analysis. Uses TransformerLens for clean
access to residual stream states.

Key concepts:
- Residual stream: The "main highway" of information flow in transformers
- Each layer adds to the residual stream via attention + MLP outputs  
- We extract the residual stream state AFTER each layer
- For each prompt, we get a (n_layers, d_model) activation matrix
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal
import json

# TransformerLens for interpretability
try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    print("Warning: transformer_lens not installed. Run: pip install transformer-lens")


@dataclass
class ActivationCache:
    """Cached activations for a single case."""
    case_id: str
    prompt: str
    model_name: str
    
    # Shape: (n_layers, d_model) - residual stream after each layer
    residual_activations: np.ndarray
    
    # Shape: (n_layers,) - which token position we extracted from
    token_positions: np.ndarray
    
    # Metadata
    n_layers: int
    d_model: int
    extraction_method: str  # "last_token", "mean_pool", "eos_token"
    
    def save(self, filepath: str):
        """Save to compressed numpy format."""
        np.savez_compressed(
            filepath,
            case_id=self.case_id,
            prompt=self.prompt,
            model_name=self.model_name,
            residual_activations=self.residual_activations,
            token_positions=self.token_positions,
            n_layers=self.n_layers,
            d_model=self.d_model,
            extraction_method=self.extraction_method
        )
    
    @classmethod
    def load(cls, filepath: str) -> "ActivationCache":
        """Load from numpy format."""
        data = np.load(filepath, allow_pickle=True)
        return cls(
            case_id=str(data["case_id"]),
            prompt=str(data["prompt"]),
            model_name=str(data["model_name"]),
            residual_activations=data["residual_activations"],
            token_positions=data["token_positions"],
            n_layers=int(data["n_layers"]),
            d_model=int(data["d_model"]),
            extraction_method=str(data["extraction_method"])
        )


class ActivationExtractor:
    """
    Extract residual stream activations from transformer models.
    
    Supports both base and RLHF-aligned model variants for comparison.
    """
    
    # Common model pairs (base, aligned)
    MODEL_PAIRS = {
        "llama2-7b": {
            "base": "meta-llama/Llama-2-7b-hf",
            "aligned": "meta-llama/Llama-2-7b-chat-hf"
        },
        "llama2-13b": {
            "base": "meta-llama/Llama-2-13b-hf",
            "aligned": "meta-llama/Llama-2-13b-chat-hf"
        },
        "mistral-7b": {
            "base": "mistralai/Mistral-7B-v0.1",
            "aligned": "mistralai/Mistral-7B-Instruct-v0.1"
        },
        "pythia-6.9b": {
            "base": "EleutherAI/pythia-6.9b",
            "aligned": "EleutherAI/pythia-6.9b"  # No official aligned version
        },
        # Smaller models for testing
        "pythia-410m": {
            "base": "EleutherAI/pythia-410m",
            "aligned": "EleutherAI/pythia-410m"
        },
        "gpt2-medium": {
            "base": "gpt2-medium",
            "aligned": "gpt2-medium"  # No aligned version
        }
    }
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize with a HuggingFace model name.
        
        Args:
            model_name: HuggingFace model identifier
            device: "auto", "cuda", "cpu", or "mps"
            dtype: torch.float16 or torch.float32
        """
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError("transformer_lens required. Install with: pip install transformer-lens")
        
        self.model_name = model_name
        self.device = self._resolve_device(device)
        
        print(f"Loading {model_name} on {self.device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            dtype=dtype
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
        # Tokenize
        tokens = self.model.to_tokens(prompt)
        seq_len = tokens.shape[1]
        
        # Run with caching
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name
            )
        
        # Extract residual stream after each layer
        # cache["blocks.{layer}.hook_resid_post"] has shape (batch, seq, d_model)
        activations = []
        
        for layer in range(self.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"]  # (1, seq, d_model)
            
            if method == "last_token":
                # Use final token position
                layer_act = resid[0, -1, :].cpu().numpy()  # (d_model,)
            elif method == "mean_pool":
                # Average across sequence
                layer_act = resid[0].mean(dim=0).cpu().numpy()  # (d_model,)
            elif method == "eos_token":
                # Try to find EOS, fall back to last
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
        
        return np.stack(activations)  # (n_layers, d_model)
    
    def extract_for_case(
        self,
        case_id: str,
        prompt: str,
        method: Literal["last_token", "mean_pool", "eos_token"] = "last_token"
    ) -> ActivationCache:
        """
        Extract and package activations for a single case.
        
        Returns:
            ActivationCache with all metadata
        """
        activations = self.extract_activations(prompt, method)
        
        # Track token positions used (for debugging)
        tokens = self.model.to_tokens(prompt)
        if method == "last_token":
            positions = np.full(self.n_layers, tokens.shape[1] - 1)
        elif method == "mean_pool":
            positions = np.full(self.n_layers, -1)  # -1 indicates pooling
        else:
            positions = np.full(self.n_layers, -2)  # -2 indicates EOS search
        
        return ActivationCache(
            case_id=case_id,
            prompt=prompt,
            model_name=self.model_name,
            residual_activations=activations,
            token_positions=positions,
            n_layers=self.n_layers,
            d_model=self.d_model,
            extraction_method=method
        )
    
    def extract_batch(
        self,
        cases: list[dict],
        method: str = "last_token",
        output_dir: Optional[str] = None
    ) -> list[ActivationCache]:
        """
        Extract activations for multiple cases.
        
        Args:
            cases: List of {"case_id": str, "prompt": str}
            method: Extraction method
            output_dir: If provided, save each cache to this directory
        
        Returns:
            List of ActivationCache objects
        """
        results = []
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, case in enumerate(cases):
            print(f"  [{i+1}/{len(cases)}] {case['case_id']}")
            
            cache = self.extract_for_case(
                case_id=case["case_id"],
                prompt=case["prompt"],
                method=method
            )
            results.append(cache)
            
            if output_dir:
                filepath = Path(output_dir) / f"{case['case_id']}.npz"
                cache.save(str(filepath))
        
        return results


def load_activation_dataset(
    directory: str,
    case_ids: Optional[list[str]] = None
) -> dict[str, ActivationCache]:
    """
    Load all cached activations from a directory.
    
    Returns:
        Dict mapping case_id -> ActivationCache
    """
    results = {}
    dir_path = Path(directory)
    
    for filepath in dir_path.glob("*.npz"):
        cache = ActivationCache.load(str(filepath))
        if case_ids is None or cache.case_id in case_ids:
            results[cache.case_id] = cache
    
    print(f"Loaded {len(results)} activation caches from {directory}")
    return results


def compare_model_activations(
    base_dir: str,
    aligned_dir: str,
    case_ids: list[str]
) -> dict:
    """
    Load and compare activations between base and aligned models.
    
    Returns summary statistics about differences.
    """
    base_caches = load_activation_dataset(base_dir, case_ids)
    aligned_caches = load_activation_dataset(aligned_dir, case_ids)
    
    # Compute statistics
    stats = {
        "n_cases": len(case_ids),
        "layer_wise_l2_diff": [],
        "layer_wise_cosine_sim": []
    }
    
    for case_id in case_ids:
        if case_id not in base_caches or case_id not in aligned_caches:
            continue
            
        base_act = base_caches[case_id].residual_activations
        aligned_act = aligned_caches[case_id].residual_activations
        
        # L2 difference per layer
        l2_diff = np.linalg.norm(base_act - aligned_act, axis=1)
        stats["layer_wise_l2_diff"].append(l2_diff)
        
        # Cosine similarity per layer
        cos_sim = np.array([
            np.dot(base_act[l], aligned_act[l]) / 
            (np.linalg.norm(base_act[l]) * np.linalg.norm(aligned_act[l]) + 1e-8)
            for l in range(base_act.shape[0])
        ])
        stats["layer_wise_cosine_sim"].append(cos_sim)
    
    # Aggregate
    if stats["layer_wise_l2_diff"]:
        stats["mean_l2_diff_by_layer"] = np.mean(stats["layer_wise_l2_diff"], axis=0).tolist()
        stats["mean_cosine_sim_by_layer"] = np.mean(stats["layer_wise_cosine_sim"], axis=0).tolist()
    
    return stats


# === Example usage ===

if __name__ == "__main__":
    print("Activation Extraction Module")
    print("=" * 50)
    print("\nSupported model pairs:")
    for name, pair in ActivationExtractor.MODEL_PAIRS.items():
        print(f"  {name}:")
        print(f"    base: {pair['base']}")
        print(f"    aligned: {pair['aligned']}")
    
    print("\nUsage:")
    print("  extractor = ActivationExtractor('gpt2-medium')")
    print("  cache = extractor.extract_for_case('tinker_1969', prompt)")
    print("  cache.save('tinker_activations.npz')")
