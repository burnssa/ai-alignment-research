"""
Causal Validation for Constitutional Geometry

This module tests whether the geometric structure found in residual streams
is causally relevant to model behavior. We use activation patching to answer:
"If we transplant geometry from aligned→base, does behavior change?"

Key experiments:
1. Activation patching: Replace base model activations with aligned model's
2. Ablation: Zero out principle-encoding directions in aligned model
3. Steering: Add principle direction vectors to shift model behavior

Methodology based on:
- Vig et al. "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"
- Meng et al. "Locating and Editing Factual Associations in GPT"
- Turner et al. "Activation Addition: Steering Language Models Without Optimization"
"""

import torch
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable
from functools import partial
import gc

# TransformerLens for hook-based interventions
from transformer_lens import HookedTransformer

# Local imports
from extract_activations import ActivationExtractor, load_activation_dataset, ActivationCache
from train_probes import LinearProbeTrainer, ProbeResult
from annotate_principles import load_annotations


# === Data Classes ===

@dataclass
class PatchingResult:
    """Result from a single patching experiment."""
    case_id: str
    case_name: str
    correct_principle: str

    # Baseline responses
    base_response: str
    base_principle: Optional[str]
    aligned_response: str
    aligned_principle: Optional[str]

    # Patched response
    patched_response: str
    patched_principle: Optional[str]

    # Metrics
    base_correct: bool
    aligned_correct: bool
    patched_correct: bool

    # Patching details
    patch_layers: list[int] = field(default_factory=list)
    patch_source: str = "aligned"  # Which model's activations we patched in


@dataclass
class CausalExperimentResults:
    """Aggregated results from causal validation experiments."""
    model_pair: str
    experiment_type: str  # "patching", "ablation", "steering"

    results: list[PatchingResult] = field(default_factory=list)

    # Summary metrics
    base_accuracy: float = 0.0
    aligned_accuracy: float = 0.0
    patched_accuracy: float = 0.0

    # Effect sizes
    patch_improvement: float = 0.0  # patched - base
    alignment_gap: float = 0.0       # aligned - base

    def compute_summary(self):
        """Compute summary statistics."""
        n = len(self.results)
        if n == 0:
            return

        self.base_accuracy = sum(r.base_correct for r in self.results) / n
        self.aligned_accuracy = sum(r.aligned_correct for r in self.results) / n
        self.patched_accuracy = sum(r.patched_correct for r in self.results) / n

        self.patch_improvement = self.patched_accuracy - self.base_accuracy
        self.alignment_gap = self.aligned_accuracy - self.base_accuracy

    def summary_report(self) -> str:
        """Generate human-readable summary."""
        self.compute_summary()
        lines = [
            "=" * 60,
            f"CAUSAL VALIDATION RESULTS: {self.experiment_type.upper()}",
            "=" * 60,
            f"Model pair: {self.model_pair}",
            f"Cases tested: {len(self.results)}",
            "",
            "Accuracy:",
            f"  Base model:     {self.base_accuracy:.1%}",
            f"  Aligned model:  {self.aligned_accuracy:.1%}",
            f"  Patched model:  {self.patched_accuracy:.1%}",
            "",
            "Effect sizes:",
            f"  Alignment gap (aligned - base):   {self.alignment_gap:+.1%}",
            f"  Patch improvement (patched - base): {self.patch_improvement:+.1%}",
            "",
        ]

        # Interpretation
        if self.patch_improvement > 0.1:
            lines.append("✓ STRONG CAUSAL EFFECT: Patching geometry improves behavior")
        elif self.patch_improvement > 0.05:
            lines.append("~ MODERATE EFFECT: Some causal influence detected")
        elif self.patch_improvement > 0:
            lines.append("? WEAK EFFECT: Minimal causal influence")
        else:
            lines.append("✗ NO EFFECT: Geometry may be epiphenomenal")

        return "\n".join(lines)


# === Probe Direction Extraction ===

def extract_principle_directions(
    activations: dict[str, ActivationCache],
    annotations: list,
    layers: list[int],
    principle: str = None
) -> dict[int, np.ndarray]:
    """
    Extract principle direction vectors from trained probes.

    For each layer, trains a probe and returns the weight vector(s).
    If principle is specified, returns only that principle's direction.
    Otherwise returns all principle directions.

    Args:
        activations: case_id -> ActivationCache
        annotations: List of PrincipleAnnotation
        layers: Which layers to extract directions for
        principle: If specified, return only this principle's direction

    Returns:
        dict mapping layer -> direction vector (d_model,) or (n_principles, d_model)
    """
    trainer = LinearProbeTrainer(regularization="ridgecv")

    # Get first cache for dimensionality
    first_cache = next(iter(activations.values()))
    n_layers_total = first_cache.n_layers

    directions = {}

    for layer in layers:
        if layer >= n_layers_total:
            print(f"Warning: Layer {layer} exceeds model layers ({n_layers_total})")
            continue

        X, y, case_ids = trainer.prepare_data(activations, annotations, layer)

        if len(case_ids) < 3:
            print(f"Layer {layer}: Insufficient data")
            continue

        result = trainer.train_probe(X, y, layer)

        if result.weights is not None:
            if principle:
                # Get index for specific principle
                try:
                    idx = trainer.PRINCIPLE_NAMES.index(principle)
                    directions[layer] = result.weights[idx]  # (d_model,)
                except ValueError:
                    print(f"Unknown principle: {principle}")
            else:
                directions[layer] = result.weights  # (n_principles, d_model)

    return directions


def load_or_train_directions(
    output_dir: str,
    model_type: str = "aligned",
    layers: list[int] = None,
    force_retrain: bool = False
) -> dict[int, np.ndarray]:
    """
    Load cached probe directions or train new ones.

    Args:
        output_dir: Experiment output directory
        model_type: "base" or "aligned"
        layers: Which layers (default: all)
        force_retrain: Retrain even if cached

    Returns:
        dict mapping layer -> (n_principles, d_model) weight matrix
    """
    cache_file = Path(output_dir) / f"probe_directions_{model_type}.npz"

    if cache_file.exists() and not force_retrain:
        print(f"Loading cached directions from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return {int(k): v for k, v in data.items()}

    # Load data and train
    act_dir = Path(output_dir) / "activations" / model_type
    activations = load_activation_dataset(str(act_dir))
    # Try shared annotations first, fall back to per-model
    ann_path = Path(output_dir).parent.parent / "data" / "annotations.json"
    if not ann_path.exists():
        ann_path = Path(output_dir) / "annotations.json"
    annotations = load_annotations(str(ann_path))

    if layers is None:
        first_cache = next(iter(activations.values()))
        layers = list(range(first_cache.n_layers))

    print(f"Training probes for {len(layers)} layers...")
    directions = extract_principle_directions(activations, annotations, layers)

    # Cache for future use
    np.savez_compressed(cache_file, **{str(k): v for k, v in directions.items()})
    print(f"Cached directions to {cache_file}")

    return directions


# === Model Name Mapping ===

def get_tl_model_name(hf_name: str) -> str:
    """
    Convert model name to TransformerLens format.

    TransformerLens uses full HuggingFace paths for most models,
    so we pass through unchanged for Gemma, Llama, Qwen, etc.
    """
    # TransformerLens uses full HuggingFace paths directly
    return hf_name


# === Activation Patching ===

class ActivationPatcher:
    """
    Perform activation patching experiments using TransformerLens hooks.

    Supports:
    - Full residual stream patching (replace entire activation)
    - Direction patching (add/subtract specific directions)
    - Ablation (zero out specific directions)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16
    ):
        """Load model for patching experiments."""
        self.hf_model_name = model_name
        self.tl_model_name = get_tl_model_name(model_name)
        self.device = self._resolve_device(device)
        self.dtype = dtype

        print(f"Loading {self.tl_model_name} (from {model_name}) for patching experiments...")
        self.model = HookedTransformer.from_pretrained(
            self.tl_model_name,
            device=self.device,
            dtype=dtype
        )

        # Get model config
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model

        print(f"  Loaded: {self.n_layers} layers, d_model={self.d_model}")

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0
    ) -> str:
        """Generate a response without any intervention."""
        tokens = self.model.to_tokens(prompt)

        with torch.no_grad():
            output_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                verbose=False
            )

        # Decode only the new tokens
        response = self.model.to_string(output_tokens[0, tokens.shape[1]:])
        return response.strip()

    def generate_with_patch(
        self,
        prompt: str,
        patch_activations: dict[int, np.ndarray],
        patch_position: int = -1,
        max_new_tokens: int = 200,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response with activations patched at specified layers.

        Uses TransformerLens hooks to replace residual stream activations.

        Args:
            prompt: Input prompt
            patch_activations: layer -> activation vector (d_model,) to patch in
            patch_position: Token position to patch (-1 for last token)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)

        Returns:
            Generated response string
        """
        tokens = self.model.to_tokens(prompt)
        seq_len = tokens.shape[1]

        # Resolve patch position
        if patch_position < 0:
            patch_position = seq_len + patch_position

        # Create hook functions for TransformerLens
        def make_patch_hook(patch_vec, pos):
            """Create a TransformerLens-style hook that patches activation."""
            patch_tensor = torch.tensor(
                patch_vec,
                device=self.model.cfg.device,
                dtype=self.dtype
            )
            def hook_fn(activation, hook):
                # activation shape: (batch, seq, d_model)
                # Only patch at the specified position
                if activation.shape[1] > pos:
                    activation[:, pos, :] = patch_tensor
                return activation
            return hook_fn

        # Build list of (hook_name, hook_fn) tuples
        fwd_hooks = []
        for layer_idx, patch_vec in patch_activations.items():
            if layer_idx < self.n_layers:
                # TransformerLens hook point for residual stream post-attention
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                fwd_hooks.append((hook_name, make_patch_hook(patch_vec, patch_position)))

        # Add hooks, generate, then reset
        for hook_name, hook_fn in fwd_hooks:
            self.model.add_hook(hook_name, hook_fn)

        try:
            with torch.no_grad():
                output_tokens = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.0,
                    verbose=False
                )
        finally:
            self.model.reset_hooks()

        response = self.model.to_string(output_tokens[0, tokens.shape[1]:])
        return response.strip()

    def generate_with_direction_add(
        self,
        prompt: str,
        directions: dict[int, np.ndarray],
        scale: float = 1.0,
        patch_position: int = -1,
        max_new_tokens: int = 200,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response with direction vectors added to activations.

        This is "activation steering" - adding a direction to shift behavior.

        Args:
            prompt: Input prompt
            directions: layer -> direction vector (d_model,) to add
            scale: Multiplier for direction vectors
            patch_position: Token position to modify
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response string
        """
        tokens = self.model.to_tokens(prompt)
        seq_len = tokens.shape[1]

        if patch_position < 0:
            patch_position = seq_len + patch_position

        def make_add_hook(direction, pos, mult):
            """Create a hook that adds a direction vector."""
            dir_tensor = torch.tensor(
                direction * mult,
                device=self.model.cfg.device,
                dtype=self.dtype
            )
            def hook_fn(activation, hook):
                if activation.shape[1] > pos:
                    activation[:, pos, :] = activation[:, pos, :] + dir_tensor
                return activation
            return hook_fn

        # Add hooks for each layer
        for layer_idx, direction in directions.items():
            if layer_idx < self.n_layers:
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                self.model.add_hook(hook_name, make_add_hook(direction, patch_position, scale))

        try:
            with torch.no_grad():
                output_tokens = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.0,
                    verbose=False
                )
        finally:
            self.model.reset_hooks()

        response = self.model.to_string(output_tokens[0, tokens.shape[1]:])
        return response.strip()

    def generate_with_direction_add_all_positions(
        self,
        prompt: str,
        directions: dict[int, np.ndarray],
        scale: float = 1.0,
        max_new_tokens: int = 200,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response with direction vectors added at ALL token positions.

        Unlike generate_with_direction_add (single position), this adds the
        steering vector to every position on every forward pass — including
        during autoregressive generation with KV caching. This follows the
        activation addition methodology from Turner et al. (2023).

        Args:
            prompt: Input prompt
            directions: layer -> direction vector (d_model,) to add
            scale: Multiplier for direction vectors
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response string
        """
        tokens = self.model.to_tokens(prompt)

        def make_add_all_hook(direction, mult):
            """Create a hook that adds a direction vector to all positions."""
            dir_tensor = torch.tensor(
                direction * mult,
                device=self.model.cfg.device,
                dtype=self.dtype
            )
            def hook_fn(activation, hook):
                # Add to all positions — works during both prefill and generation
                activation[:, :, :] = activation[:, :, :] + dir_tensor
                return activation
            return hook_fn

        # Add hooks for each layer
        for layer_idx, direction in directions.items():
            if layer_idx < self.n_layers:
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                self.model.add_hook(hook_name, make_add_all_hook(direction, scale))

        try:
            with torch.no_grad():
                output_tokens = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.0,
                    verbose=False
                )
        finally:
            self.model.reset_hooks()

        response = self.model.to_string(output_tokens[0, tokens.shape[1]:])
        return response.strip()

    def generate_with_ablation(
        self,
        prompt: str,
        directions: dict[int, np.ndarray],
        patch_position: int = -1,
        max_new_tokens: int = 200,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response with specified directions ablated (projected out).

        This tests necessity: if we remove the direction, does behavior degrade?

        Args:
            prompt: Input prompt
            directions: layer -> direction vector(s) to ablate
            patch_position: Token position to modify
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response string
        """
        tokens = self.model.to_tokens(prompt)
        seq_len = tokens.shape[1]

        if patch_position < 0:
            patch_position = seq_len + patch_position

        def make_ablate_hook(direction, pos):
            """Create a hook that projects out a direction."""
            dir_tensor = torch.tensor(
                direction,
                device=self.model.cfg.device,
                dtype=self.dtype
            )
            dir_norm = dir_tensor / (torch.norm(dir_tensor) + 1e-8)

            def hook_fn(activation, hook):
                if activation.shape[1] > pos:
                    # Project out: a - (a · d)d where d is unit direction
                    act_at_pos = activation[:, pos, :]
                    projection = torch.sum(act_at_pos * dir_norm, dim=-1, keepdim=True) * dir_norm
                    activation[:, pos, :] = act_at_pos - projection
                return activation
            return hook_fn

        # Add hooks for each layer
        for layer_idx, direction in directions.items():
            if layer_idx < self.n_layers:
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                self.model.add_hook(hook_name, make_ablate_hook(direction, patch_position))

        try:
            with torch.no_grad():
                output_tokens = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 0.0,
                    verbose=False
                )
        finally:
            self.model.reset_hooks()

        response = self.model.to_string(output_tokens[0, tokens.shape[1]:])
        return response.strip()


# === Response Parsing ===

PRINCIPLE_KEYWORDS = {
    "free_expression": ["free expression", "first amendment", "speech", "free speech"],
    "equal_protection": ["equal protection", "fourteenth amendment", "discrimination", "equality"],
    "due_process": ["due process", "procedural", "fair trial", "counsel", "miranda"],
    "federalism": ["federalism", "commerce clause", "state sovereignty", "federal power", "tenth amendment"],
    "privacy_liberty": ["privacy", "liberty", "autonomy", "substantive due process", "bodily"]
}


def parse_principle_from_response(response: str) -> Optional[str]:
    """
    Extract the primary constitutional principle from a model response.

    Returns the principle name or None if unclear.
    """
    response_lower = response.lower()

    # Look for explicit "1." or "primary" markers
    lines = response.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line_lower = line.lower()

        # Check for numbered list format
        if line_lower.strip().startswith(('1.', '1)', '1:')):
            for principle, keywords in PRINCIPLE_KEYWORDS.items():
                for kw in keywords:
                    if kw in line_lower:
                        return principle

    # Fallback: find first mentioned principle
    first_positions = {}
    for principle, keywords in PRINCIPLE_KEYWORDS.items():
        for kw in keywords:
            pos = response_lower.find(kw)
            if pos != -1:
                if principle not in first_positions or pos < first_positions[principle]:
                    first_positions[principle] = pos

    if first_positions:
        return min(first_positions, key=first_positions.get)

    return None


def normalize_principle(principle: str) -> str:
    """Normalize principle names to standard format."""
    mappings = {
        "free expression": "free_expression",
        "equal protection": "equal_protection",
        "due process": "due_process",
        "privacy/liberty": "privacy_liberty",
        "privacy": "privacy_liberty",
        "liberty": "privacy_liberty",
    }
    return mappings.get(principle.lower(), principle.lower().replace(" ", "_"))


# === Experiment Runner ===

def run_patching_experiment(
    base_model_name: str,
    aligned_model_name: str,
    output_dir: str,
    cases: list[dict],
    patch_layers: list[int],
    device: str = "auto"
) -> CausalExperimentResults:
    """
    Run full activation patching experiment.

    For each case:
    1. Generate response with base model (unpatched)
    2. Generate response with aligned model (for comparison)
    3. Generate response with base model + aligned activations patched in
    4. Compare principle identification across conditions

    Args:
        base_model_name: HuggingFace model name for base model
        aligned_model_name: HuggingFace model name for aligned model
        output_dir: Directory with activations and annotations
        cases: List of case dicts with case_id, case_name, etc.
        patch_layers: Which layers to patch
        device: Compute device

    Returns:
        CausalExperimentResults with all outcomes
    """
    from cases import format_prompt

    results = CausalExperimentResults(
        model_pair=f"{base_model_name} / {aligned_model_name}",
        experiment_type="patching"
    )

    # Load aligned activations (source for patching)
    aligned_act_dir = Path(output_dir) / "activations" / "aligned"
    aligned_activations = load_activation_dataset(str(aligned_act_dir))

    # Load annotations for ground truth
    ann_path = Path(output_dir).parent.parent / "data" / "annotations.json"
    if not ann_path.exists():
        ann_path = Path(output_dir) / "annotations.json"
    annotations = load_annotations(str(ann_path))
    annotation_lookup = {a.case_id: a for a in annotations}

    print(f"\nRunning patching experiment on {len(cases)} cases...")
    print(f"Patch layers: {patch_layers}")
    print("=" * 60)

    # === PHASE 1: Generate aligned model responses ===
    # Load aligned model first, generate all responses, then free memory
    print(f"\n--- PHASE 1: Generating aligned model responses ---")
    print(f"Loading aligned model: {aligned_model_name}")
    aligned_patcher = ActivationPatcher(aligned_model_name, device=device)

    aligned_responses = {}
    for i, case in enumerate(cases):
        case_id = case["case_id"]
        case_name = case.get("case_name", case_id)

        if case_id not in annotation_lookup:
            continue
        if case_id not in aligned_activations:
            continue

        prompt = format_prompt(case)
        print(f"  [{i+1}/{len(cases)}] {case_name[:40]}...")
        aligned_responses[case_id] = aligned_patcher.generate_response(prompt, max_new_tokens=300)

    # Free aligned model memory
    print("\nFreeing aligned model memory...")
    del aligned_patcher
    gc.collect()
    torch.cuda.empty_cache()

    # === PHASE 2: Generate base and patched responses ===
    print(f"\n--- PHASE 2: Generating base and patched responses ---")
    print(f"Loading base model: {base_model_name}")
    base_patcher = ActivationPatcher(base_model_name, device=device)

    for i, case in enumerate(cases):
        case_id = case["case_id"]
        case_name = case.get("case_name", case_id)

        print(f"\n[{i+1}/{len(cases)}] {case_name[:50]}...")

        # Get ground truth
        if case_id not in annotation_lookup:
            print(f"  No annotation for {case_id}, skipping")
            continue

        annotation = annotation_lookup[case_id]
        # Get primary principle from weights (highest weight)
        correct_principle = max(annotation.weights, key=annotation.weights.get)

        # Get aligned activations for patching
        if case_id not in aligned_activations:
            print(f"  No activations for {case_id}, skipping")
            continue

        aligned_cache = aligned_activations[case_id]

        # Prepare patch activations (aligned model's residual stream)
        patch_acts = {
            layer: aligned_cache.residual_activations[layer]
            for layer in patch_layers
            if layer < aligned_cache.n_layers
        }

        # Format prompt
        prompt = format_prompt(case)

        # 1. Base model response (unpatched)
        print("  Generating base response...")
        base_response = base_patcher.generate_response(prompt, max_new_tokens=300)
        base_principle = parse_principle_from_response(base_response)

        # 2. Get aligned response from phase 1
        aligned_response = aligned_responses.get(case_id, "")
        aligned_principle = parse_principle_from_response(aligned_response)

        # 3. Patched base model response
        print("  Generating patched response...")
        patched_response = base_patcher.generate_with_patch(
            prompt,
            patch_acts,
            max_new_tokens=300
        )
        patched_principle = parse_principle_from_response(patched_response)

        # Evaluate
        base_correct = base_principle == correct_principle if base_principle else False
        aligned_correct = aligned_principle == correct_principle if aligned_principle else False
        patched_correct = patched_principle == correct_principle if patched_principle else False

        print(f"  Correct: {correct_principle}")
        print(f"  Base: {base_principle} ({'✓' if base_correct else '✗'})")
        print(f"  Aligned: {aligned_principle} ({'✓' if aligned_correct else '✗'})")
        print(f"  Patched: {patched_principle} ({'✓' if patched_correct else '✗'})")

        results.results.append(PatchingResult(
            case_id=case_id,
            case_name=case_name,
            correct_principle=correct_principle,
            base_response=base_response,
            base_principle=base_principle,
            aligned_response=aligned_response,
            aligned_principle=aligned_principle,
            patched_response=patched_response,
            patched_principle=patched_principle,
            base_correct=base_correct,
            aligned_correct=aligned_correct,
            patched_correct=patched_correct,
            patch_layers=list(patch_acts.keys()),
            patch_source="aligned"
        ))

    results.compute_summary()
    return results


def save_experiment_results(results: CausalExperimentResults, filepath: str):
    """Save experiment results to JSON."""
    data = {
        "model_pair": results.model_pair,
        "experiment_type": results.experiment_type,
        "summary": {
            "base_accuracy": results.base_accuracy,
            "aligned_accuracy": results.aligned_accuracy,
            "patched_accuracy": results.patched_accuracy,
            "patch_improvement": results.patch_improvement,
            "alignment_gap": results.alignment_gap
        },
        "results": [
            {
                "case_id": r.case_id,
                "case_name": r.case_name,
                "correct_principle": r.correct_principle,
                "base_response": r.base_response,
                "base_principle": r.base_principle,
                "aligned_response": r.aligned_response,
                "aligned_principle": r.aligned_principle,
                "patched_response": r.patched_response,
                "patched_principle": r.patched_principle,
                "base_correct": r.base_correct,
                "aligned_correct": r.aligned_correct,
                "patched_correct": r.patched_correct,
                "patch_layers": r.patch_layers
            }
            for r in results.results
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved results to {filepath}")


# === CLI ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Causal Validation Experiments")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Experiment output directory (with activations/)")
    parser.add_argument("--model-pair", type=str, default="gemma2-27b",
                       help="Model pair to test")
    parser.add_argument("--patch-layers", type=str, default="20-30",
                       help="Layer range to patch (e.g., '20-30' or '15,18,21')")
    parser.add_argument("--device", type=str, default="auto",
                       help="Compute device")
    parser.add_argument("--max-cases", type=int, default=None,
                       help="Maximum cases to test")

    args = parser.parse_args()

    # Parse layer range
    if "-" in args.patch_layers:
        start, end = map(int, args.patch_layers.split("-"))
        patch_layers = list(range(start, end + 1))
    else:
        patch_layers = [int(x) for x in args.patch_layers.split(",")]

    # Get model names
    model_info = ActivationExtractor.MODEL_PAIRS.get(args.model_pair)
    if not model_info:
        raise ValueError(f"Unknown model pair: {args.model_pair}")

    # Load cases
    from cases import ALL_CASES
    cases = ALL_CASES[:args.max_cases] if args.max_cases else ALL_CASES

    # Run experiment
    results = run_patching_experiment(
        base_model_name=model_info["base"],
        aligned_model_name=model_info["aligned"],
        output_dir=args.output_dir,
        cases=cases,
        patch_layers=patch_layers,
        device=args.device
    )

    # Print and save results
    print("\n" + results.summary_report())

    save_path = Path(args.output_dir) / f"causal_validation_{args.model_pair}.json"
    save_experiment_results(results, str(save_path))
