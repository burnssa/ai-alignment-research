"""
Data Schemas for Criminal Planning Geometry Experiment

Defines dataclasses for:
- Prompt annotations (Claude-generated severity scores)
- Patronus response scores
- Activation caches (reused from SCOTUS experiment)
"""

import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal
from pathlib import Path
import json


# Annotation dimensions for prompt transgressiveness
ANNOTATION_DIMENSIONS = [
    "severity",        # Overall transgressiveness (0-1)
    "specificity",     # How actionable/specific (0-1)
    "real_world_risk", # Likelihood of real-world harm (0-1)
]

# Harm type categories
HARM_TYPES = [
    "violence",
    "fraud",
    "drugs",
    "weapons",
    "cyber",
    "other"
]


@dataclass
class PromptAnnotation:
    """Claude-generated annotation for a single prompt's transgressiveness."""
    prompt_id: str
    prompt_text: str

    # Core severity dimensions (0.0-1.0)
    severity: float           # Overall transgressiveness
    specificity: float        # How actionable/specific the request is
    real_world_risk: float    # Likelihood of enabling real harm

    # Categorical
    harm_type: str            # Primary harm category

    # Metadata
    annotator_model: str = "claude-opus-4-5-20251101"
    annotation_timestamp: str = ""
    reasoning: str = ""       # Brief explanation

    def to_vector(self) -> list[float]:
        """Return annotation dimensions as ordered vector for ML pipeline."""
        return [self.severity, self.specificity, self.real_world_risk]

    def validate(self) -> list[str]:
        """Check for potential issues with annotation."""
        issues = []

        for dim in ["severity", "specificity", "real_world_risk"]:
            val = getattr(self, dim)
            if not 0.0 <= val <= 1.0:
                issues.append(f"{dim} out of range [0,1]: {val}")

        if self.harm_type not in HARM_TYPES:
            issues.append(f"Unknown harm_type: {self.harm_type}")

        return issues


@dataclass
class PatronusScore:
    """Patronus API response scoring for a model-generated response."""
    prompt_id: str
    model_name: str           # Which model generated the response
    response_text: str

    # Core toxicity scores (0.0-1.0, higher = more toxic)
    toxicity_score: float     # Overall toxicity

    # Sub-dimensions (if available from Patronus)
    harmfulness: Optional[float] = None
    helpfulness_to_harm: Optional[float] = None  # Did model help with harmful request?

    # Metadata
    evaluator_model: str = ""
    evaluation_timestamp: str = ""
    raw_response: Optional[dict] = None  # Full Patronus API response

    def to_vector(self) -> list[float]:
        """Return scores as ordered vector."""
        return [self.toxicity_score]


@dataclass
class ActivationCache:
    """
    Cached activations for a single prompt.

    Adapted from scotus-constitutional-geometry with minimal changes.
    """
    prompt_id: str            # Changed from case_id
    prompt_text: str          # Changed from prompt
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
            prompt_id=self.prompt_id,
            prompt_text=self.prompt_text,
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
            prompt_id=str(data["prompt_id"]),
            prompt_text=str(data["prompt_text"]),
            model_name=str(data["model_name"]),
            residual_activations=data["residual_activations"],
            token_positions=data["token_positions"],
            n_layers=int(data["n_layers"]),
            d_model=int(data["d_model"]),
            extraction_method=str(data["extraction_method"])
        )


@dataclass
class RegressionTarget:
    """
    Combined regression targets for a single prompt.

    Links prompt annotation, response scores, and computes derived targets.
    """
    prompt_id: str

    # From Claude annotation
    prompt_severity: float
    prompt_specificity: float
    prompt_risk: float

    # From Patronus scoring (per model)
    base_toxicity: Optional[float] = None
    aligned_toxicity: Optional[float] = None

    # Derived: restraint = how much model "held back"
    # Higher = more restraint (prompt was bad but response was clean)
    @property
    def base_restraint(self) -> Optional[float]:
        if self.base_toxicity is None:
            return None
        return self.prompt_severity - self.base_toxicity

    @property
    def aligned_restraint(self) -> Optional[float]:
        if self.aligned_toxicity is None:
            return None
        return self.prompt_severity - self.aligned_toxicity

    def to_target_vector(self, target: Literal["severity", "toxicity", "restraint"]) -> list[float]:
        """Get target vector for regression."""
        if target == "severity":
            return [self.prompt_severity]
        elif target == "toxicity":
            # Returns aligned toxicity by default
            return [self.aligned_toxicity or 0.0]
        elif target == "restraint":
            return [self.aligned_restraint or 0.0]
        else:
            raise ValueError(f"Unknown target: {target}")


# === Serialization utilities ===

def save_annotations(annotations: list[PromptAnnotation], filepath: str):
    """Save annotations to JSON file."""
    data = [asdict(a) for a in annotations]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(annotations)} annotations to {filepath}")


def load_annotations(filepath: str) -> list[PromptAnnotation]:
    """Load annotations from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [PromptAnnotation(**d) for d in data]


def save_patronus_scores(scores: list[PatronusScore], filepath: str):
    """Save Patronus scores to JSON file."""
    data = []
    for s in scores:
        d = asdict(s)
        # Remove response_text for space (can be large)
        d["response_text_truncated"] = d["response_text"][:500] + "..." if len(d["response_text"]) > 500 else d["response_text"]
        del d["response_text"]
        data.append(d)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(scores)} Patronus scores to {filepath}")


def load_patronus_scores(filepath: str) -> list[PatronusScore]:
    """Load Patronus scores from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Restore truncated text
    for d in data:
        if "response_text_truncated" in d:
            d["response_text"] = d.pop("response_text_truncated")
    return [PatronusScore(**d) for d in data]
