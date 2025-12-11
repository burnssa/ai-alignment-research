"""
Linear Probe Training for Transgression Detection

Adapted from scotus-constitutional-geometry for the criminal planning experiment.
Trains linear probes to predict transgression metrics from residual stream activations.

Regression targets:
1. Prompt severity (Claude annotation) - does geometry encode "how bad is this prompt?"
2. Response toxicity (Patronus score) - does geometry predict "how bad will output be?"
3. Restraint delta (severity - toxicity) - does geometry encode "how much will model refuse?"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import json
import warnings

from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from .schemas import ActivationCache, PromptAnnotation, PatronusScore, RegressionTarget


# Target names for this experiment
TARGET_NAMES = [
    "prompt_severity",
    "response_toxicity",
    "restraint_delta"
]

# Joint annotation dimensions (like SCOTUS 5 principles)
# Using severity + harm_type only (6 dimensions)
# - Dropped specificity and real_world_risk as they may overlap with severity
#   or be harder to label reliably
# - This gives us more orthogonal dimensions like SCOTUS constitutional principles

CONTINUOUS_DIMENSIONS = ["severity"]  # Just severity - clearer concept

# Categorical harm types (will be one-hot encoded)
HARM_TYPES = ["violence", "fraud", "drugs", "weapons", "cyber"]  # 'other' is reference

# All dimensions for joint regression (6 total)
ANNOTATION_DIMENSIONS = CONTINUOUS_DIMENSIONS + HARM_TYPES


@dataclass
class ProbeResult:
    """Results from training a linear probe on one layer."""
    layer: int
    target_name: str         # What we're predicting
    r2_score: float          # Cross-validated R²
    r2_std: float            # Standard deviation across folds
    mse: float               # Mean squared error

    # Probe weights (for analysis)
    weights: Optional[np.ndarray] = None  # (d_model,)
    bias: Optional[float] = None

    # Regularization used
    alpha: float = 1.0


@dataclass
class JointProbeResult:
    """Results from training a joint multi-output probe (like SCOTUS 5 principles)."""
    layer: int
    r2_score: float          # Overall cross-validated R²
    r2_std: float            # Standard deviation across folds
    mse: float               # Mean squared error

    # Per-dimension breakdown
    dimension_r2: dict[str, float] = field(default_factory=dict)

    # Probe weights (for analysis)
    weights: Optional[np.ndarray] = None  # (n_dimensions, d_model)
    bias: Optional[np.ndarray] = None     # (n_dimensions,)

    # Regularization used
    alpha: float = 1.0


@dataclass
class ProbeComparison:
    """Comparison of probe performance between base and aligned models."""
    target_name: str
    base_results: list[ProbeResult]
    aligned_results: list[ProbeResult]

    # Summary statistics
    best_base_layer: int = 0
    best_base_r2: float = 0.0
    best_aligned_layer: int = 0
    best_aligned_r2: float = 0.0

    # Per-layer comparison
    r2_difference_by_layer: list[float] = field(default_factory=list)

    def compute_summary(self):
        """Compute summary statistics."""
        base_r2s = [r.r2_score for r in self.base_results]
        aligned_r2s = [r.r2_score for r in self.aligned_results]

        self.best_base_layer = int(np.argmax(base_r2s))
        self.best_base_r2 = max(base_r2s)
        self.best_aligned_layer = int(np.argmax(aligned_r2s))
        self.best_aligned_r2 = max(aligned_r2s)

        self.r2_difference_by_layer = [
            aligned_r2s[i] - base_r2s[i]
            for i in range(len(base_r2s))
        ]

    def summary_report(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"LINEAR PROBE COMPARISON: {self.target_name}",
            "=" * 60,
            "",
            f"Best Base Model Performance:",
            f"  Layer {self.best_base_layer}: R² = {self.best_base_r2:.4f}",
            "",
            f"Best Aligned Model Performance:",
            f"  Layer {self.best_aligned_layer}: R² = {self.best_aligned_r2:.4f}",
            "",
            f"Improvement from RLHF: {self.best_aligned_r2 - self.best_base_r2:+.4f}",
            "",
            "Layer-by-layer R² difference (aligned - base):",
        ]

        for i, diff in enumerate(self.r2_difference_by_layer):
            marker = "**" if diff > 0.05 else "  "
            lines.append(f"  Layer {i:2d}: {diff:+.4f} {marker}")

        return "\n".join(lines)


class LinearProbeTrainer:
    """
    Train linear probes to predict transgression metrics from activations.

    Uses Ridge regression with cross-validation for regularization selection.
    """

    def __init__(
        self,
        regularization: Literal["ridge", "ridgecv", "none"] = "ridgecv",
        cv_folds: int = 5,
        alphas: Optional[list[float]] = None
    ):
        self.regularization = regularization
        self.cv_folds = cv_folds
        self.alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    def prepare_data(
        self,
        activations: dict[str, ActivationCache],
        targets: dict[str, float],
        layer: int
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare X (activations) and y (target values) for a specific layer.

        Args:
            activations: prompt_id -> ActivationCache
            targets: prompt_id -> target value (float)
            layer: Which layer to extract

        Returns:
            X: (n_samples, d_model) activation matrix
            y: (n_samples,) target values
            prompt_ids: List of prompt IDs in order
        """
        X_list = []
        y_list = []
        prompt_ids = []

        for prompt_id, cache in activations.items():
            if prompt_id not in targets:
                continue

            act = cache.residual_activations[layer]
            X_list.append(act)
            y_list.append(targets[prompt_id])
            prompt_ids.append(prompt_id)

        X = np.stack(X_list)
        y = np.array(y_list)

        return X, y, prompt_ids

    def train_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer: int,
        target_name: str
    ) -> ProbeResult:
        """
        Train a linear probe for one layer with cross-validation.

        Args:
            X: (n_samples, d_model) activations
            y: (n_samples,) targets
            layer: Layer index (for metadata)
            target_name: Name of what we're predicting

        Returns:
            ProbeResult with R² scores and probe weights
        """
        n_samples = X.shape[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if n_samples < 10:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=min(self.cv_folds, n_samples), shuffle=True, random_state=42)

        if self.regularization == "ridgecv":
            model = RidgeCV(alphas=self.alphas, cv=cv)
        elif self.regularization == "ridge":
            model = Ridge(alpha=1.0)
        else:
            model = LinearRegression()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_scaled, y)

        cv_scores = cross_val_score(
            model, X_scaled, y,
            cv=cv,
            scoring='r2'
        )

        weights = model.coef_
        alpha = model.alpha_ if hasattr(model, 'alpha_') else 1.0

        return ProbeResult(
            layer=layer,
            target_name=target_name,
            r2_score=float(np.mean(cv_scores)),
            r2_std=float(np.std(cv_scores)),
            mse=float(mean_squared_error(y, model.predict(X_scaled))),
            weights=weights,
            bias=float(model.intercept_),
            alpha=alpha
        )

    def train_all_layers(
        self,
        activations: dict[str, ActivationCache],
        targets: dict[str, float],
        n_layers: int,
        target_name: str
    ) -> list[ProbeResult]:
        """
        Train probes for all layers.

        Returns:
            List of ProbeResult, one per layer
        """
        results = []

        for layer in range(n_layers):
            X, y, prompt_ids = self.prepare_data(activations, targets, layer)

            if len(prompt_ids) < 3:
                print(f"  Layer {layer}: Insufficient data ({len(prompt_ids)} samples)")
                results.append(ProbeResult(
                    layer=layer, target_name=target_name,
                    r2_score=0.0, r2_std=0.0, mse=float('inf')
                ))
                continue

            result = self.train_probe(X, y, layer, target_name)
            results.append(result)

            print(f"  Layer {layer:2d}: R² = {result.r2_score:.4f} (±{result.r2_std:.4f})")

        return results


def compare_models(
    base_activations: dict[str, ActivationCache],
    aligned_activations: dict[str, ActivationCache],
    targets: dict[str, float],
    n_layers: int,
    target_name: str,
    cv_folds: int = 5
) -> ProbeComparison:
    """
    Compare linear probe performance between base and aligned models.

    Args:
        base_activations: prompt_id -> ActivationCache for base model
        aligned_activations: prompt_id -> ActivationCache for aligned model
        targets: prompt_id -> target value
        n_layers: Number of layers to probe
        target_name: Name of the target variable
        cv_folds: Cross-validation folds

    Returns:
        ProbeComparison with results for both models
    """
    trainer = LinearProbeTrainer(cv_folds=cv_folds)

    print("\n" + "=" * 50)
    print(f"Training probes for: {target_name}")
    print("=" * 50)

    print("\n--- BASE MODEL ---")
    base_results = trainer.train_all_layers(base_activations, targets, n_layers, target_name)

    print("\n--- ALIGNED MODEL ---")
    aligned_results = trainer.train_all_layers(aligned_activations, targets, n_layers, target_name)

    comparison = ProbeComparison(
        target_name=target_name,
        base_results=base_results,
        aligned_results=aligned_results
    )
    comparison.compute_summary()

    return comparison


def save_comparison(comparison: ProbeComparison, filepath: str):
    """Save comparison results to JSON."""
    data = {
        "target_name": comparison.target_name,
        "base_results": [
            {
                "layer": r.layer,
                "r2_score": r.r2_score,
                "r2_std": r.r2_std,
                "mse": r.mse,
                "alpha": r.alpha
            }
            for r in comparison.base_results
        ],
        "aligned_results": [
            {
                "layer": r.layer,
                "r2_score": r.r2_score,
                "r2_std": r.r2_std,
                "mse": r.mse,
                "alpha": r.alpha
            }
            for r in comparison.aligned_results
        ],
        "summary": {
            "best_base_layer": comparison.best_base_layer,
            "best_base_r2": comparison.best_base_r2,
            "best_aligned_layer": comparison.best_aligned_layer,
            "best_aligned_r2": comparison.best_aligned_r2,
            "r2_difference_by_layer": comparison.r2_difference_by_layer
        }
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved comparison to {filepath}")


def plot_layer_comparison(comparison: ProbeComparison, output_path: Optional[str] = None):
    """Plot R² by layer for base vs aligned models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    n_layers = len(comparison.base_results)
    layers = list(range(n_layers))

    base_r2 = [r.r2_score for r in comparison.base_results]
    aligned_r2 = [r.r2_score for r in comparison.aligned_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(layers, base_r2, 'b-o', label='Base Model', markersize=4)
    ax1.plot(layers, aligned_r2, 'r-o', label='Aligned Model', markersize=4)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Cross-validated R²')
    ax1.set_title(f'Linear Probe Performance: {comparison.target_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    diff = comparison.r2_difference_by_layer
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax2.bar(layers, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('R² Difference (Aligned - Base)')
    ax2.set_title('RLHF Effect by Layer (Green = Improvement)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


# === Joint Multi-Dimensional Regression (like SCOTUS 5 principles) ===

@dataclass
class JointProbeComparison:
    """Comparison for joint multi-output probes."""
    base_results: list[JointProbeResult]
    aligned_results: list[JointProbeResult]

    # Summary statistics
    best_base_layer: int = 0
    best_base_r2: float = 0.0
    best_aligned_layer: int = 0
    best_aligned_r2: float = 0.0

    r2_difference_by_layer: list[float] = field(default_factory=list)

    def compute_summary(self):
        """Compute summary statistics."""
        base_r2s = [r.r2_score for r in self.base_results]
        aligned_r2s = [r.r2_score for r in self.aligned_results]

        self.best_base_layer = int(np.argmax(base_r2s))
        self.best_base_r2 = max(base_r2s)
        self.best_aligned_layer = int(np.argmax(aligned_r2s))
        self.best_aligned_r2 = max(aligned_r2s)

        self.r2_difference_by_layer = [
            aligned_r2s[i] - base_r2s[i]
            for i in range(len(base_r2s))
        ]

    def summary_report(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "JOINT LINEAR PROBE COMPARISON (All Dimensions)",
            "=" * 60,
            "",
            f"Dimensions: {ANNOTATION_DIMENSIONS}",
            "",
            f"Best Base Model Performance:",
            f"  Layer {self.best_base_layer}: R² = {self.best_base_r2:.4f}",
            "",
            f"Best Aligned Model Performance:",
            f"  Layer {self.best_aligned_layer}: R² = {self.best_aligned_r2:.4f}",
            "",
            f"Improvement from RLHF: {self.best_aligned_r2 - self.best_base_r2:+.4f}",
            "",
            "Per-dimension R² at best aligned layer:",
        ]

        best_result = self.aligned_results[self.best_aligned_layer]
        for dim, r2 in best_result.dimension_r2.items():
            lines.append(f"  {dim}: {r2:.4f}")

        lines.append("")
        lines.append("Layer-by-layer R² difference (aligned - base):")

        for i, diff in enumerate(self.r2_difference_by_layer):
            marker = "**" if diff > 0.05 else "  "
            lines.append(f"  Layer {i:2d}: {diff:+.4f} {marker}")

        return "\n".join(lines)


def prepare_joint_targets(
    annotations: list[PromptAnnotation]
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Prepare joint target matrix from annotations.

    Uses severity + harm_type only (6 dimensions total):
    - severity (continuous)
    - violence, fraud, drugs, weapons, cyber (one-hot)

    Returns:
        targets: prompt_id -> np.ndarray of shape (n_dimensions,)
        prompt_ids: list of prompt IDs
    """
    targets = {}

    for ann in annotations:
        # Just severity (dropped specificity and real_world_risk)
        continuous = [ann.severity]

        # One-hot encode harm_type
        harm_onehot = [1.0 if ann.harm_type == ht else 0.0 for ht in HARM_TYPES]

        # Combine: [severity, violence, fraud, drugs, weapons, cyber]
        target_vec = np.array(continuous + harm_onehot)
        targets[ann.prompt_id] = target_vec

    return targets, list(targets.keys())


def train_joint_probe(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    cv_folds: int = 5,
    alphas: Optional[list[float]] = None
) -> JointProbeResult:
    """
    Train a joint multi-output probe for one layer.

    Args:
        X: (n_samples, d_model) activations
        y: (n_samples, n_dimensions) targets
        layer: Layer index
        cv_folds: Cross-validation folds
        alphas: Regularization strengths

    Returns:
        JointProbeResult with overall and per-dimension R²
    """
    alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    n_samples = X.shape[0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_samples < 10:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=min(cv_folds, n_samples), shuffle=True, random_state=42)

    # Use RidgeCV for multi-output regression
    model = RidgeCV(alphas=alphas, cv=cv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)

    # Overall cross-validated R²
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')

    # Per-dimension R²
    dimension_r2 = {}
    for i, dim_name in enumerate(ANNOTATION_DIMENSIONS):
        y_dim = y[:, i]
        if np.std(y_dim) < 1e-6:
            dimension_r2[dim_name] = 0.0
        else:
            dim_model = Ridge(alpha=model.alpha_ if hasattr(model, 'alpha_') else 1.0)
            dim_scores = cross_val_score(dim_model, X_scaled, y_dim, cv=cv, scoring='r2')
            dimension_r2[dim_name] = float(np.mean(dim_scores))

    weights = model.coef_
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    return JointProbeResult(
        layer=layer,
        r2_score=float(np.mean(cv_scores)),
        r2_std=float(np.std(cv_scores)),
        mse=float(mean_squared_error(y, model.predict(X_scaled))),
        dimension_r2=dimension_r2,
        weights=weights,
        bias=model.intercept_,
        alpha=model.alpha_ if hasattr(model, 'alpha_') else 1.0
    )


def train_joint_all_layers(
    activations: dict[str, ActivationCache],
    targets: dict[str, np.ndarray],
    n_layers: int,
    cv_folds: int = 5
) -> list[JointProbeResult]:
    """
    Train joint probes for all layers.

    Returns:
        List of JointProbeResult, one per layer
    """
    results = []

    for layer in range(n_layers):
        # Prepare data
        X_list = []
        y_list = []
        prompt_ids = []

        for prompt_id, cache in activations.items():
            if prompt_id not in targets:
                continue
            X_list.append(cache.residual_activations[layer])
            y_list.append(targets[prompt_id])
            prompt_ids.append(prompt_id)

        if len(prompt_ids) < 3:
            print(f"  Layer {layer}: Insufficient data ({len(prompt_ids)} samples)")
            results.append(JointProbeResult(
                layer=layer, r2_score=0.0, r2_std=0.0, mse=float('inf')
            ))
            continue

        X = np.stack(X_list)
        y = np.stack(y_list)

        result = train_joint_probe(X, y, layer, cv_folds)
        results.append(result)

        print(f"  Layer {layer:2d}: R² = {result.r2_score:.4f} (±{result.r2_std:.4f})")

    return results


def compare_models_joint(
    base_activations: dict[str, ActivationCache],
    aligned_activations: dict[str, ActivationCache],
    annotations: list[PromptAnnotation],
    n_layers: int,
    cv_folds: int = 5
) -> JointProbeComparison:
    """
    Compare joint multi-output probes between base and aligned models.

    This is the main experiment - like SCOTUS comparing 5 constitutional principles.
    """
    # Prepare joint targets
    targets, _ = prepare_joint_targets(annotations)

    print("\n" + "=" * 60)
    print("JOINT PROBE TRAINING (All Annotation Dimensions)")
    print(f"Dimensions: {ANNOTATION_DIMENSIONS}")
    print("=" * 60)

    print("\n--- BASE MODEL ---")
    base_results = train_joint_all_layers(base_activations, targets, n_layers, cv_folds)

    print("\n--- ALIGNED MODEL ---")
    aligned_results = train_joint_all_layers(aligned_activations, targets, n_layers, cv_folds)

    comparison = JointProbeComparison(
        base_results=base_results,
        aligned_results=aligned_results
    )
    comparison.compute_summary()

    return comparison


def save_joint_comparison(comparison: JointProbeComparison, filepath: str):
    """Save joint comparison results to JSON."""
    data = {
        "dimensions": ANNOTATION_DIMENSIONS,
        "base_results": [
            {
                "layer": r.layer,
                "r2_score": r.r2_score,
                "r2_std": r.r2_std,
                "mse": r.mse,
                "dimension_r2": r.dimension_r2,
                "alpha": r.alpha
            }
            for r in comparison.base_results
        ],
        "aligned_results": [
            {
                "layer": r.layer,
                "r2_score": r.r2_score,
                "r2_std": r.r2_std,
                "mse": r.mse,
                "dimension_r2": r.dimension_r2,
                "alpha": r.alpha
            }
            for r in comparison.aligned_results
        ],
        "summary": {
            "best_base_layer": comparison.best_base_layer,
            "best_base_r2": comparison.best_base_r2,
            "best_aligned_layer": comparison.best_aligned_layer,
            "best_aligned_r2": comparison.best_aligned_r2,
            "r2_difference_by_layer": comparison.r2_difference_by_layer
        }
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved joint comparison to {filepath}")


def plot_joint_comparison(comparison: JointProbeComparison, output_path: Optional[str] = None):
    """Plot joint R² by layer for base vs aligned models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    n_layers = len(comparison.base_results)
    layers = list(range(n_layers))

    base_r2 = [r.r2_score for r in comparison.base_results]
    aligned_r2 = [r.r2_score for r in comparison.aligned_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(layers, base_r2, 'b-o', label='Base Model', markersize=4)
    ax1.plot(layers, aligned_r2, 'r-o', label='Aligned Model', markersize=4)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Cross-validated R²')
    ax1.set_title('Joint Linear Probe: All Transgression Dimensions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    diff = comparison.r2_difference_by_layer
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax2.bar(layers, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('R² Difference (Aligned - Base)')
    ax2.set_title('RLHF Effect by Layer (Green = Improvement)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved joint plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    print("Linear Probe Training Module (Criminal Planning)")
    print("=" * 50)
    print("\nSingle-target regression:")
    for target in TARGET_NAMES:
        print(f"  - {target}")
    print("\nJoint multi-dimensional regression:")
    for dim in ANNOTATION_DIMENSIONS:
        print(f"  - {dim}")
