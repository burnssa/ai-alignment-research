"""
Linear Probe Training for Constitutional Principle Detection

This module trains linear probes to predict principle weights from residual
stream activations. The core hypothesis: if RLHF creates more linearly
separable representations of constitutional principles, probes trained on
aligned models should achieve higher R² than those on base models.

Key methodology:
1. For each layer, train a linear regression: activations -> principle_weights
2. Use cross-validation to avoid overfitting on small datasets
3. Compare R² between base and aligned models
4. Identify which layers show the strongest effect
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import json

# ML imports
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings


@dataclass
class ProbeResult:
    """Results from training a linear probe on one layer."""
    layer: int
    r2_score: float  # Cross-validated R²
    r2_std: float    # Standard deviation across folds
    mse: float       # Mean squared error
    
    # Per-principle breakdown
    principle_r2: dict[str, float] = field(default_factory=dict)
    
    # Probe weights (for analysis)
    weights: Optional[np.ndarray] = None  # (n_principles, d_model)
    bias: Optional[np.ndarray] = None     # (n_principles,)
    
    # Regularization used
    alpha: float = 1.0


@dataclass 
class ProbeComparison:
    """Comparison of probe performance between base and aligned models."""
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
        # Find best layers
        base_r2s = [r.r2_score for r in self.base_results]
        aligned_r2s = [r.r2_score for r in self.aligned_results]
        
        self.best_base_layer = int(np.argmax(base_r2s))
        self.best_base_r2 = max(base_r2s)
        self.best_aligned_layer = int(np.argmax(aligned_r2s))
        self.best_aligned_r2 = max(aligned_r2s)
        
        # Per-layer difference
        self.r2_difference_by_layer = [
            aligned_r2s[i] - base_r2s[i] 
            for i in range(len(base_r2s))
        ]
    
    def summary_report(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "LINEAR PROBE COMPARISON: Base vs Aligned Model",
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
    Train linear probes to predict principle weights from activations.
    
    Uses Ridge regression with cross-validation for regularization selection.
    """
    
    PRINCIPLE_NAMES = [
        "free_expression",
        "equal_protection", 
        "due_process",
        "federalism",
        "privacy_liberty"
    ]
    
    def __init__(
        self,
        regularization: Literal["ridge", "ridgecv", "none"] = "ridgecv",
        cv_folds: int = 5,
        alphas: Optional[list[float]] = None
    ):
        """
        Initialize probe trainer.
        
        Args:
            regularization: Type of regularization
                - "ridgecv": Ridge with CV alpha selection (recommended)
                - "ridge": Ridge with fixed alpha
                - "none": Ordinary least squares
            cv_folds: Number of cross-validation folds
            alphas: Regularization strengths to try (for ridgecv)
        """
        self.regularization = regularization
        self.cv_folds = cv_folds
        self.alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    def prepare_data(
        self,
        activations: dict,  # case_id -> ActivationCache
        annotations: list,  # List of PrincipleAnnotation
        layer: int
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare X (activations) and y (principle weights) for a specific layer.
        
        Returns:
            X: (n_cases, d_model) activation matrix
            y: (n_cases, n_principles) target weights
            case_ids: List of case IDs in order
        """
        X_list = []
        y_list = []
        case_ids = []
        
        # Create lookup for annotations
        annotation_lookup = {a.case_id: a for a in annotations}
        
        for case_id, cache in activations.items():
            if case_id not in annotation_lookup:
                continue
            
            annotation = annotation_lookup[case_id]
            
            # Get activation at this layer
            act = cache.residual_activations[layer]  # (d_model,)
            X_list.append(act)
            
            # Get principle vector
            y_vec = annotation.to_vector()  # [free_exp, equal_prot, ...]
            y_list.append(y_vec)
            
            case_ids.append(case_id)
        
        X = np.stack(X_list)  # (n_cases, d_model)
        y = np.stack(y_list)  # (n_cases, n_principles)
        
        return X, y, case_ids
    
    def train_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer: int
    ) -> ProbeResult:
        """
        Train a linear probe for one layer with cross-validation.
        
        Args:
            X: (n_cases, d_model) activations
            y: (n_cases, n_principles) targets
            layer: Layer index (for metadata)
        
        Returns:
            ProbeResult with R² scores and probe weights
        """
        n_samples = X.shape[0]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Choose CV strategy based on dataset size
        if n_samples < 10:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=min(self.cv_folds, n_samples), shuffle=True, random_state=42)
        
        # Select model
        if self.regularization == "ridgecv":
            model = RidgeCV(alphas=self.alphas, cv=cv)
        elif self.regularization == "ridge":
            model = Ridge(alpha=1.0)
        else:
            model = LinearRegression()
        
        # Train on full data (for weights) and evaluate via CV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_scaled, y)
        
        # Cross-validated R² for overall prediction
        # Note: We predict all 5 principles jointly
        cv_scores = cross_val_score(
            model, X_scaled, y, 
            cv=cv, 
            scoring='r2'
        )
        
        # Per-principle R² (train separate models for cleaner measurement)
        principle_r2 = {}
        for i, principle in enumerate(self.PRINCIPLE_NAMES):
            y_principle = y[:, i]
            if np.std(y_principle) < 1e-6:
                # No variance in this principle
                principle_r2[principle] = 0.0
            else:
                scores = cross_val_score(
                    Ridge(alpha=model.alpha_ if hasattr(model, 'alpha_') else 1.0),
                    X_scaled, y_principle,
                    cv=cv,
                    scoring='r2'
                )
                principle_r2[principle] = float(np.mean(scores))
        
        # Get final model weights
        weights = model.coef_  # (n_principles, d_model) or (d_model,) for single output
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        
        alpha = model.alpha_ if hasattr(model, 'alpha_') else 1.0
        
        return ProbeResult(
            layer=layer,
            r2_score=float(np.mean(cv_scores)),
            r2_std=float(np.std(cv_scores)),
            mse=float(mean_squared_error(y, model.predict(X_scaled))),
            principle_r2=principle_r2,
            weights=weights,
            bias=model.intercept_,
            alpha=alpha
        )
    
    def train_all_layers(
        self,
        activations: dict,
        annotations: list,
        n_layers: int
    ) -> list[ProbeResult]:
        """
        Train probes for all layers.
        
        Returns:
            List of ProbeResult, one per layer
        """
        results = []
        
        for layer in range(n_layers):
            X, y, case_ids = self.prepare_data(activations, annotations, layer)
            
            if len(case_ids) < 3:
                print(f"  Layer {layer}: Insufficient data ({len(case_ids)} cases)")
                results.append(ProbeResult(
                    layer=layer, r2_score=0.0, r2_std=0.0, mse=float('inf')
                ))
                continue
            
            result = self.train_probe(X, y, layer)
            results.append(result)
            
            print(f"  Layer {layer:2d}: R² = {result.r2_score:.4f} (±{result.r2_std:.4f})")
        
        return results


def compare_models(
    base_activations: dict,
    aligned_activations: dict,
    annotations: list,
    n_layers: int,
    cv_folds: int = 5
) -> ProbeComparison:
    """
    Compare linear probe performance between base and aligned models.
    
    This is the main experiment: does RLHF improve linear separability
    of constitutional principles?
    
    Args:
        base_activations: case_id -> ActivationCache for base model
        aligned_activations: case_id -> ActivationCache for aligned model
        annotations: List of PrincipleAnnotation
        n_layers: Number of layers to probe
        cv_folds: Cross-validation folds
    
    Returns:
        ProbeComparison with results for both models
    """
    trainer = LinearProbeTrainer(cv_folds=cv_folds)
    
    print("\n" + "=" * 50)
    print("Training probes on BASE model...")
    print("=" * 50)
    base_results = trainer.train_all_layers(base_activations, annotations, n_layers)
    
    print("\n" + "=" * 50)
    print("Training probes on ALIGNED model...")
    print("=" * 50)
    aligned_results = trainer.train_all_layers(aligned_activations, annotations, n_layers)
    
    comparison = ProbeComparison(
        base_results=base_results,
        aligned_results=aligned_results
    )
    comparison.compute_summary()
    
    return comparison


def analyze_probe_weights(
    result: ProbeResult,
    top_k: int = 20
) -> dict:
    """
    Analyze which dimensions the probe uses most heavily.
    
    Returns indices and weights of most important dimensions.
    """
    if result.weights is None:
        return {}
    
    analysis = {}
    
    for i, principle in enumerate(LinearProbeTrainer.PRINCIPLE_NAMES):
        weights = result.weights[i]  # (d_model,)
        
        # Find top positive and negative weights
        indices = np.argsort(np.abs(weights))[::-1][:top_k]
        
        analysis[principle] = {
            "top_indices": indices.tolist(),
            "top_weights": weights[indices].tolist(),
            "weight_norm": float(np.linalg.norm(weights)),
            "weight_sparsity": float(np.mean(np.abs(weights) < 0.01))
        }
    
    return analysis


def save_comparison(comparison: ProbeComparison, filepath: str):
    """Save comparison results to JSON."""
    data = {
        "base_results": [
            {
                "layer": r.layer,
                "r2_score": r.r2_score,
                "r2_std": r.r2_std,
                "mse": r.mse,
                "principle_r2": r.principle_r2,
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
                "principle_r2": r.principle_r2,
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


# === Visualization helpers ===

def plot_layer_comparison(comparison: ProbeComparison, output_path: Optional[str] = None):
    """
    Plot R² by layer for base vs aligned models.
    
    Requires matplotlib.
    """
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
    
    # Top plot: R² by layer
    ax1.plot(layers, base_r2, 'b-o', label='Base Model', markersize=4)
    ax1.plot(layers, aligned_r2, 'r-o', label='Aligned Model', markersize=4)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Cross-validated R²')
    ax1.set_title('Linear Probe Performance: Constitutional Principle Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.1, max(max(base_r2), max(aligned_r2)) + 0.1])
    
    # Bottom plot: Difference
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


# === Example usage ===

if __name__ == "__main__":
    print("Linear Probe Training Module")
    print("=" * 50)
    print("\nCore functionality:")
    print("1. LinearProbeTrainer - Train probes layer by layer")
    print("2. compare_models() - Full base vs aligned comparison")
    print("3. plot_layer_comparison() - Visualize results")
    print("\nUsage:")
    print("  trainer = LinearProbeTrainer()")
    print("  results = trainer.train_all_layers(activations, annotations, n_layers)")
    print("  comparison = compare_models(base_act, aligned_act, annotations, n_layers)")
    print("  print(comparison.summary_report())")
