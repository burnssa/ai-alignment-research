#!/usr/bin/env python3
"""
Cross-model layer-by-layer CV R² comparison.

Generates two charts:
1. Absolute CV R² by normalized layer depth for all models (base vs IT)
2. R² delta (IT - base) by normalized layer depth for all models

Usage:
    python plot_cross_model_layer_r2.py

Output saved to results/cross_model/cross_model_layer_r2.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Model configurations: (dir_name, display_name, is_base_notable)
MODELS = {
    "gemma2_27b": {
        "base_name": "Gemma 2 27B (base)",
        "aligned_name": "Gemma 2 27B-IT",
        "family": "gemma",
    },
    "llama31_8b": {
        "base_name": "Llama 3.1 8B (base)",
        "aligned_name": "Llama 3.1 8B-Inst",
        "family": "llama",
    },
    "llama32_3b": {
        "base_name": "Llama 3.2 3B (base)",
        "aligned_name": "Llama 3.2 3B-Inst",
        "family": "llama",
    },
    "mistral_7b": {
        "base_name": "Mistral 7B v0.3 (base)",
        "aligned_name": "Mistral 7B-Inst v0.3",
        "family": "mistral",
    },
    "qwen25_7b": {
        "base_name": "Qwen 2.5 7B (base)",
        "aligned_name": "Qwen 2.5 7B-Inst",
        "family": "qwen",
    },
    "qwen25_32b": {
        "base_name": "Qwen 2.5 32B (base)",
        "aligned_name": "Qwen 2.5 32B-Inst",
        "family": "qwen",
    },
}

# Color scheme: base models get dashed lighter lines, aligned get solid darker lines
# Group by family for visual coherence
FAMILY_COLORS = {
    "gemma": "#e41a1c",   # red
    "llama": "#377eb8",   # blue
    "mistral": "#4daf4a", # green
    "qwen": "#984ea3",    # purple
}

# Second llama/qwen model gets a shifted hue
MODEL_COLORS = {
    "gemma2_27b": "#e41a1c",
    "llama31_8b": "#377eb8",
    "llama32_3b": "#6baed6",
    "mistral_7b": "#4daf4a",
    "qwen25_7b": "#984ea3",
    "qwen25_32b": "#c994c7",
}


def load_probe_comparison(results_dir: str, model_dir: str) -> dict:
    path = os.path.join(results_dir, model_dir, "probe_comparison.json")
    with open(path) as f:
        return json.load(f)


def extract_r2_by_layer(data: dict) -> tuple:
    """Returns (layers, base_r2, aligned_r2, delta)."""
    base_r2 = [r["r2_score"] for r in data["base_results"]]
    aligned_r2 = [r["r2_score"] for r in data["aligned_results"]]
    n_layers = len(base_r2)
    layers = list(range(n_layers))
    delta = [a - b for a, b in zip(aligned_r2, base_r2)]
    return layers, base_r2, aligned_r2, delta


def normalize_layers(layers: list) -> np.ndarray:
    """Normalize layer indices to [0, 1] fractional depth."""
    n = len(layers)
    return np.array(layers) / (n - 1)


def main():
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    output_dir = results_dir / "cross_model"
    output_dir.mkdir(exist_ok=True)

    # Load all model data
    model_data = {}
    for model_dir in MODELS:
        path = results_dir / model_dir / "probe_comparison.json"
        if not path.exists():
            print(f"Warning: {path} not found, skipping {model_dir}")
            continue
        with open(path) as f:
            data = json.load(f)
        layers, base_r2, aligned_r2, delta = extract_r2_by_layer(data)
        norm_layers = normalize_layers(layers)
        model_data[model_dir] = {
            "layers": layers,
            "norm_layers": norm_layers,
            "base_r2": base_r2,
            "aligned_r2": aligned_r2,
            "delta": delta,
            "n_layers": len(layers),
        }

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1.2, 1])

    # --- Top panel: absolute R² by normalized depth ---
    # Plot base lines first (background), then aligned lines (foreground)
    for model_dir, info in MODELS.items():
        if model_dir not in model_data:
            continue
        md = model_data[model_dir]
        color = MODEL_COLORS[model_dir]
        n = md["n_layers"]
        size_label = f" ({n}L)"

        ax1.plot(
            md["norm_layers"], md["base_r2"],
            linestyle="--", color=color, alpha=0.4, linewidth=1.0,
            label=info["base_name"] + size_label,
        )

    for model_dir, info in MODELS.items():
        if model_dir not in model_data:
            continue
        md = model_data[model_dir]
        color = MODEL_COLORS[model_dir]
        n = md["n_layers"]
        size_label = f" ({n}L)"

        ax1.plot(
            md["norm_layers"], md["aligned_r2"],
            linestyle="-", color=color, alpha=0.9, linewidth=2.0,
            label=info["aligned_name"] + size_label,
        )

    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Normalized Layer Depth (0 = first, 1 = last)", fontsize=11)
    ax1.set_ylabel("Cross-Validated R²", fontsize=11)
    ax1.set_title(
        "Constitutional Principle Probe R² by Layer Depth Across Models",
        fontsize=13, fontweight="bold",
    )
    # Clip y-axis: negative R² values are noise floor; focus on the
    # meaningful range where models separate.
    ax1.set_ylim(-0.2, 0.6)
    ax1.annotate(
        "y-axis floored at −0.2 (some base values extend much lower)",
        xy=(0.02, -0.18), fontsize=7, fontstyle="italic", color="gray",
    )

    # Legend: base models first (dashed), then aligned (solid)
    handles, labels = ax1.get_legend_handles_labels()
    n_models = len([m for m in MODELS if m in model_data])
    # Reorder: all base first, then all aligned
    ax1.legend(
        handles, labels,
        loc="lower right", fontsize=7, ncol=2,
        framealpha=0.9, columnspacing=1.0,
        title="Dashed = base, Solid = instruction-tuned", title_fontsize=7.5,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)

    # --- Bottom panel: IT - base delta ---
    for model_dir, info in MODELS.items():
        if model_dir not in model_data:
            continue
        md = model_data[model_dir]
        color = MODEL_COLORS[model_dir]
        short_name = info["aligned_name"].split("(")[0].strip()

        ax2.plot(
            md["norm_layers"], md["delta"],
            linestyle="-", color=color, alpha=0.85, linewidth=1.8,
            label=short_name,
        )

    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    # Shade negative region (base outperforms aligned)
    ymin = min(min(md["delta"]) for md in model_data.values())
    ax2.fill_between(
        [-0.02, 1.02], 0, min(ymin - 0.1, -0.2),
        alpha=0.04, color="red", zorder=0,
    )
    ax2.set_xlabel("Normalized Layer Depth (0 = first, 1 = last)", fontsize=11)
    ax2.set_ylabel("R² Difference (IT − Base)", fontsize=11)
    ax2.set_title(
        "Instruction-Tuning Effect on Probe R² by Layer",
        fontsize=13, fontweight="bold",
    )
    ax2.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    output_path = output_dir / "cross_model_layer_r2.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
