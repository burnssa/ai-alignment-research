"""Central path configuration for scotus-constitutional-geometry.

All scripts import paths from here. When directories move,
only this file needs updating.
"""

from pathlib import Path

ROOT = Path(__file__).parent

# ── Shared data ──────────────────────────────────────────────
DATA_DIR = ROOT / "data"
CASES_DIR = DATA_DIR / "cases"
OPINIONS_DIR = DATA_DIR / "opinions"
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"

# ── Per-model results ────────────────────────────────────────
RESULTS_DIR = ROOT / "results"

MODELS = {
    "gemma2_27b": {
        "results": RESULTS_DIR / "gemma2_27b",
        "activations_aligned": RESULTS_DIR / "gemma2_27b" / "activations" / "aligned",
        "activations_base": RESULTS_DIR / "gemma2_27b" / "activations" / "base",
    },
    "llama32_3b": {
        "results": RESULTS_DIR / "llama32_3b",
        "activations_aligned": RESULTS_DIR / "llama32_3b" / "activations" / "aligned",
        "activations_base": RESULTS_DIR / "llama32_3b" / "activations" / "base",
    },
    "llama31_8b": {
        "results": RESULTS_DIR / "llama31_8b",
        "activations_aligned": RESULTS_DIR / "llama31_8b" / "activations" / "aligned",
        "activations_base": RESULTS_DIR / "llama31_8b" / "activations" / "base",
    },
    "mistral_7b": {
        "results": RESULTS_DIR / "mistral_7b",
        "activations_aligned": RESULTS_DIR / "mistral_7b" / "activations" / "aligned",
        "activations_base": RESULTS_DIR / "mistral_7b" / "activations" / "base",
    },
    "qwen25_7b": {
        "results": RESULTS_DIR / "qwen25_7b",
        "activations_aligned": RESULTS_DIR / "qwen25_7b" / "activations" / "aligned",
        "activations_base": RESULTS_DIR / "qwen25_7b" / "activations" / "base",
    },
    "qwen25_32b": {
        "results": RESULTS_DIR / "qwen25_32b",
        "activations_aligned": RESULTS_DIR / "qwen25_32b" / "activations" / "aligned",
        "activations_base": RESULTS_DIR / "qwen25_32b" / "activations" / "base",
    },
}

# ── Cross-model results ─────────────────────────────────────
CROSS_MODEL_DIR = RESULTS_DIR / "cross_model"


def model_results(model_key):
    """Return results directory for a model."""
    return MODELS[model_key]["results"]


def model_activations(model_key, variant="aligned"):
    """Return activations directory for a model + variant (aligned/base)."""
    return MODELS[model_key][f"activations_{variant}"]
