"""
Criminal Planning Geometry Experiment

Tests whether activation geometry predicts:
1. How transgressive a prompt is (Claude annotation)
2. How transgressive the response is (Patronus scoring)
3. The delta between these (measuring "restraint")
"""

from .schemas import (
    PromptAnnotation,
    PatronusScore,
    ActivationCache,
)

__all__ = [
    "PromptAnnotation",
    "PatronusScore",
    "ActivationCache",
]
