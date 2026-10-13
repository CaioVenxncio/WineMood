"""Model inference behind a framework-agnostic interface."""

from winemood.inference.predictor import (
    ArtifactError,
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    PredictionResult,
    Predictor,
)

__all__ = [
    "ArtifactError",
    "ArtifactIncompatibleError",
    "ArtifactNotFoundError",
    "PredictionResult",
    "Predictor",
]
