"""Predictor interface and result types shared by all model backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from winemood.data.labels import LabelScheme


class ArtifactError(RuntimeError):
    """Base class for problems with model artifacts."""


class ArtifactNotFoundError(ArtifactError):
    """A required artifact file does not exist."""


class ArtifactIncompatibleError(ArtifactError):
    """An artifact exists but cannot be used with this code or label scheme."""


@dataclass(frozen=True)
class PredictionResult:
    text: str
    label_index: int
    label: str
    display_label: str
    probabilities: tuple[float, ...]
    # Number of input tokens found in the model vocabulary. Zero means the
    # model saw only padding and its output carries no information.
    n_known_tokens: int


@runtime_checkable
class Predictor(Protocol):
    scheme: LabelScheme

    def predict(self, texts: Sequence[str]) -> list[PredictionResult]: ...
