"""Explicit, versioned target-label schemes.

Every scheme states exactly how a label is derived from the data. Schemes are
never modified in place: a new definition gets a new name, so that results
are always attributable to a specific labelling rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from winemood.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LabelScheme:
    name: str
    description: str
    class_names: tuple[str, ...]
    display_names: tuple[str, ...]
    assign: Callable[[pd.Series], np.ndarray]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _points_band_v0(points: pd.Series) -> np.ndarray:
    # Reproduced verbatim from the original notebook (cell-13).
    values = points.to_numpy(dtype=float)
    return np.where(values >= 90, 2, np.where(values >= 80, 1, 0)).astype(np.int64)


POINTS_BAND_V0 = LabelScheme(
    name="points_band_v0",
    description=(
        "Legacy rule from the original notebook: points >= 90 -> 2, 80-89 -> 1, "
        "< 80 -> 0. Historically called positive/neutral/negative 'sentiment', "
        "but it encodes critic score bands. Class 0 is empty on the Kaggle "
        "dataset, whose scores range from 80 to 100 (see docs/audit.md A-01)."
    ),
    class_names=("negative", "neutral", "positive"),
    display_names=("Negative (< 80 pts)", "Neutral (80-89 pts)", "Positive (>= 90 pts)"),
    assign=_points_band_v0,
)

SCHEMES: dict[str, LabelScheme] = {POINTS_BAND_V0.name: POINTS_BAND_V0}


def get_scheme(name: str) -> LabelScheme:
    try:
        return SCHEMES[name]
    except KeyError:
        raise KeyError(f"Unknown label scheme '{name}'. Available: {sorted(SCHEMES)}") from None


def assign_labels(points: pd.Series, scheme: LabelScheme) -> pd.Series:
    """Apply ``scheme`` to a numeric target series. Missing targets are an error."""
    if points.isna().any():
        raise ValueError("Cannot assign labels: target contains missing values")
    return pd.Series(scheme.assign(points), index=points.index, name="label")


def class_support(labels: pd.Series, scheme: LabelScheme) -> dict[str, int]:
    """Count samples per class and warn loudly about empty classes."""
    counts = labels.value_counts()
    support = {name: int(counts.get(i, 0)) for i, name in enumerate(scheme.class_names)}
    empty = [name for name, n in support.items() if n == 0]
    if empty:
        logger.warning(
            "label scheme has empty classes; a classifier cannot learn them",
            extra={"scheme": scheme.name, "empty_classes": empty, "support": support},
        )
    return support
