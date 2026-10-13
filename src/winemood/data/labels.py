"""Explicit, versioned target definitions.

Every scheme states exactly how a target is derived from the data. Schemes are
never modified in place: a new definition gets a new name, so that results
are always attributable to a specific rule. See docs/problem-formulation.md.

None of these targets is a sentiment label. They are all functions of the
critic's own 80-100 score.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from winemood.logging_utils import get_logger

logger = get_logger(__name__)

POINTS_MIN, POINTS_MAX = 80, 100

TaskType = Literal["classification", "ordinal", "regression"]


class InvalidTargetError(ValueError):
    """Raised when scores are missing, non-integer or outside the published scale."""


def validate_points(points: pd.Series, *, strict_range: bool = True) -> None:
    """Reject missing, non-numeric, non-integer and (optionally) out-of-range scores."""
    if not pd.api.types.is_numeric_dtype(points):
        raise InvalidTargetError(f"Scores must be numeric, got dtype {points.dtype}")
    if points.isna().any():
        raise InvalidTargetError(f"{int(points.isna().sum())} missing score(s)")
    values = points.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values != np.round(values)):
        raise InvalidTargetError("Scores must be finite integers")
    if strict_range:
        bad = (values < POINTS_MIN) | (values > POINTS_MAX)
        if bad.any():
            raise InvalidTargetError(
                f"{int(bad.sum())} score(s) outside the published {POINTS_MIN}-{POINTS_MAX} scale"
            )


@dataclass(frozen=True)
class LabelScheme:
    """A rule mapping integer scores to class indices."""

    name: str
    description: str
    class_names: tuple[str, ...]
    display_names: tuple[str, ...]
    assign: Callable[[pd.Series], np.ndarray]
    task: TaskType = "classification"
    research_use: bool = True
    strict_range: bool = True

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True)
class RegressionTarget:
    """A rule mapping integer scores to a continuous target."""

    name: str
    description: str
    task: TaskType = "regression"
    research_use: bool = True


def _bands(lower_bounds: Sequence[int]) -> Callable[[pd.Series], np.ndarray]:
    """Class i covers [lower_bounds[i], lower_bounds[i+1]). Bounds must ascend."""
    bounds = np.asarray(lower_bounds[1:], dtype=float)

    def assign(points: pd.Series) -> np.ndarray:
        return np.searchsorted(bounds, points.to_numpy(dtype=float), side="right").astype(np.int64)

    return assign


def _points_band_v0(points: pd.Series) -> np.ndarray:
    # Reproduced verbatim from the original notebook (cell-13).
    values = points.to_numpy(dtype=float)
    return np.where(values >= 90, 2, np.where(values >= 80, 1, 0)).astype(np.int64)


POINTS_BAND_V0 = LabelScheme(
    name="points_band_v0",
    description=(
        "DEPRECATED legacy rule from the original notebook: points >= 90 -> 2, "
        "80-89 -> 1, < 80 -> 0. Historically called positive/neutral/negative "
        "'sentiment', but it encodes critic score bands. Class 0 cannot occur: "
        "Wine Enthusiast does not review wines below 80 (docs/audit.md A-01). "
        "Kept only so the legacy model can be served; not a research target."
    ),
    class_names=("negative", "neutral", "positive"),
    display_names=("Negative (< 80 pts)", "Neutral (80-89 pts)", "Positive (>= 90 pts)"),
    assign=_points_band_v0,
    research_use=False,
    strict_range=False,
)

# Wine Enthusiast's published rating categories (verified 2026-10-12; see
# docs/problem-formulation.md for sources).
WE_CATEGORY_BOUNDS = (80, 83, 87, 90, 94, 98)
WE_CATEGORY_NAMES = ("acceptable", "good", "very_good", "excellent", "superb", "classic")
WE_CATEGORY_DISPLAY = (
    "Acceptable (80-82)",
    "Good (83-86)",
    "Very Good (87-89)",
    "Excellent (90-93)",
    "Superb (94-97)",
    "Classic (98-100)",
)

WE_CATEGORY_V1 = LabelScheme(
    name="we_category_v1",
    description=(
        "Wine Enthusiast's six published rating categories: 80-82 Acceptable, "
        "83-86 Good, 87-89 Very Good, 90-93 Excellent, 94-97 Superb, 98-100 Classic. "
        "Thresholds are defined by the publisher, not chosen by this project. Ordered."
    ),
    class_names=WE_CATEGORY_NAMES,
    display_names=WE_CATEGORY_DISPLAY,
    assign=_bands(WE_CATEGORY_BOUNDS),
    task="ordinal",
)

WE_CATEGORY_MERGED_V1 = LabelScheme(
    name="we_category_merged_v1",
    description=(
        "we_category_v1 with 'classic' (98-100) merged into 'superb' (94-97), frozen "
        "after applying the pre-declared rule in merge_sparse_top_classes "
        "(min_class_fraction = 1%) to the training split of grouped_v1, where "
        "'classic' holds about 0.11% of rows. The EDA re-derives the rule and checks "
        "that it still yields these classes. Ordered; five classes."
    ),
    class_names=("acceptable", "good", "very_good", "excellent", "superb_or_classic"),
    display_names=(*WE_CATEGORY_DISPLAY[:4], "Superb or Classic (94-100)"),
    assign=_bands(WE_CATEGORY_BOUNDS[:5]),
    task="ordinal",
)

POINTS_BINARY_90_V1 = LabelScheme(
    name="points_binary_90_v1",
    description=(
        "1 if points >= 90 (Wine Enthusiast 'Excellent' or above), else 0. "
        "Experimental candidate: simple and balanced, but discards within-band "
        "information and separates 89 from 90 sharply."
    ),
    class_names=("below_90", "at_least_90"),
    display_names=("< 90 pts", ">= 90 pts"),
    assign=_bands((80, 90)),
)

POINTS_REGRESSION_V1 = RegressionTarget(
    name="points_regression_v1",
    description=(
        "Primary research target: the integer critic score (80-100) used directly "
        "as a continuous target. No thresholds."
    ),
)

SCHEMES: dict[str, LabelScheme] = {
    s.name: s for s in (POINTS_BAND_V0, WE_CATEGORY_V1, WE_CATEGORY_MERGED_V1, POINTS_BINARY_90_V1)
}
REGRESSION_TARGETS: dict[str, RegressionTarget] = {POINTS_REGRESSION_V1.name: POINTS_REGRESSION_V1}


def get_scheme(name: str) -> LabelScheme:
    try:
        return SCHEMES[name]
    except KeyError:
        raise KeyError(f"Unknown label scheme '{name}'. Available: {sorted(SCHEMES)}") from None


def get_target(name: str) -> LabelScheme | RegressionTarget:
    if name in REGRESSION_TARGETS:
        return REGRESSION_TARGETS[name]
    return get_scheme(name)


def assign_labels(points: pd.Series, scheme: LabelScheme) -> pd.Series:
    """Apply ``scheme`` to an integer score series after validating it."""
    validate_points(points, strict_range=scheme.strict_range)
    return pd.Series(scheme.assign(points), index=points.index, name="label")


def regression_target(points: pd.Series, target: RegressionTarget = POINTS_REGRESSION_V1):
    validate_points(points)
    return points.astype(float).rename(target.name)


# --- Pre-declared merge rule ----------------------------------------------------


def merge_sparse_top_classes(
    train_labels: pd.Series, scheme: LabelScheme, min_fraction: float
) -> LabelScheme:
    """Derive ``<scheme>_merged`` by merging sparse classes into their lower neighbour.

    Rule (declared before looking at data): starting from the highest class,
    any class holding less than ``min_fraction`` of the *training* labels is
    merged into the class immediately below it; repeat until all classes pass.
    The same is applied from the bottom upward. Only adjacent classes of an
    ordered scheme are merged, so the order is preserved.
    """
    if scheme.task != "ordinal":
        raise ValueError("Merging is only defined for ordinal schemes")
    counts = train_labels.value_counts().reindex(range(scheme.n_classes), fill_value=0)
    groups = [[i] for i in range(scheme.n_classes)]
    total = float(counts.sum())

    def frac(g: list[int]) -> float:
        return float(counts[g].sum()) / total if total else 0.0

    while len(groups) > 1 and frac(groups[-1]) < min_fraction:
        groups[-2].extend(groups.pop())
    while len(groups) > 1 and frac(groups[0]) < min_fraction:
        lowest = groups.pop(0)
        groups[0][:0] = lowest

    mapping = np.empty(scheme.n_classes, dtype=np.int64)
    for new, members in enumerate(groups):
        mapping[members] = new
    names = tuple("_or_".join(scheme.class_names[i] for i in g) for g in groups)
    display = tuple(_merge_display(scheme, g) for g in groups)
    base_assign = scheme.assign

    def assign(points: pd.Series) -> np.ndarray:
        return mapping[base_assign(points)]

    return LabelScheme(
        name=f"{scheme.name.removesuffix('_v1')}_merged_v1",
        description=(
            f"{scheme.name} with adjacent sparse classes merged (each class must hold "
            f">= {min_fraction:.1%} of the training split). Groups: " + "; ".join(names)
        ),
        class_names=names,
        display_names=display,
        assign=assign,
        task=scheme.task,
    )


def _merge_display(scheme: LabelScheme, members: list[int]) -> str:
    if len(members) == 1:
        return scheme.display_names[members[0]]
    label = " or ".join(scheme.class_names[i].replace("_", " ").title() for i in members)
    if scheme is not WE_CATEGORY_V1:
        return label
    upper_bounds = (*WE_CATEGORY_BOUNDS, POINTS_MAX + 1)
    return f"{label} ({upper_bounds[members[0]]}-{upper_bounds[members[-1] + 1] - 1})"


# --- Distribution reporting -----------------------------------------------------


@dataclass
class TargetReport:
    scheme: str
    task: str
    n: int
    support: dict[str, int]
    proportions: dict[str, float]
    empty_classes: list[str]
    imbalance_ratio: float | None  # largest / smallest non-empty class
    extra: dict[str, Any] = field(default_factory=dict)


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


def target_report(points: pd.Series, scheme: LabelScheme | RegressionTarget) -> TargetReport:
    """Distribution summary for any target, with empty-class and imbalance checks."""
    if isinstance(scheme, RegressionTarget):
        y = regression_target(points, scheme)
        values = y.value_counts().sort_index()
        return TargetReport(
            scheme=scheme.name,
            task=scheme.task,
            n=len(y),
            support={str(int(k)): int(v) for k, v in values.items()},
            proportions={str(int(k)): float(v / len(y)) for k, v in values.items()},
            empty_classes=[],
            imbalance_ratio=None,
            extra={
                "mean": float(y.mean()),
                "std": float(y.std()),
                "median": float(y.median()),
                "min": float(y.min()),
                "max": float(y.max()),
            },
        )
    labels = assign_labels(points, scheme)
    support = class_support(labels, scheme)
    n = len(labels)
    non_empty = [v for v in support.values() if v > 0]
    return TargetReport(
        scheme=scheme.name,
        task=scheme.task,
        n=n,
        support=support,
        proportions={k: (v / n if n else 0.0) for k, v in support.items()},
        empty_classes=[k for k, v in support.items() if v == 0],
        imbalance_ratio=(max(non_empty) / min(non_empty)) if non_empty else None,
    )
