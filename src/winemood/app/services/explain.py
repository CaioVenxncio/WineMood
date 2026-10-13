"""Exact attributions for the linear TF-IDF + Ridge model.

For a review x the model computes  raw(x) = b + sum_j tfidf_j(x) * w_j  over the
features present in x (unigrams and bigrams after the model's own
preprocessing). Each term tfidf_j(x) * w_j is that feature's contribution, so
the contributions plus the intercept reconstruct the raw prediction exactly
(up to float32 storage of the weights). The displayed score additionally clips
the raw value to 80-100.

Contributions describe how *this fitted model* arrives at its number. Features
are correlated, unigrams and bigrams overlap, and TF-IDF weights depend on the
whole review, so a contribution is not the causal effect of a word on a
critic's score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from winemood.evaluation.metrics import clip_scores

RECONSTRUCTION_TOLERANCE = 1e-6  # points, float64 recomputation
STORED_PRECISION_TOLERANCE = 1e-3  # points, vs. the model's float32 prediction path
_NUMBER = re.compile(r"\d")


class ExplanationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Contribution:
    feature: str
    kind: str  # "unigram" | "bigram" | "number"
    tfidf: float
    coefficient: float
    contribution: float


@dataclass(frozen=True)
class Explanation:
    intercept: float
    raw_score: float
    displayed_score: float
    contributions: tuple[Contribution, ...]
    reconstruction_error: float

    @property
    def clipped(self) -> bool:
        return self.raw_score != self.displayed_score


def feature_kind(name: str) -> str:
    if _NUMBER.search(name):
        return "number"
    return "bigram" if " " in name else "unigram"


@lru_cache(maxsize=4)
def feature_names(pipe) -> np.ndarray:
    """Feature names of a loaded pipeline (cached per pipeline object)."""
    return pipe.feature_names()


def explain(pipe, head, text: str) -> Explanation:
    if head.kind not in ("ridge", "ols"):
        raise ExplanationError("Exact attributions are defined for linear regression heads only")
    x = pipe.transform(pd.DataFrame({pipe.text_column: [text]})).tocsr()
    coef = np.asarray(head.weights["coef"], dtype=np.float64).ravel()
    intercept = float(head.weights["intercept"][0])
    idx, values = x.indices, x.data.astype(np.float64)
    contrib = values * coef[idx]
    raw = intercept + float(contrib.sum())
    exact = intercept + float(np.asarray(x.astype(np.float64) @ coef).ravel()[0])
    model_path = float(np.asarray(head.predict(x)).ravel()[0])
    error = abs(raw - exact)
    if error > RECONSTRUCTION_TOLERANCE or abs(raw - model_path) > STORED_PRECISION_TOLERANCE:
        raise ExplanationError(
            f"Attribution does not reconstruct the prediction (|error| = {error:.2e})"
        )
    names = feature_names(pipe)
    rows = sorted(
        (
            Contribution(
                str(names[j]), feature_kind(str(names[j])), float(v), float(coef[j]), float(c)
            )
            for j, v, c in zip(idx, values, contrib, strict=True)
        ),
        key=lambda r: -abs(r.contribution),
    )
    return Explanation(
        intercept=intercept,
        raw_score=raw,
        displayed_score=float(clip_scores(np.array([raw]))[0]),
        contributions=tuple(rows),
        reconstruction_error=error,
    )


def coefficient_table(pipe, head) -> pd.DataFrame:
    """All learned coefficients with their feature kind (for global exploration)."""
    coef = np.asarray(head.weights["coef"], dtype=np.float64).ravel()
    names = feature_names(pipe)
    table = pd.DataFrame({"feature": names.astype(str), "coefficient": coef})
    table["kind"] = table["feature"].map(feature_kind)
    table["direction"] = np.where(table["coefficient"] >= 0, "raises", "lowers")
    return table
