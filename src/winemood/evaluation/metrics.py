"""Evaluation metrics for score regression (T1) and score-category classification (T2/T3).

Regression and classification metrics measure different tasks and are reported
separately; they are never compared with each other directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from winemood.data.labels import POINTS_MAX, POINTS_MIN, LabelScheme

SCORE_RANGE = (POINTS_MIN, POINTS_MAX)


def clip_scores(pred: np.ndarray) -> np.ndarray:
    """All regression predictions are clipped to the published 80-100 scale."""
    return np.clip(np.asarray(pred, dtype=float), *SCORE_RANGE)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    err = p - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_ae": float(np.median(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),  # mean(pred - true); > 0 means overestimation
        "within_1": float(np.mean(np.abs(err) <= 1.0)),
        "within_2": float(np.mean(np.abs(err) <= 2.0)),
        "pred_std": float(np.std(p)),
        "true_std": float(np.std(y)),
    }


def scores_to_classes(pred_scores: np.ndarray, scheme: LabelScheme) -> np.ndarray:
    """Derive class predictions from regression output: round, clip, apply scheme."""
    rounded = np.clip(np.rint(np.asarray(pred_scores, dtype=float)), *SCORE_RANGE)
    return scheme.assign(pd.Series(rounded))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scheme: LabelScheme,
    proba: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = list(range(scheme.n_classes))
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    prec, rec, f1, support = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    out: dict[str, Any] = {
        "n": int(y.size),
        "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "accuracy": float(np.mean(y == p)),
        "per_class": {
            name: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(scheme.class_names)
        },
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
    }
    if scheme.task == "ordinal":
        out["qwk"] = float(cohen_kappa_score(y, p, labels=labels, weights="quadratic"))
        out["mae_class_index"] = float(np.mean(np.abs(y - p)))
    if proba is not None and scheme.n_classes == 2:
        positive = np.asarray(proba, dtype=float)
        positive = positive[:, 1] if positive.ndim == 2 else positive
        out["brier"] = float(np.mean((positive - y) ** 2))
    return out
