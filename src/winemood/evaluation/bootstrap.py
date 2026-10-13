"""Cluster (group) bootstrap for metric uncertainty and paired model comparison.

Resampling unit: split groups (protocol amendment, 2026-10-12). Metrics are
computed from additive per-group sufficient statistics, so each resample is a
weighted sum and thousands of resamples are cheap. Paired comparisons apply
the same resampled groups to both models.

Supported metrics: MAE, RMSE, bias, R² (regression) and macro-F1, balanced
accuracy (classification). The median absolute error is not additive and is
reported without an interval.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

StatFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n_resamples: int

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "ci95_low": self.low, "ci95_high": self.high}


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return inverse, int(inverse.max()) + 1 if inverse.size else 0


def _group_sums(values: np.ndarray, inverse: np.ndarray, n_groups: int) -> np.ndarray:
    """values: (n_rows, k) -> (n_groups, k) sums."""
    out = np.zeros((n_groups, values.shape[1]))
    np.add.at(out, inverse, values)
    return out


# --- Row-level sufficient statistics and metric functions ----------------------


def regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Columns: n, sum|e|, sum e^2, sum e, sum y, sum y^2."""
    y = np.asarray(y_true, dtype=float)
    p = np.atleast_2d(np.asarray(y_pred, dtype=float))  # (n_seeds, n)
    e = p - y
    return np.column_stack(
        [
            np.ones_like(y),
            np.abs(e).mean(axis=0),  # averaging over seeds gives the mean-of-seeds metric
            (e**2).mean(axis=0),
            e.mean(axis=0),
            y,
            y**2,
        ]
    )


def _mae(s: np.ndarray) -> np.ndarray:
    return s[..., 1] / s[..., 0]


def _rmse(s: np.ndarray) -> np.ndarray:
    return np.sqrt(s[..., 2] / s[..., 0])


def _bias(s: np.ndarray) -> np.ndarray:
    return s[..., 3] / s[..., 0]


def _r2(s: np.ndarray) -> np.ndarray:
    n = s[..., 0]
    ss_tot = s[..., 5] - s[..., 4] ** 2 / n
    return 1.0 - s[..., 2] / ss_tot


REGRESSION_METRICS: dict[str, StatFn] = {"mae": _mae, "rmse": _rmse, "bias": _bias, "r2": _r2}


def classification_stats(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Columns per class k: TP_k, FP_k, FN_k, support_k (4 * n_classes)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    cols = []
    for k in range(n_classes):
        cols += [(p == k) & (y == k), (p == k) & (y != k), (p != k) & (y == k), y == k]
    return np.column_stack(cols).astype(float)


def _macro_f1(s: np.ndarray) -> np.ndarray:
    k = s.shape[-1] // 4
    tp, fp, fn = (s[..., i::4][..., :k] for i in range(3))
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(axis=-1)


def _balanced_accuracy(s: np.ndarray) -> np.ndarray:
    tp, support = s[..., 0::4], s[..., 3::4]
    recall = np.divide(tp, support, out=np.full_like(tp, np.nan), where=support > 0)
    return np.nanmean(recall, axis=-1)


CLASSIFICATION_METRICS: dict[str, StatFn] = {
    "macro_f1": _macro_f1,
    "balanced_accuracy": _balanced_accuracy,
}


# --- Resampling -------------------------------------------------------------------


def _resample_weights(n_groups: int, n_resamples: int, seed: int, chunk: int = 250):
    """Yield (chunk, n_groups) arrays of multinomial group counts."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_resamples:
        size = min(chunk, n_resamples - done)
        idx = rng.integers(0, n_groups, size=(size, n_groups))
        flat = (np.arange(size)[:, None] * n_groups + idx).ravel()
        yield np.bincount(flat, minlength=size * n_groups).reshape(size, n_groups).astype(float)
        done += size


def _percentile_interval(estimate: float, samples: np.ndarray, n: int) -> Interval:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return Interval(float(estimate), float(lo), float(hi), n)


def bootstrap_metric(
    stats: np.ndarray,
    groups: np.ndarray,
    metric: StatFn,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Interval:
    inverse, g = _group_index(groups)
    sums = _group_sums(stats, inverse, g)
    estimate = float(metric(sums.sum(axis=0)))
    samples = np.concatenate([metric(w @ sums) for w in _resample_weights(g, n_resamples, seed)])
    return _percentile_interval(estimate, samples, n_resamples)


def paired_difference(
    stats_a: np.ndarray,
    stats_b: np.ndarray,
    groups: np.ndarray,
    metric: StatFn,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Interval:
    """Interval for metric(A) - metric(B) on the same rows and resampled groups."""
    inverse, g = _group_index(groups)
    sums_a = _group_sums(stats_a, inverse, g)
    sums_b = _group_sums(stats_b, inverse, g)
    estimate = float(metric(sums_a.sum(axis=0)) - metric(sums_b.sum(axis=0)))
    samples = np.concatenate(
        [metric(w @ sums_a) - metric(w @ sums_b) for w in _resample_weights(g, n_resamples, seed)]
    )
    return _percentile_interval(estimate, samples, n_resamples)


def unpaired_difference(
    stats_a: np.ndarray,
    groups_a: np.ndarray,
    stats_b: np.ndarray,
    groups_b: np.ndarray,
    metric: StatFn,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Interval:
    """Interval for metric(A) - metric(B) on different test sets (independent resampling)."""
    inv_a, ga = _group_index(groups_a)
    inv_b, gb = _group_index(groups_b)
    sums_a = _group_sums(stats_a, inv_a, ga)
    sums_b = _group_sums(stats_b, inv_b, gb)
    estimate = float(metric(sums_a.sum(axis=0)) - metric(sums_b.sum(axis=0)))
    samples = np.concatenate(
        [
            metric(wa @ sums_a) - metric(wb @ sums_b)
            for wa, wb in zip(
                _resample_weights(ga, n_resamples, seed),
                _resample_weights(gb, n_resamples, seed + 1),
                strict=True,
            )
        ]
    )
    return _percentile_interval(estimate, samples, n_resamples)
