import numpy as np
import pytest

from winemood.data.labels import POINTS_BINARY_90_V1, WE_CATEGORY_MERGED_V1
from winemood.evaluation.bootstrap import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    bootstrap_metric,
    classification_stats,
    paired_difference,
    regression_stats,
    unpaired_difference,
)
from winemood.evaluation.metrics import (
    classification_metrics,
    clip_scores,
    regression_metrics,
    scores_to_classes,
)


def test_regression_metrics_hand_computed():
    m = regression_metrics(np.array([88, 90, 92]), np.array([89, 90, 90]))
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["median_ae"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(-1 / 3)
    assert m["r2"] == pytest.approx(1 - 5 / 8)
    assert m["within_1"] == pytest.approx(2 / 3)


def test_clip_and_derived_classes():
    assert clip_scores(np.array([75.0, 85.2, 104.0])).tolist() == [80.0, 85.2, 100.0]
    pred = np.array([82.4, 82.6, 89.49, 89.5, 99.0])
    assert scores_to_classes(pred, WE_CATEGORY_MERGED_V1).tolist() == [0, 1, 2, 3, 4]
    assert scores_to_classes(pred, POINTS_BINARY_90_V1).tolist() == [0, 0, 0, 1, 1]


def test_classification_metrics_ordinal_and_binary():
    y = np.array([0, 1, 2, 3, 4, 4])
    p = np.array([0, 1, 2, 3, 3, 4])
    m = classification_metrics(y, p, WE_CATEGORY_MERGED_V1)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["per_class"]["superb_or_classic"]["recall"] == pytest.approx(0.5)
    assert m["mae_class_index"] == pytest.approx(1 / 6)
    assert "qwk" in m and m["qwk"] > 0.9
    assert np.array(m["confusion_matrix"]).sum() == 6
    b = classification_metrics(
        np.array([0, 1]),
        np.array([0, 1]),
        POINTS_BINARY_90_V1,
        proba=np.array([[0.9, 0.1], [0.2, 0.8]]),
    )
    assert b["brier"] == pytest.approx((0.1**2 + 0.2**2) / 2)


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(1)
    y = rng.integers(80, 101, size=400).astype(float)
    a = y + rng.normal(scale=2.0, size=400)
    b = y + rng.normal(scale=3.0, size=400)
    groups = np.repeat(np.arange(200), 2)  # pairs of rows share a group
    return y, a, b, groups


def test_bootstrap_estimate_equals_metric_and_is_deterministic(reg_data):
    y, a, _, groups = reg_data
    stats = regression_stats(y, a)
    ci = bootstrap_metric(stats, groups, REGRESSION_METRICS["mae"], n_resamples=500, seed=3)
    assert ci.estimate == pytest.approx(regression_metrics(y, a)["mae"])
    assert ci.low < ci.estimate < ci.high
    again = bootstrap_metric(stats, groups, REGRESSION_METRICS["mae"], n_resamples=500, seed=3)
    assert (again.low, again.high) == (ci.low, ci.high)
    r2 = bootstrap_metric(stats, groups, REGRESSION_METRICS["r2"], n_resamples=200)
    assert r2.estimate == pytest.approx(regression_metrics(y, a)["r2"])


def test_paired_difference(reg_data):
    y, a, b, groups = reg_data
    sa, sb = regression_stats(y, a), regression_stats(y, b)
    same = paired_difference(sa, sa, groups, REGRESSION_METRICS["mae"], n_resamples=200)
    assert same.estimate == 0 and same.low == 0 and same.high == 0
    diff = paired_difference(sa, sb, groups, REGRESSION_METRICS["mae"], n_resamples=500)
    assert diff.estimate < 0 and diff.excludes_zero()  # a is clearly better than b


def test_multi_seed_stats_average_seeds(reg_data):
    y, a, b, groups = reg_data
    stats = regression_stats(y, np.vstack([a, b]))
    ci = bootstrap_metric(stats, groups, REGRESSION_METRICS["mae"], n_resamples=50)
    expected = (regression_metrics(y, a)["mae"] + regression_metrics(y, b)["mae"]) / 2
    assert ci.estimate == pytest.approx(expected)


def test_unpaired_difference(reg_data):
    y, a, b, groups = reg_data
    diff = unpaired_difference(
        regression_stats(y, b),
        groups,
        regression_stats(y, a),
        groups,
        REGRESSION_METRICS["mae"],
        n_resamples=300,
    )
    assert diff.estimate > 0 and diff.low < diff.estimate < diff.high


def test_classification_bootstrap_matches_sklearn():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, size=300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 5, size=300))
    groups = np.arange(300)
    stats = classification_stats(y, p, 5)
    expected = classification_metrics(y, p, WE_CATEGORY_MERGED_V1)
    for name, fn in CLASSIFICATION_METRICS.items():
        ci = bootstrap_metric(stats, groups, fn, n_resamples=100)
        assert ci.estimate == pytest.approx(expected[name])
