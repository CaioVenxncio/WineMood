import logging

import numpy as np
import pandas as pd
import pytest

from winemood.data.labels import (
    POINTS_BAND_V0,
    POINTS_BINARY_90_V1,
    POINTS_REGRESSION_V1,
    SCHEMES,
    WE_CATEGORY_MERGED_V1,
    WE_CATEGORY_V1,
    InvalidTargetError,
    assign_labels,
    class_support,
    get_scheme,
    get_target,
    merge_sparse_top_classes,
    regression_target,
    target_report,
)


def test_points_band_v0_matches_legacy_thresholds():
    points = pd.Series([79, 80, 89, 90, 100])
    assert assign_labels(points, POINTS_BAND_V0).tolist() == [0, 1, 1, 2, 2]


def test_points_band_v0_is_flagged_not_for_research():
    assert POINTS_BAND_V0.research_use is False
    assert all(s.research_use for n, s in SCHEMES.items() if n != "points_band_v0")


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (80, 0), (82, 0),  # acceptable
        (83, 1), (86, 1),  # good
        (87, 2), (89, 2),  # very good
        (90, 3), (93, 3),  # excellent
        (94, 4), (97, 4),  # superb
        (98, 5), (100, 5),  # classic
    ],
)  # fmt: skip
def test_we_category_boundaries(points, expected):
    assert assign_labels(pd.Series([points]), WE_CATEGORY_V1).iloc[0] == expected


@pytest.mark.parametrize(("points", "expected"), [(94, 4), (97, 4), (98, 4), (100, 4), (93, 3)])
def test_we_category_merged_boundaries(points, expected):
    assert assign_labels(pd.Series([points]), WE_CATEGORY_MERGED_V1).iloc[0] == expected


@pytest.mark.parametrize(("points", "expected"), [(80, 0), (89, 0), (90, 1), (100, 1)])
def test_binary_90_boundaries(points, expected):
    assert assign_labels(pd.Series([points]), POINTS_BINARY_90_V1).iloc[0] == expected


@pytest.mark.parametrize("scheme", [WE_CATEGORY_V1, WE_CATEGORY_MERGED_V1, POINTS_BINARY_90_V1])
@pytest.mark.parametrize(
    "bad",
    [
        pd.Series([90, None], dtype="float64"),  # missing
        pd.Series([89.5]),  # non-integer
        pd.Series([79]),  # below the published scale
        pd.Series([101]),  # above the scale
        pd.Series([np.inf]),
        pd.Series(["ninety"]),  # non-numeric
    ],
)
def test_invalid_scores_rejected(scheme, bad):
    with pytest.raises(InvalidTargetError):
        assign_labels(bad, scheme)


def test_legacy_scheme_accepts_out_of_range_but_not_missing():
    assert assign_labels(pd.Series([70]), POINTS_BAND_V0).iloc[0] == 0
    with pytest.raises(InvalidTargetError):
        assign_labels(pd.Series([90, None], dtype="float64"), POINTS_BAND_V0)


def test_regression_target_is_identity_with_validation():
    y = regression_target(pd.Series([80, 91, 100]))
    assert y.tolist() == [80.0, 91.0, 100.0]
    assert y.name == "points_regression_v1"
    with pytest.raises(InvalidTargetError):
        regression_target(pd.Series([79]))


def test_empty_negative_class_is_reported(reviews_df, caplog):
    # Mirrors the real dataset: all scores are >= 80, so class 0 never occurs.
    labels = assign_labels(reviews_df["points"], POINTS_BAND_V0)
    with caplog.at_level(logging.WARNING, logger="winemood"):
        support = class_support(labels, POINTS_BAND_V0)
    assert support["negative"] == 0
    assert support["neutral"] > 0 and support["positive"] > 0
    assert "empty classes" in caplog.text


def test_target_report_classification():
    points = pd.Series([80, 85, 85, 88, 91, 91, 91, 95])
    rep = target_report(points, WE_CATEGORY_V1)
    assert rep.support == {
        "acceptable": 1, "good": 2, "very_good": 1, "excellent": 3, "superb": 1, "classic": 0,
    }  # fmt: skip
    assert rep.empty_classes == ["classic"]
    assert rep.imbalance_ratio == 3.0
    assert sum(rep.proportions.values()) == pytest.approx(1.0)


def test_target_report_regression():
    rep = target_report(pd.Series([80, 90, 100]), POINTS_REGRESSION_V1)
    assert rep.task == "regression"
    assert rep.extra["mean"] == 90.0
    assert rep.empty_classes == []


def test_merge_rule_merges_sparse_top_class():
    # 'classic' holds 0.5% < 1%; everything else is well populated.
    labels = pd.Series([0] * 50 + [1] * 300 + [2] * 300 + [3] * 300 + [4] * 45 + [5] * 5)
    merged = merge_sparse_top_classes(labels, WE_CATEGORY_V1, 0.01)
    assert merged.name == WE_CATEGORY_MERGED_V1.name
    assert merged.class_names == WE_CATEGORY_MERGED_V1.class_names
    pts = pd.Series(range(80, 101))
    assert assign_labels(pts, merged).tolist() == assign_labels(pts, WE_CATEGORY_MERGED_V1).tolist()


def test_merge_rule_cascades_and_handles_bottom():
    labels = pd.Series([0] * 1 + [1] * 500 + [2] * 500 + [3] * 3 + [4] * 2 + [5] * 1)
    merged = merge_sparse_top_classes(labels, WE_CATEGORY_V1, 0.01)
    assert merged.class_names == (
        "acceptable_or_good",
        "very_good_or_excellent_or_superb_or_classic",
    )


def test_merge_rule_noop_when_balanced():
    labels = pd.Series(np.repeat(np.arange(6), 100))
    merged = merge_sparse_top_classes(labels, WE_CATEGORY_V1, 0.01)
    assert merged.class_names == WE_CATEGORY_V1.class_names


def test_merge_rejects_non_ordinal():
    with pytest.raises(ValueError):
        merge_sparse_top_classes(pd.Series([0, 1]), POINTS_BINARY_90_V1, 0.01)


def test_lookup():
    assert get_scheme("points_band_v0") is POINTS_BAND_V0
    assert get_target("points_regression_v1") is POINTS_REGRESSION_V1
    with pytest.raises(KeyError):
        get_scheme("sentiment_v1")
