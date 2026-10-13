import logging

import pandas as pd
import pytest

from winemood.data.labels import POINTS_BAND_V0, assign_labels, class_support, get_scheme


def test_points_band_v0_matches_legacy_thresholds():
    points = pd.Series([79, 79.9, 80, 89, 89.9, 90, 100])
    assert assign_labels(points, POINTS_BAND_V0).tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_missing_points_rejected():
    with pytest.raises(ValueError):
        assign_labels(pd.Series([90, None]), POINTS_BAND_V0)


def test_empty_negative_class_is_reported(reviews_df, caplog):
    # Mirrors the real dataset: all scores are >= 80, so class 0 never occurs.
    labels = assign_labels(reviews_df["points"], POINTS_BAND_V0)
    with caplog.at_level(logging.WARNING, logger="winemood"):
        support = class_support(labels, POINTS_BAND_V0)
    assert support["negative"] == 0
    assert support["neutral"] > 0 and support["positive"] > 0
    assert "empty classes" in caplog.text


def test_unknown_scheme():
    assert get_scheme("points_band_v0") is POINTS_BAND_V0
    with pytest.raises(KeyError):
        get_scheme("sentiment_v1")
