import pandas as pd
import pytest

from winemood.data.labels import POINTS_BAND_V0, assign_labels
from winemood.data.split import (
    SplitResult,
    assert_no_text_overlap,
    deduplicate,
    normalize_for_dedup,
    split_dataset,
)


@pytest.fixture
def labelled(reviews_df):
    df = reviews_df.copy()
    df["label"] = assign_labels(df["points"], POINTS_BAND_V0)
    return df


def test_normalize_for_dedup():
    assert normalize_for_dedup("  Rich   LAYERED\tfruit ") == "rich layered fruit"


def test_deduplicate_removes_normalised_duplicates_and_counts_conflicts(labelled):
    out, report = deduplicate(labelled, "description", "label")
    assert len(out) == 60
    assert report.n_before == 64
    assert report.n_after == 60
    assert report.n_removed == 4
    assert report.n_conflicting_groups == 1


def test_split_is_disjoint_and_sized(labelled):
    result = split_dataset(
        labelled, "description", "label", test_size=0.2, validation_size=0.1, seed=0
    )
    sizes = (len(result.train), len(result.validation), len(result.test))
    assert sum(sizes) == 60
    assert len(result.test) == 12
    assert len(result.validation) == 6
    assert_no_text_overlap(result, "description")


def test_split_is_deterministic(labelled):
    a = split_dataset(labelled, "description", "label", seed=7)
    b = split_dataset(labelled, "description", "label", seed=7)
    assert a.test.index.tolist() == b.test.index.tolist()


def test_split_is_stratified(labelled):
    result = split_dataset(labelled, "description", "label", test_size=0.5, seed=1)
    counts = result.test["label"].value_counts()
    assert abs(counts.get(1, 0) - counts.get(2, 0)) <= 1


def test_overlap_detection():
    df = pd.DataFrame({"description": ["a b", "c"]})
    leaky = SplitResult(train=df, validation=df.iloc[0:0], test=df.iloc[[0]], dedup=None)
    with pytest.raises(AssertionError, match="train and test"):
        assert_no_text_overlap(leaky, "description")
