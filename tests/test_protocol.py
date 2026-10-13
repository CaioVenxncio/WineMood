import numpy as np
import pandas as pd
import pytest

from winemood.data.labels import WE_CATEGORY_V1, assign_labels
from winemood.data.protocol import (
    SplitLeakageError,
    assert_split_isolation,
    build_groups,
    config_hash,
    drop_exact_duplicates,
    fingerprint_ids,
    grouped_stratified_assignment,
    load_split_ids,
    run_split_protocol,
    taster_holdout_assignment,
    write_manifest_ids,
)
from winemood.data.split import normalize_for_dedup

NEARDUP = {"shingle_size": 3, "num_perm": 64, "bands": 16, "threshold": 0.8}


def _run(df, strategy="grouped", seed=42, **kw):
    strata = assign_labels(df["points"], WE_CATEGORY_V1).to_numpy()
    return run_split_protocol(
        df,
        name="t",
        text_column="description",
        target_column="points",
        id_column="review_id",
        strata=strata,
        strata_names=WE_CATEGORY_V1.class_names,
        dataset_sha256="x",
        config={"seed": seed},
        seed=seed,
        neardup=NEARDUP,
        strategy=strategy,
        **kw,
    )


def test_exact_dedup_keeps_lowest_id(corpus):
    shuffled = corpus.sample(frac=1, random_state=0)
    kept, stats = drop_exact_duplicates(shuffled, "description", "points", "review_id")
    assert stats.n_removed == 20
    assert stats.n_duplicate_groups == 20
    assert set(range(20)) <= set(kept.review_id)
    assert not set(range(600, 620)) & set(kept.review_id)


def test_dedup_counts_conflicting_scores():
    df = pd.DataFrame(
        {
            "review_id": [0, 1, 2, 3],
            "description": ["a", "A ", "b", "b"],
            "points": [90, 91, 85, 85],
        }
    )
    _, stats = drop_exact_duplicates(df, "description", "points", "review_id")
    assert stats.n_groups_with_conflicting_scores == 1


def test_build_groups_links_text_title_and_clusters():
    df = pd.DataFrame(
        {
            "description": ["x y z", "X  y z", "p q r", "s t u", "v w"],
            "title": ["T1", "T2", "T3", "t3 ", None],
        }
    )
    cluster = np.array([0, 1, 2, 3, 0])
    g = build_groups(df, "description", near_dup_cluster=cluster)
    assert g[0] == g[1] == g[4]  # text link + near-dup cluster link
    assert g[2] == g[3]  # normalised title link
    assert g[0] != g[2]


def test_grouped_split_isolation_and_proportions(corpus):
    result = _run(corpus)
    data = result.data
    counts = data.split.value_counts()
    assert counts["test"] / len(data) == pytest.approx(0.2, abs=0.03)
    assert counts["validation"] / len(data) == pytest.approx(0.1, abs=0.03)
    for column in ("group",):
        assert (data.groupby(column).split.nunique() == 1).all()
    norm = data.description.map(normalize_for_dedup)
    assert (data.groupby(norm).split.nunique() == 1).all()
    assert data.review_id.is_unique


def test_planted_title_and_near_duplicates_share_a_split(corpus):
    data = _run(corpus).data.set_index("review_id")
    for j in range(30, 40):  # shared titles
        assert data.at[j, "split"] == data.at[j + 10, "split"]
    for j in range(20, 30):  # near-duplicates (ids 620..629)
        assert data.at[j, "split"] == data.at[600 + j, "split"]


def test_split_is_deterministic_and_seed_sensitive(corpus):
    a, b = _run(corpus), _run(corpus)
    assert a.manifest.id_sha256 == b.manifest.id_sha256
    c = _run(corpus, seed=7)
    assert c.manifest.id_sha256 != a.manifest.id_sha256


def test_split_independent_of_row_order(corpus):
    a = _run(corpus)
    b = _run(corpus.sample(frac=1, random_state=3))
    assert a.manifest.id_sha256 == b.manifest.id_sha256


def test_isolation_assertion_detects_leaks():
    df = pd.DataFrame({"review_id": [1, 2, 3], "description": ["a", "a", "b"]})
    groups = np.array([0, 1, 2])
    with pytest.raises(SplitLeakageError, match="text"):
        assert_split_isolation(
            df,
            np.array(["train", "test", "train"], dtype=object),
            id_column="review_id",
            text_column="description",
            groups=groups,
        )
    with pytest.raises(SplitLeakageError, match="group"):
        assert_split_isolation(
            df.assign(description=["a", "c", "b"]),
            np.array(["train", "test", "train"], dtype=object),
            id_column="review_id",
            text_column="description",
            groups=np.array([0, 0, 1]),
        )


def test_grouped_assignment_validates_folds():
    with pytest.raises(ValueError):
        grouped_stratified_assignment(np.arange(10), np.zeros(10), n_folds=3, test_folds=2)


def test_taster_holdout(corpus):
    result = _run(corpus, strategy="taster_holdout")
    data = result.data
    assert (data.loc[data.taster_name.isna(), "split"] == "excluded").all()
    assigned = data[data.split != "excluded"]
    tasters_by_split = assigned.groupby("split").taster_name.agg(set)
    assert not tasters_by_split["train"] & tasters_by_split["test"]
    assert not tasters_by_split["train"] & tasters_by_split["validation"]
    assert (assigned.groupby("group").split.nunique() == 1).all()


def test_taster_holdout_excludes_straddling_groups():
    tasters = pd.Series(["A", "B", "C", "D", "E"] * 4)
    groups = np.arange(20)
    groups[1] = 0  # rows 0 (taster A) and 1 (taster B) are the same wine
    out = taster_holdout_assignment(tasters, groups, seed=0)
    if out[0] != out[1]:
        pytest.fail("straddling group must be excluded or kept in a single partition")


def test_fingerprints_and_manifest_files(tmp_path, corpus):
    result = _run(corpus)
    assert fingerprint_ids([3, 1, 2]) == fingerprint_ids([1, 2, 3])
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    write_manifest_ids(tmp_path, result.data, result.data.split.to_numpy(), "review_id")
    test_ids = load_split_ids(tmp_path, "test")
    assert fingerprint_ids(test_ids) == result.manifest.id_sha256["test"]
    assert sum(result.manifest.counts.values()) == len(result.data)
    with pytest.raises(FileNotFoundError):
        load_split_ids(tmp_path / "missing", "train")
