import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from winemood.features.metadata import MetadataEncoder
from winemood.features.pipeline import FeaturePipeline, FeatureSpec
from winemood.features.text import SafeTfidf, length_features, sequence_text
from winemood.preprocessing import TextPreprocessor, mask_numbers, preprocessing_config

DOCS = [
    "Ripe cherry and plum, cherry again. Drink 2018-2025.",
    "Dense tannins and a long finish; not simple.",
    "Light, crisp citrus with 12% alcohol.",
    "Cherry, plum and oak from the 1998 vintage.",
]


def test_mask_numbers_modes():
    text = "Drink 2018-2025; 12% alcohol, 1998 vintage, 3 years"
    assert mask_numbers(text, "none") == text
    years = mask_numbers(text, "years")
    assert "2018" not in years and "1998" not in years and "12%" in years
    assert years.count("__year__") == 3
    everything = mask_numbers(text, "all")
    assert not any(ch.isdigit() for ch in everything)
    with pytest.raises(ValueError):
        mask_numbers(text, "some")


def test_preprocessing_versions(stopwords):
    text = "The wine is NOT balanced, drink 2020."
    minimal = TextPreprocessor(preprocessing_config("norm_min_v1"), stopwords=stopwords)
    legacy = TextPreprocessor(preprocessing_config("norm_legacy_v0"), stopwords=stopwords)
    keep = TextPreprocessor(preprocessing_config("keep_negations"), stopwords=stopwords)
    masked = TextPreprocessor(preprocessing_config("norm_min_v1", "years"), stopwords=stopwords)
    assert minimal(text) == "the wine is not balanced, drink 2020."
    assert legacy(text) == "wine balanced drink 2020"
    assert keep(text) == "wine not balanced drink 2020"
    assert "__year__" in masked(text) and "2020" not in masked(text)
    with pytest.raises(ValueError):
        preprocessing_config("norm_unknown")


def test_placeholder_survives_legacy_pipeline(stopwords):
    legacy_masked = TextPreprocessor(
        preprocessing_config("norm_legacy_v0", "years"), stopwords=stopwords
    )
    assert "__year__" in legacy_masked("Drink 2020-2025.").split()


@pytest.mark.parametrize(("analyzer", "ngrams"), [("word", (1, 2)), ("char_wb", (3, 5))])
def test_safe_tfidf_matches_sklearn_and_roundtrips(tmp_path, analyzer, ngrams):
    reference = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngrams,
        min_df=1,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    ).fit(DOCS)
    safe = SafeTfidf(analyzer=analyzer, ngram_range=ngrams, min_df=1).fit(DOCS)
    new = ["cherry plum unseen words", "tannins finish 2020"]
    np.testing.assert_allclose(
        safe.transform(new).toarray(), reference.transform(new).toarray(), atol=1e-6
    )
    safe.save(tmp_path)
    assert not list(tmp_path.glob("*.pkl")) and not list(tmp_path.glob("*.pickle"))
    loaded = SafeTfidf.load(tmp_path)
    np.testing.assert_array_equal(loaded.transform(new).toarray(), safe.transform(new).toarray())


def test_vocabulary_fitted_on_training_only():
    train = pd.DataFrame({"description": ["cherry plum oak", "cherry tannin finish"]})
    val = pd.DataFrame({"description": ["validationonlyword cherry"]})
    pipe = FeaturePipeline(FeatureSpec(word={"ngram_range": [1, 1], "min_df": 1})).fit(train)
    assert "validationonlyword" not in pipe.word.vocabulary_
    x = pipe.transform(val)
    assert x.shape == (1, pipe.word.n_features)


def test_length_features():
    feats = length_features(["one two three", "a"])
    assert feats.tolist() == [[3.0, 13.0], [1.0, 1.0]]
    assert sequence_text("Ripe, cherry __year__!") == "ripe cherry __year__"


def test_metadata_encoder_handles_unseen_and_missing(tmp_path, corpus):
    enc = MetadataEncoder({"taster_name": 3, "country": 2, "variety": 1}).fit(corpus)
    frame = pd.DataFrame(
        {
            "taster_name": [None, "Unknown"],
            "country": ["Mars", "US"],
            "variety": ["X", None],
            "price": [np.nan, 20.0],
        }
    )
    x = enc.transform(frame)
    assert x.shape == (2, (3 + 1) + (2 + 1) + (1 + 1) + 2)
    assert x[0, -1] == 1.0  # price missing flag
    enc.save(tmp_path / "m.json")
    json.loads((tmp_path / "m.json").read_text())
    np.testing.assert_array_equal(MetadataEncoder.load(tmp_path / "m.json").transform(frame), x)


@pytest.mark.parametrize(
    "spec",
    [
        FeatureSpec(word={"ngram_range": [1, 2], "min_df": 1}, add_length=True),
        FeatureSpec(word=None, char={"ngram_range": [3, 4], "min_df": 1}, mask_numbers="years"),
        FeatureSpec(kind="length"),
        FeatureSpec(kind="metadata"),
    ],
)
def test_pipeline_roundtrip(tmp_path, corpus, spec):
    train, test = corpus.iloc[:400], corpus.iloc[400:]
    pipe = FeaturePipeline(spec).fit(train)
    pipe.save(tmp_path)
    loaded = FeaturePipeline.load(tmp_path)
    a, b = pipe.transform(test), loaded.transform(test)
    a = a.toarray() if hasattr(a, "toarray") else a
    b = b.toarray() if hasattr(b, "toarray") else b
    np.testing.assert_allclose(a, b)
