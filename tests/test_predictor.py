"""Inference tests. TensorFlow-backed tests build a tiny synthetic model."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from winemood.config import ArtifactConfig, PreprocessingConfig
from winemood.data.labels import POINTS_BAND_V0
from winemood.inference.legacy_lstm import LegacyLSTMPredictor, _TokenizerUnpickler
from winemood.inference.predictor import (
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    Predictor,
)
from winemood.preprocessing import TextPreprocessor


def test_missing_artifacts_raise_typed_error(tmp_path: Path):
    artifacts = ArtifactConfig(dir=tmp_path)
    with pytest.raises(ArtifactNotFoundError):
        LegacyLSTMPredictor.from_artifacts(artifacts, POINTS_BAND_V0, PreprocessingConfig())


def test_missing_tokenizer_raises_typed_error(tmp_path: Path):
    (tmp_path / "model.h5").write_bytes(b"")
    with pytest.raises(ArtifactNotFoundError, match="tokenizer"):
        LegacyLSTMPredictor.from_artifacts(
            ArtifactConfig(dir=tmp_path), POINTS_BAND_V0, PreprocessingConfig()
        )


class _Evil:
    def __reduce__(self):
        return (print, ("pwned",))


def test_restricted_unpickler_rejects_arbitrary_globals(tmp_path: Path):
    path = tmp_path / "tokenizer.pickle"
    path.write_bytes(pickle.dumps(_Evil()))
    with path.open("rb") as fh, pytest.raises(pickle.UnpicklingError):
        _TokenizerUnpickler(fh).load()


# --- TensorFlow-backed tests -------------------------------------------------

tf_keras = pytest.importorskip("tf_keras")


def _tiny_model(n_classes: int, vocab: int = 50, seq_len: int = 8):
    model = tf_keras.Sequential(
        [
            tf_keras.layers.Input(shape=(seq_len,)),
            tf_keras.layers.Embedding(vocab, 4),
            tf_keras.layers.LSTM(4),
            tf_keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


@pytest.fixture
def legacy_artifacts(tmp_path: Path, stopwords):
    from tf_keras.preprocessing.text import Tokenizer

    pre = TextPreprocessor(stopwords=stopwords)
    tokenizer = Tokenizer(num_words=50)
    tokenizer.fit_on_texts(pre.batch(["rich dark fruit long finish", "light citrus simple"]))
    _tiny_model(3).save(tmp_path / "model.h5")
    with (tmp_path / "tokenizer.pickle").open("wb") as fh:
        pickle.dump(tokenizer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ArtifactConfig(dir=tmp_path)


def test_roundtrip_legacy_artifacts(legacy_artifacts, stopwords):
    predictor = LegacyLSTMPredictor.from_artifacts(
        legacy_artifacts, POINTS_BAND_V0, PreprocessingConfig(max_sequence_length=8)
    )
    predictor.preprocessor = TextPreprocessor(stopwords=stopwords)
    assert isinstance(predictor, Predictor)

    results = predictor.predict(["Rich dark fruit.", "zzz qqq"])
    assert len(results) == 2
    for r in results:
        assert len(r.probabilities) == 3
        assert np.isclose(sum(r.probabilities), 1.0, atol=1e-5)
        assert r.label == POINTS_BAND_V0.class_names[r.label_index]
    assert results[0].n_known_tokens == 3
    assert results[1].n_known_tokens == 0
    assert predictor.predict([]) == []


def test_keras2_era_tokenizer_pickle_is_remapped():
    """A pickle referencing the Keras 2 module path fails with plain pickle under
    Keras 3 but loads through the restricted unpickler (audit A-16)."""
    import io

    from tf_keras.preprocessing.text import Tokenizer

    tok = Tokenizer(num_words=10)
    tok.fit_on_texts(["rich dark fruit"])
    # Protocol 2 stores class references as text (GLOBAL opcode), so the module
    # path can be rewritten in place.
    data = pickle.dumps(tok, protocol=2, fix_imports=False).replace(
        b"tf_keras.src.preprocessing.text", b"keras.preprocessing.text"
    )
    assert b"ckeras.preprocessing.text\nTokenizer" in data
    with pytest.raises((ImportError, AttributeError)):
        pickle.loads(data)
    loaded = _TokenizerUnpickler(io.BytesIO(data)).load()
    assert loaded.texts_to_sequences(["dark fruit"]) == tok.texts_to_sequences(["dark fruit"])


def test_output_size_mismatch_is_incompatible(stopwords):
    with pytest.raises(ArtifactIncompatibleError, match="2 outputs"):
        LegacyLSTMPredictor(
            _tiny_model(2), object(), POINTS_BAND_V0, TextPreprocessor(stopwords=stopwords)
        )


def test_corrupt_model_is_incompatible(tmp_path: Path, legacy_artifacts):
    (legacy_artifacts.dir / "model.h5").write_bytes(b"not an hdf5 file")
    with pytest.raises(ArtifactIncompatibleError):
        LegacyLSTMPredictor.from_artifacts(legacy_artifacts, POINTS_BAND_V0, PreprocessingConfig())
