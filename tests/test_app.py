"""Headless end-to-end tests of the Streamlit UI (streamlit.testing.AppTest)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

tf_keras = pytest.importorskip("tf_keras")


def _run_app(config_path: str) -> None:
    from winemood.ui.streamlit_app import main

    main(config_path)


def _write_config(tmp_path: Path, artifacts_dir: Path) -> str:
    # Stopword removal is disabled so the test does not need NLTK data.
    cfg = tmp_path / "test.toml"
    cfg.write_text(
        f'[artifacts]\ndir = "{artifacts_dir.as_posix()}"\n\n'
        "[preprocessing]\nremove_stopwords = false\nmax_sequence_length = 12\n",
        encoding="utf-8",
    )
    return str(cfg)


def _app(config_path: str) -> AppTest:
    at = AppTest.from_function(_run_app, args=(config_path,), default_timeout=120)
    at.run()
    return at


def test_missing_artifacts_show_guidance_not_traceback(tmp_path: Path):
    at = _app(_write_config(tmp_path, tmp_path / "empty"))
    assert not at.exception
    assert any("No usable model" in e.value for e in at.error)
    assert at.button[0].disabled


def test_prediction_with_synthetic_artifacts(tmp_path: Path):
    from tf_keras.preprocessing.text import Tokenizer

    art = tmp_path / "art"
    art.mkdir()
    model = tf_keras.Sequential(
        [
            tf_keras.layers.Input(shape=(12,)),
            tf_keras.layers.Embedding(20, 4),
            tf_keras.layers.GlobalAveragePooling1D(),
            tf_keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.save(art / "model.h5")
    tok = Tokenizer(num_words=20)
    tok.fit_on_texts(["rich dark fruit long finish", "light citrus"])
    (art / "tokenizer.json").write_text(tok.to_json(), encoding="utf-8")

    at = _app(_write_config(tmp_path, art))
    assert not at.exception
    assert not at.error
    assert not at.button[0].disabled

    at.text_area[0].input("Rich dark fruit\nLight citrus").run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) == 1
    assert at.dataframe[0].value.shape == (2, 3)

    at.text_area[0].input("Rich dark fruit").run()
    at.button[0].click().run()
    assert any("Predicted band" in s.value for s in at.subheader)


def test_tokenizer_pickle_written_by_tf_keras_loads(tmp_path: Path):
    """The notebook's artifact format (pickle, HIGHEST_PROTOCOL) still loads."""
    from tf_keras.preprocessing.text import Tokenizer

    from winemood.config import ArtifactConfig
    from winemood.inference.legacy_lstm import load_tokenizer

    tok = Tokenizer(num_words=20)
    tok.fit_on_texts(["rich dark fruit"])
    with (tmp_path / "tokenizer.pickle").open("wb") as fh:
        pickle.dump(tok, fh, protocol=pickle.HIGHEST_PROTOCOL)
    loaded = load_tokenizer(ArtifactConfig(dir=tmp_path))
    assert loaded.word_index == tok.word_index
