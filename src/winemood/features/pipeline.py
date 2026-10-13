"""Configurable feature pipeline: text normalisation -> feature blocks -> matrix.

``fit`` sees the training frame only; ``transform`` applies the frozen state to
any frame. State is persisted as JSON/NumPy (no pickle).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from winemood.features.metadata import MetadataEncoder
from winemood.features.text import SafeTfidf, length_features
from winemood.preprocessing import TextPreprocessor, mask_numbers, preprocessing_config

FEATURE_KINDS = ("none", "length", "metadata", "tfidf", "encoder")


@dataclass(frozen=True)
class FeatureSpec:
    kind: str = "tfidf"
    preprocessing: str = "norm_min_v1"
    mask_numbers: str = "none"
    word: dict[str, Any] | None = field(
        default_factory=lambda: {"ngram_range": [1, 2], "min_df": 2}
    )
    char: dict[str, Any] | None = None
    add_length: bool = False
    encoder_model: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in FEATURE_KINDS:
            raise ValueError(f"Unknown feature kind '{self.kind}'")
        if self.kind == "tfidf" and not (self.word or self.char):
            raise ValueError("tfidf features need a word and/or char block")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeaturePipeline:
    def __init__(self, spec: FeatureSpec, *, text_column: str = "description", stopwords=None):
        self.spec = spec
        self.text_column = text_column
        self.preprocessor = TextPreprocessor(
            preprocessing_config(spec.preprocessing, spec.mask_numbers), stopwords=stopwords
        )
        self.word: SafeTfidf | None = None
        self.char: SafeTfidf | None = None
        self.metadata: MetadataEncoder | None = None
        self.length_mean_: np.ndarray | None = None
        self.length_std_: np.ndarray | None = None

    # -- text ------------------------------------------------------------------
    def texts(self, df: pd.DataFrame) -> list[str]:
        return self.preprocessor.batch(df[self.text_column].astype(str).tolist())

    def raw_masked_texts(self, df: pd.DataFrame) -> list[str]:
        """Encoder input: original casing/punctuation, only number masking applied."""
        return [mask_numbers(t, self.spec.mask_numbers) for t in df[self.text_column].astype(str)]

    # -- fit / transform -------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> FeaturePipeline:
        kind = self.spec.kind
        if kind == "length" or (kind == "tfidf" and self.spec.add_length):
            lengths = length_features(train[self.text_column].astype(str).tolist())
            self.length_mean_ = lengths.mean(axis=0)
            self.length_std_ = lengths.std(axis=0) + 1e-9
        if kind == "metadata":
            self.metadata = MetadataEncoder().fit(train)
        if kind == "tfidf":
            texts = self.texts(train)
            if self.spec.word:
                self.word = SafeTfidf(analyzer="word", **self.spec.word).fit(texts)
            if self.spec.char:
                self.char = SafeTfidf(analyzer="char_wb", **self.spec.char).fit(texts)
        return self

    def _lengths(self, df: pd.DataFrame) -> np.ndarray:
        raw = length_features(df[self.text_column].astype(str).tolist())
        return (raw - self.length_mean_) / self.length_std_

    def transform(self, df: pd.DataFrame, embeddings: np.ndarray | None = None):
        kind = self.spec.kind
        if kind == "none":
            return np.zeros((len(df), 0))
        if kind == "length":
            return self._lengths(df)
        if kind == "metadata":
            return self.metadata.transform(df)
        if kind == "encoder":
            if embeddings is None:
                raise ValueError("encoder features require precomputed embeddings")
            return np.asarray(embeddings, dtype=np.float32)
        texts = self.texts(df)
        blocks = [b.transform(texts) for b in (self.word, self.char) if b is not None]
        if self.spec.add_length:
            blocks.append(sp.csr_matrix(self._lengths(df).astype(np.float32)))
        return sp.hstack(blocks, format="csr")

    def feature_names(self) -> np.ndarray:
        names: list[np.ndarray] = []
        if self.word is not None:
            names.append(self.word.feature_names())
        if self.char is not None:
            names.append(np.array([f"char:{n}" for n in self.char.feature_names()], dtype=object))
        if self.spec.kind == "tfidf" and self.spec.add_length:
            names.append(np.array(["__words__", "__chars__"], dtype=object))
        return np.concatenate(names) if names else np.array([], dtype=object)

    # -- persistence -----------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {"spec": self.spec.to_dict(), "text_column": self.text_column}
        if self.length_mean_ is not None:
            state["length_mean"] = self.length_mean_.tolist()
            state["length_std"] = self.length_std_.tolist()
        (directory / "features.json").write_text(json.dumps(state), encoding="utf-8")
        if self.word is not None:
            self.word.save(directory, "tfidf_word")
        if self.char is not None:
            self.char.save(directory, "tfidf_char")
        if self.metadata is not None:
            self.metadata.save(directory / "metadata.json")

    @classmethod
    def load(cls, directory: Path, stopwords=None) -> FeaturePipeline:
        state = json.loads((directory / "features.json").read_text(encoding="utf-8"))
        obj = cls(
            FeatureSpec(**state["spec"]), text_column=state["text_column"], stopwords=stopwords
        )
        if "length_mean" in state:
            obj.length_mean_ = np.asarray(state["length_mean"])
            obj.length_std_ = np.asarray(state["length_std"])
        if (directory / "tfidf_word.json").is_file():
            obj.word = SafeTfidf.load(directory, "tfidf_word")
        if (directory / "tfidf_char.json").is_file():
            obj.char = SafeTfidf.load(directory, "tfidf_char")
        if (directory / "metadata.json").is_file():
            obj.metadata = MetadataEncoder.load(directory / "metadata.json")
        return obj
