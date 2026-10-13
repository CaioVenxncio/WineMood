"""Text features: safe TF-IDF, structural length features, sequence tokenisation.

``SafeTfidf`` is fitted with scikit-learn's ``TfidfVectorizer`` but persisted as
plain JSON + NumPy arrays (vocabulary, idf weights, parameters) and reloaded
without pickle: a fixed-vocabulary ``CountVectorizer`` plus explicit idf
weighting and L2 normalisation reproduce the fitted transform exactly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

_WORD = re.compile(r"\w+")

# Parameters that fully determine the transform once vocabulary and idf are known.
_TFIDF_PARAMS = (
    "analyzer",
    "ngram_range",
    "min_df",
    "max_df",
    "max_features",
    "sublinear_tf",
    "lowercase",
    "token_pattern",
)


class SafeTfidf:
    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(_TFIDF_PARAMS)
        if unknown:
            raise ValueError(f"Unsupported TF-IDF parameter(s): {sorted(unknown)}")
        self.params: dict[str, Any] = {
            "analyzer": "word",
            "ngram_range": (1, 1),
            "min_df": 2,
            "max_df": 1.0,
            "max_features": None,
            "sublinear_tf": True,
            "lowercase": False,  # text arrives already normalised by TextPreprocessor
            "token_pattern": r"(?u)\b\w\w+\b",
            **params,
        }
        self.params["ngram_range"] = tuple(self.params["ngram_range"])
        self.vocabulary_: dict[str, int] | None = None
        self.idf_: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> SafeTfidf:
        vec = TfidfVectorizer(dtype=np.float32, norm="l2", use_idf=True, **self.params)
        vec.fit(texts)
        self.vocabulary_ = {k: int(v) for k, v in vec.vocabulary_.items()}
        self.idf_ = vec.idf_.astype(np.float32)
        return self

    def _counter(self) -> CountVectorizer:
        if self.vocabulary_ is None:
            raise RuntimeError("SafeTfidf is not fitted")
        p = self.params
        return CountVectorizer(
            analyzer=p["analyzer"],
            ngram_range=p["ngram_range"],
            lowercase=p["lowercase"],
            token_pattern=p["token_pattern"],
            vocabulary=self.vocabulary_,
            dtype=np.float32,
        )

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        counts = self._counter().transform(texts).tocsr().astype(np.float32)
        if self.params["sublinear_tf"]:
            counts.data = np.log(counts.data) + 1.0
        weighted = counts @ sp.diags(self.idf_)
        return normalize(weighted, norm="l2", copy=False).tocsr()

    def fit_transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        return self.fit(texts).transform(texts)

    @property
    def n_features(self) -> int:
        return len(self.vocabulary_ or {})

    def feature_names(self) -> np.ndarray:
        names = np.empty(self.n_features, dtype=object)
        for term, idx in (self.vocabulary_ or {}).items():
            names[idx] = term
        return names

    def save(self, directory: Path, prefix: str = "tfidf") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        params = {**self.params, "ngram_range": list(self.params["ngram_range"])}
        (directory / f"{prefix}.json").write_text(
            json.dumps({"params": params, "vocabulary": self.vocabulary_}), encoding="utf-8"
        )
        np.save(directory / f"{prefix}_idf.npy", self.idf_, allow_pickle=False)

    @classmethod
    def load(cls, directory: Path, prefix: str = "tfidf") -> SafeTfidf:
        payload = json.loads((directory / f"{prefix}.json").read_text(encoding="utf-8"))
        obj = cls(**payload["params"])
        obj.vocabulary_ = {k: int(v) for k, v in payload["vocabulary"].items()}
        obj.idf_ = np.load(directory / f"{prefix}_idf.npy", allow_pickle=False)
        return obj


def length_features(texts: Sequence[str]) -> np.ndarray:
    """Structural features with no learned component: word and character counts."""
    words = np.fromiter((len(_WORD.findall(t)) for t in texts), dtype=float, count=len(texts))
    chars = np.fromiter((len(t) for t in texts), dtype=float, count=len(texts))
    return np.column_stack([words, chars])


def sequence_text(text: str) -> str:
    """Whitespace-joined word tokens for the LSTM's vectoriser (placeholders kept)."""
    return " ".join(_WORD.findall(text.lower()))
