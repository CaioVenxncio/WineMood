"""Metadata features for the B2 reference baseline (not text).

Categories and the price imputation value are learned from the training split
only and stored as JSON; unseen or missing categories map to an "other" column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TOP_K = {"taster_name": 100, "country": 20, "variety": 50}


class MetadataEncoder:
    def __init__(self, top_k: dict[str, int] | None = None) -> None:
        self.top_k = dict(top_k or DEFAULT_TOP_K)
        self.categories_: dict[str, list[str]] | None = None
        self.price_median_: float | None = None

    def fit(self, df: pd.DataFrame) -> MetadataEncoder:
        self.categories_ = {
            col: [str(v) for v in df[col].dropna().value_counts().head(k).index]
            for col, k in self.top_k.items()
        }
        self.price_median_ = float(np.log1p(df["price"]).median())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.categories_ is None or self.price_median_ is None:
            raise RuntimeError("MetadataEncoder is not fitted")
        blocks = []
        for col, cats in self.categories_.items():
            values = df[col].astype("object").where(df[col].notna(), None).to_numpy()
            index = {c: i for i, c in enumerate(cats)}
            block = np.zeros((len(df), len(cats) + 1))
            for row, v in enumerate(values):
                block[row, index.get(str(v), len(cats)) if v is not None else len(cats)] = 1.0
            blocks.append(block)
        log_price = np.log1p(df["price"].to_numpy(dtype=float))
        missing = np.isnan(log_price)
        blocks.append(np.column_stack([np.where(missing, self.price_median_, log_price), missing]))
        return np.hstack(blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "categories": self.categories_,
            "price_median": self.price_median_,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MetadataEncoder:
        payload = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(payload["top_k"])
        obj.categories_ = payload["categories"]
        obj.price_median_ = payload["price_median"]
        return obj
