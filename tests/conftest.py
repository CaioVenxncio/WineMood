"""Synthetic fixtures: tests never download the real dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_VOCAB = [
    "cherry",
    "plum",
    "oak",
    "vanilla",
    "tannins",
    "acidity",
    "finish",
    "palate",
    "aromas",
    "spice",
    "pepper",
    "berry",
    "citrus",
    "apple",
    "pear",
    "mineral",
    "smoke",
    "leather",
    "tobacco",
    "cedar",
    "herbal",
    "floral",
    "honey",
    "toast",
    "structure",
    "balance",
    "concentrated",
    "rich",
    "simple",
    "light",
    "crisp",
    "fresh",
    "ripe",
    "dense",
    "long",
]
_TASTERS = ["Taster A", "Taster B", "Taster C", "Taster D", "Taster E", None]


def make_corpus(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Random reviews shaped like the Kaggle CSV, with planted structure:

    * rows n..n+19 are exact duplicates (case/whitespace variants) of rows 0..19;
    * rows n+20..n+29 are near-duplicates (one word changed) of rows 20..29;
    * rows 30..39 share their ``title`` with rows 40..49 (same wine, new text).
    """
    rng = np.random.default_rng(seed)
    points = rng.choice(np.arange(80, 101), size=n, p=_score_probs())
    rows = []
    for i in range(n):
        length = int(20 + (points[i] - 80) * 1.5 + rng.integers(0, 15))
        text = " ".join(rng.choice(_VOCAB, size=length)) + f" note{i}."
        rows.append(
            {
                "description": text.capitalize(),
                "points": int(points[i]),
                "title": f"Winery {i} 20{10 + i % 10} Red",
                "taster_name": _TASTERS[i % len(_TASTERS)],
                "country": ["US", "France", "Italy"][i % 3],
                "price": float(rng.integers(10, 200)) if i % 7 else np.nan,
                "variety": ["Pinot Noir", "Chardonnay"][i % 2],
            }
        )
    for j in range(30, 40):
        rows[j]["title"] = rows[j + 10]["title"]
    for j in range(20):
        dup = dict(rows[j])
        dup["description"] = "  " + rows[j]["description"].upper() + " "
        rows.append(dup)
    for j in range(20, 30):
        near = dict(rows[j])
        words = rows[j]["description"].split()
        # Replacing the final word changes one shingle: Jaccard = (m-1)/(m+1) >= 0.89.
        words[-1] = "zzzword."
        near["description"] = " ".join(words)
        near["title"] = f"Other {j}"
        rows.append(near)
    df = pd.DataFrame(rows)
    df.insert(0, "review_id", np.arange(len(df)))
    return df


def _score_probs() -> np.ndarray:
    w = np.exp(-0.5 * ((np.arange(80, 101) - 88) / 3.0) ** 2)
    return w / w.sum()


@pytest.fixture
def corpus() -> pd.DataFrame:
    return make_corpus()


@pytest.fixture
def corpus_csv(tmp_path: Path, corpus: pd.DataFrame) -> Path:
    """Write the corpus the way Kaggle ships it: anonymous index column first."""
    path = tmp_path / "raw" / "winemag-data-130k-v2.csv"
    path.parent.mkdir(parents=True)
    corpus.rename(columns={"review_id": ""}).to_csv(path, index=False)
    return path


# A tiny, fixed stopword list keeps preprocessing tests independent of NLTK data.
SMALL_STOPWORDS = frozenset({"a", "an", "and", "the", "with", "is", "of", "no", "not", "nor"})


@pytest.fixture
def stopwords() -> frozenset[str]:
    return SMALL_STOPWORDS


@pytest.fixture
def reviews_df() -> pd.DataFrame:
    """60 unique reviews (30 per populated band) plus 4 duplicates of earlier rows."""
    rows = []
    for i in range(30):
        rows.append({"description": f"Rich layered dark fruit note {i}", "points": 90 + i % 10})
        rows.append({"description": f"Simple light citrus note {i}", "points": 80 + i % 10})
    rows += [
        {"description": "Rich layered dark fruit note 0", "points": 90},
        {"description": "  rich LAYERED dark fruit   note 1 ", "points": 91},
        {"description": "Simple light citrus note 2", "points": 82},
        {"description": "Simple light citrus note 3", "points": 95},  # conflicting label
    ]
    return pd.DataFrame(rows)
