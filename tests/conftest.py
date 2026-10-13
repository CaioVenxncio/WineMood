"""Synthetic fixtures: tests never download the real dataset."""

from __future__ import annotations

import pandas as pd
import pytest

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
