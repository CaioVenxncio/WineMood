"""Loading the raw Kaggle wine-review CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from winemood.logging_utils import get_logger

logger = get_logger(__name__)

ACQUISITION_HINT = (
    "Download 'winemag-data-130k-v2.csv' from "
    "https://www.kaggle.com/datasets/zynicide/wine-reviews and place it at the "
    "configured path (default: data/raw/). See data/README.md."
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw dataset file is not present locally."""


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV, dropping the anonymous index column if present."""
    path = Path(path)
    if not path.is_file():
        raise DatasetNotFoundError(f"Dataset not found at {path}. {ACQUISITION_HINT}")
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    logger.info("loaded dataset", extra={"path": str(path), "rows": len(df)})
    return df
