"""Loading the raw Kaggle wine-review CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from winemood.logging_utils import get_logger

logger = get_logger(__name__)

ACQUISITION_HINT = (
    "Run 'winemood data download', or download 'winemag-data-130k-v2.csv' from "
    "https://www.kaggle.com/datasets/zynicide/wine-reviews and place it at the "
    "configured path (default: data/raw/). See data/README.md."
)

ID_COLUMN = "review_id"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw dataset file is not present locally."""


def load_reviews(path: str | Path, id_column: str = ID_COLUMN) -> pd.DataFrame:
    """Read the raw CSV and expose its anonymous index column as ``id_column``.

    The Kaggle CSV's first (unnamed) column is a unique row number. It is kept as
    a stable identifier so that splits, manifests and error analysis can refer to
    rows unambiguously. Files without it get a positional id, with a warning.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetNotFoundError(f"Dataset not found at {path}. {ACQUISITION_HINT}")
    df = pd.read_csv(path, encoding="utf-8")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if id_column not in df.columns:
        if unnamed and df[unnamed[0]].is_unique:
            df = df.rename(columns={unnamed[0]: id_column})
            unnamed = unnamed[1:]
        else:
            logger.warning("no unique index column; assigning positional review ids")
            df.insert(0, id_column, range(len(df)))
    if unnamed:
        df = df.drop(columns=unnamed)
    if not df[id_column].is_unique:
        raise ValueError(f"Column '{id_column}' is not unique")
    df[id_column] = df[id_column].astype("int64")
    logger.info("loaded dataset", extra={"path": str(path), "rows": len(df)})
    return df
