"""Validated CSV batch prediction (size, encoding, schema and length limits)."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import pandas as pd

from winemood.preprocessing import has_score_prefix

MAX_BYTES = 1_000_000
MAX_ROWS = 1_000
MIN_CHARS = 10
MAX_CHARS = 5_000
TEXT_COLUMNS = ("review", "description", "text", "tasting_note")


class BatchError(ValueError):
    """A user-facing validation problem (message is safe to display)."""


@dataclass(frozen=True)
class BatchInput:
    reviews: list[str]
    column: str
    skipped_empty: int


def validate_single(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise BatchError("Please enter a review.")
    if len(text) < MIN_CHARS:
        raise BatchError(f"The review is too short (minimum {MIN_CHARS} characters).")
    if len(text) > MAX_CHARS:
        raise BatchError(f"The review is too long (maximum {MAX_CHARS:,} characters).")
    return text


def parse_csv(data: bytes) -> BatchInput:
    if len(data) > MAX_BYTES:
        raise BatchError(f"The file is larger than {MAX_BYTES // 1_000_000} MB.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise BatchError("The file is not valid UTF-8 text. Please save it as UTF-8 CSV.") from None
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error):
        raise BatchError("The file could not be read as CSV.") from None
    columns = {c.strip().lower(): c for c in frame.columns}
    column = next((columns[c] for c in TEXT_COLUMNS if c in columns), None)
    if column is None:
        raise BatchError(
            f"No review column found. Name one column {', '.join(repr(c) for c in TEXT_COLUMNS)}."
        )
    values = frame[column].astype(str).str.strip()
    non_empty = values[values != ""]
    if non_empty.empty:
        raise BatchError("The review column is empty.")
    if len(non_empty) > MAX_ROWS:
        raise BatchError(f"Too many reviews ({len(non_empty):,}); the limit is {MAX_ROWS:,}.")
    too_long = int((non_empty.str.len() > MAX_CHARS).sum())
    if too_long:
        raise BatchError(f"{too_long} review(s) exceed {MAX_CHARS:,} characters.")
    return BatchInput(non_empty.tolist(), str(column), int(len(values) - len(non_empty)))


def predict_batch(predictor, batch: BatchInput) -> pd.DataFrame:
    raw = predictor.predict_raw(batch.reviews)
    preds = predictor.predict(batch.reviews)
    return pd.DataFrame(
        {
            "review": batch.reviews,
            "predicted_critic_score": [round(p.score, 1) for p in preds],
            "wine_enthusiast_category": [p.category for p in preds],
            "raw_output_outside_80_100": (raw < 80) | (raw > 100),
            "starts_with_score_range": [has_score_prefix(t) for t in batch.reviews],
        }
    )


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    header = "# Model estimates of critic scores (WineMood E04 TF-IDF + Ridge); not wine quality.\n"
    return (header + frame.to_csv(index=False)).encode("utf-8")
