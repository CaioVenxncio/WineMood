"""Validation of the raw wine-review table."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from winemood.data.split import normalize_for_dedup

# Wine Enthusiast's published scale covers 80-100; wines below 80 are not reviewed.
EXPECTED_POINTS_RANGE: tuple[int, int] = (80, 100)
SHORT_TEXT_WORDS = 5

# Heuristic markers of UTF-8 text decoded as Latin-1/CP1252 ("mojibake").
_MOJIBAKE = re.compile(r"\u00c3[\u0080-\u00bf]|\u00e2\u20ac|\u00c2[\u00a0-\u00bf]")
_REPLACEMENT = chr(0xFFFD)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class DataValidationError(ValueError):
    """Raised when the dataset violates a hard schema requirement."""


@dataclass
class ValidationReport:
    n_rows: int
    columns: dict[str, str] = field(default_factory=dict)
    missing_by_column: dict[str, int] = field(default_factory=dict)
    n_missing_text: int = 0
    n_empty_text: int = 0
    n_short_text: int = 0
    n_missing_target: int = 0
    n_non_integer_target: int = 0
    n_out_of_range_target: int = 0
    n_duplicate_text: int = 0
    n_duplicate_text_normalized: int = 0
    n_duplicate_rows: int = 0
    n_duplicate_title: int = 0
    n_texts_with_control_chars: int = 0
    n_texts_with_replacement_char: int = 0
    n_texts_with_suspected_mojibake: int = 0
    n_texts_non_ascii: int = 0
    points_min: float | None = None
    points_max: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def validate_reviews(
    df: pd.DataFrame,
    text_column: str = "description",
    target_column: str = "points",
    points_range: tuple[int, int] = EXPECTED_POINTS_RANGE,
    id_column: str | None = "review_id",
    title_column: str | None = "title",
) -> ValidationReport:
    """Check required columns, types, missing values, duplicates, score range and
    text quality.

    Missing columns or a non-numeric target are errors. Everything else is
    reported as a count plus a warning, so the caller decides explicitly how to
    handle it. Nothing is modified.
    """
    report = ValidationReport(n_rows=len(df))
    report.columns = {str(c): str(t) for c, t in df.dtypes.items()}
    report.missing_by_column = {str(c): int(n) for c, n in df.isna().sum().items()}

    missing_cols = [c for c in (text_column, target_column) if c not in df.columns]
    if missing_cols:
        report.errors.append(f"Missing required column(s): {missing_cols}")
        return report

    text = df[text_column]
    present = text.dropna().astype(str)
    report.n_missing_text = int(text.isna().sum())
    report.n_empty_text = int((present.str.strip() == "").sum())
    report.n_short_text = int((present.str.split().str.len() < SHORT_TEXT_WORDS).sum())
    report.n_duplicate_text = int(present.duplicated().sum())
    report.n_duplicate_text_normalized = int(present.map(normalize_for_dedup).duplicated().sum())
    report.n_texts_with_control_chars = int(
        present.map(lambda t: _CONTROL.search(t) is not None).sum()
    )
    report.n_texts_with_replacement_char = int(present.map(lambda t: _REPLACEMENT in t).sum())
    report.n_texts_with_suspected_mojibake = int(
        present.map(lambda t: _MOJIBAKE.search(t) is not None).sum()
    )
    report.n_texts_non_ascii = int((~present.map(str.isascii)).sum())

    other_cols = [c for c in df.columns if c != id_column]
    report.n_duplicate_rows = int(df[other_cols].duplicated().sum())
    if title_column and title_column in df.columns:
        report.n_duplicate_title = int(df[title_column].dropna().duplicated().sum())

    target = df[target_column]
    if not pd.api.types.is_numeric_dtype(target):
        report.errors.append(f"Column '{target_column}' must be numeric, got {target.dtype}")
        return report
    report.n_missing_target = int(target.isna().sum())
    valid = target.dropna()
    report.n_non_integer_target = int((valid != valid.round()).sum())

    if not valid.empty:
        report.points_min = float(valid.min())
        report.points_max = float(valid.max())
        lo, hi = points_range
        report.n_out_of_range_target = int(((valid < lo) | (valid > hi)).sum())

    _add_warnings(report, target_column, points_range)
    return report


def _add_warnings(report: ValidationReport, target_column: str, rng: tuple[int, int]) -> None:
    w = report.warnings
    if report.n_out_of_range_target:
        w.append(
            f"{report.n_out_of_range_target} row(s) have '{target_column}' outside {list(rng)}"
        )
    if report.n_non_integer_target:
        w.append(f"{report.n_non_integer_target} non-integer '{target_column}' value(s)")
    if report.n_missing_text or report.n_empty_text:
        w.append(f"{report.n_missing_text} missing and {report.n_empty_text} empty text value(s)")
    if report.n_missing_target:
        w.append(f"{report.n_missing_target} missing target value(s)")
    if report.n_duplicate_text or report.n_duplicate_text_normalized:
        w.append(
            f"{report.n_duplicate_text} exact / {report.n_duplicate_text_normalized} normalised "
            "duplicate text value(s); deduplicate before splitting to avoid train/test leakage"
        )
    for count, what in (
        (report.n_texts_with_control_chars, "control characters"),
        (report.n_texts_with_replacement_char, "U+FFFD replacement characters"),
        (report.n_texts_with_suspected_mojibake, "suspected mojibake"),
    ):
        if count:
            w.append(f"{count} text(s) contain {what}")


def require_valid(report: ValidationReport) -> None:
    if not report.ok:
        raise DataValidationError("; ".join(report.errors))
