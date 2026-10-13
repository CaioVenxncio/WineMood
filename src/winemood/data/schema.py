"""Validation of the raw wine-review table."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# The WineEnthusiast scale as published in the Kaggle dataset covers 80-100.
EXPECTED_POINTS_RANGE: tuple[int, int] = (80, 100)


class DataValidationError(ValueError):
    """Raised when the dataset violates a hard schema requirement."""


@dataclass
class ValidationReport:
    n_rows: int
    n_missing_text: int = 0
    n_empty_text: int = 0
    n_missing_target: int = 0
    n_duplicate_text: int = 0
    points_min: float | None = None
    points_max: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_reviews(
    df: pd.DataFrame,
    text_column: str = "description",
    target_column: str = "points",
    points_range: tuple[int, int] = EXPECTED_POINTS_RANGE,
) -> ValidationReport:
    """Check required columns, types, missing values, duplicates and score range.

    Missing columns or a non-numeric target are errors. Missing values,
    duplicates and out-of-range scores are reported as warnings so that the
    caller decides how to handle them explicitly.
    """
    report = ValidationReport(n_rows=len(df))

    missing_cols = [c for c in (text_column, target_column) if c not in df.columns]
    if missing_cols:
        report.errors.append(f"Missing required column(s): {missing_cols}")
        return report

    text = df[text_column]
    report.n_missing_text = int(text.isna().sum())
    report.n_empty_text = int((text.dropna().astype(str).str.strip() == "").sum())
    report.n_duplicate_text = int(text.dropna().duplicated().sum())

    target = df[target_column]
    if not pd.api.types.is_numeric_dtype(target):
        report.errors.append(f"Column '{target_column}' must be numeric, got {target.dtype}")
        return report
    report.n_missing_target = int(target.isna().sum())

    if target.notna().any():
        report.points_min = float(target.min())
        report.points_max = float(target.max())
        lo, hi = points_range
        n_out = int(((target < lo) | (target > hi)).sum())
        if n_out:
            report.warnings.append(f"{n_out} row(s) have '{target_column}' outside [{lo}, {hi}]")

    if report.n_missing_text or report.n_empty_text:
        report.warnings.append(
            f"{report.n_missing_text} missing and {report.n_empty_text} empty text value(s)"
        )
    if report.n_missing_target:
        report.warnings.append(f"{report.n_missing_target} missing target value(s)")
    if report.n_duplicate_text:
        report.warnings.append(
            f"{report.n_duplicate_text} exact duplicate text value(s); "
            "deduplicate before splitting to avoid train/test leakage"
        )
    return report


def require_valid(report: ValidationReport) -> None:
    if not report.ok:
        raise DataValidationError("; ".join(report.errors))
