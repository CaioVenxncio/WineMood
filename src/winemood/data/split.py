"""Leakage-safe deduplication and train/validation/test splitting.

Splitting happens on raw rows, before any vocabulary, tokenizer or
vectorizer is fitted. Anything fitted later must use ``train`` only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from winemood.logging_utils import get_logger

logger = get_logger(__name__)

_WS = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    return _WS.sub(" ", str(text).strip().lower())


@dataclass
class DedupReport:
    n_before: int
    n_after: int
    n_conflicting_groups: int

    @property
    def n_removed(self) -> int:
        return self.n_before - self.n_after


def deduplicate(
    df: pd.DataFrame, text_column: str, label_column: str | None = None
) -> tuple[pd.DataFrame, DedupReport]:
    """Keep the first occurrence of each normalised text.

    If ``label_column`` is given, groups of duplicate texts carrying different
    labels are counted, because they indicate label noise.
    """
    key = df[text_column].map(normalize_for_dedup)
    conflicts = 0
    if label_column is not None:
        conflicts = int((df.groupby(key)[label_column].nunique() > 1).sum())
    out = df.loc[~key.duplicated()].copy()
    report = DedupReport(len(df), len(out), conflicts)
    logger.info(
        "deduplicated",
        extra={
            "removed": report.n_removed,
            "kept": report.n_after,
            "conflicting_groups": conflicts,
        },
    )
    return out, report


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    dedup: DedupReport | None


def _can_stratify(labels: pd.Series) -> bool:
    return labels.nunique() > 1 and int(labels.value_counts().min()) >= 2


def _split(df: pd.DataFrame, size: float, label_column: str, seed: int):
    stratify = df[label_column] if _can_stratify(df[label_column]) else None
    if stratify is None:
        logger.warning("stratification disabled: a class has fewer than 2 samples")
    return train_test_split(df, test_size=size, random_state=seed, stratify=stratify)


def split_dataset(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
    dedupe: bool = True,
) -> SplitResult:
    """Deduplicate (optionally) then make a stratified train/validation/test split.

    ``validation_size`` is a fraction of the full dataset, not of the
    training remainder.
    """
    dedup_report = None
    if dedupe:
        df, dedup_report = deduplicate(df, text_column, label_column)

    train_val, test = _split(df, test_size, label_column, seed)
    if validation_size > 0:
        relative = validation_size / (1.0 - test_size)
        train, val = _split(train_val, relative, label_column, seed)
    else:
        train, val = train_val, train_val.iloc[0:0]

    result = SplitResult(train, val, test, dedup_report)
    if dedupe:
        assert_no_text_overlap(result, text_column)
    return result


def assert_no_text_overlap(result: SplitResult, text_column: str) -> None:
    """Raise if any normalised text appears in more than one split."""
    sets = {
        name: set(part[text_column].map(normalize_for_dedup))
        for name, part in (
            ("train", result.train),
            ("validation", result.validation),
            ("test", result.test),
        )
    }
    for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise AssertionError(f"{len(overlap)} text(s) shared between {a} and {b}")
