"""Reproducible exploratory analysis.

Two clearly separated parts:

* **Part I, descriptive (whole corpus).** Non-lexical statistics: scores,
  lengths, missingness, duplication, metadata, candidate target distributions.
  These describe the dataset; they are not used to choose model settings.
* **Part II, modelling-relevant (training split only).** Vocabulary, n-grams,
  discriminative terms and length/score association. Anything that could inform
  a modelling decision is computed on the training split. The only exception is
  the validation out-of-vocabulary rate, a standard diagnostic for choosing the
  vocabulary size; the test split is never read here.

All statistics are associational. Nothing here supports causal claims.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from winemood.data.labels import (
    POINTS_BAND_V0,
    POINTS_BINARY_90_V1,
    POINTS_REGRESSION_V1,
    WE_CATEGORY_MERGED_V1,
    WE_CATEGORY_V1,
    assign_labels,
    merge_sparse_top_classes,
    target_report,
)
from winemood.data.schema import validate_reviews

_WORD = re.compile(r"\w+")
_VINTAGE = re.compile(r"\b(19[5-9]\d|20[01]\d)\b")
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def word_counts(texts: pd.Series) -> pd.Series:
    return texts.map(lambda t: len(_WORD.findall(t))).astype(int)


def _describe(values: pd.Series) -> dict[str, float]:
    v = values.dropna().astype(float)
    out = {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min())}
    out |= {f"p{p}": float(np.percentile(v, p)) for p in PERCENTILES}
    out["max"] = float(v.max())
    return out


def _spearman_with_ci(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = 1000, seed: int = 42
) -> dict[str, float]:
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        boots[b] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": rho, "ci95_low": float(lo), "ci95_high": float(hi), "n": len(x)}


# --- Part I ---------------------------------------------------------------------------


def descriptive_summary(
    raw: pd.DataFrame, dedup: pd.DataFrame, *, text_column: str, target_column: str
) -> dict[str, Any]:
    """Whole-corpus descriptive statistics. ``raw`` is as loaded; ``dedup`` has
    exact duplicates removed (distributions are reported on ``dedup`` to avoid
    double counting)."""
    points = dedup[target_column]
    words = word_counts(dedup[text_column])
    chars = dedup[text_column].str.len()

    targets = {
        s.name: asdict(target_report(points, s))
        for s in (POINTS_REGRESSION_V1, WE_CATEGORY_V1, WE_CATEGORY_MERGED_V1, POINTS_BINARY_90_V1)
    }
    targets[POINTS_BAND_V0.name] = asdict(target_report(points, POINTS_BAND_V0))

    we = assign_labels(points, WE_CATEGORY_V1)
    length_by_category = {
        WE_CATEGORY_V1.class_names[int(k)]: float(v) for k, v in words.groupby(we).median().items()
    }

    summary: dict[str, Any] = {
        "n_rows_raw": len(raw),
        "n_rows_dedup": len(dedup),
        "validation": validate_reviews(raw, text_column, target_column).to_dict(),
        "points": _describe(points),
        "points_counts": {
            str(int(k)): int(v) for k, v in points.value_counts().sort_index().items()
        },
        "targets": targets,
        "length_words": _describe(words),
        "length_chars": _describe(chars),
        "median_words_by_we_category": length_by_category,
        "spearman_words_points_all": float(spearmanr(words, points).statistic),
        "missing_fraction": {c: float(v) for c, v in raw.isna().mean().items()},
        "metadata": _metadata_summary(dedup, target_column),
    }
    return summary


def _metadata_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(df)
    if "country" in df:
        top = df["country"].value_counts().head(10)
        out["top_countries"] = {k: int(v) for k, v in top.items()}
        out["n_countries"] = int(df["country"].nunique())
    if "variety" in df:
        top = df["variety"].value_counts().head(10)
        out["top_varieties"] = {k: int(v) for k, v in top.items()}
        out["n_varieties"] = int(df["variety"].nunique())
    if "price" in df:
        price = df["price"]
        both = df[[target_column, "price"]].dropna()
        out["price"] = _describe(price) | {"missing": int(price.isna().sum())}
        out["spearman_price_points"] = float(
            spearmanr(both["price"], both[target_column]).statistic
        )
    if "taster_name" in df:
        # Tasters are real, named critics: report anonymised aggregates only.
        counts = df["taster_name"].value_counts()
        means = df.groupby("taster_name")[target_column].mean()
        out["tasters"] = {
            "n_tasters": int(counts.size),
            "rows_missing_taster": int(df["taster_name"].isna().sum()),
            "rows_per_taster_sorted": [int(v) for v in counts.to_numpy()],
            "largest_taster_share": float(counts.iloc[0] / n) if counts.size else 0.0,
            "mean_points_per_taster_min": float(means.min()) if means.size else None,
            "mean_points_per_taster_max": float(means.max()) if means.size else None,
        }
    if "title" in df:
        vintage = df["title"].str.extract(_VINTAGE, expand=False).astype(float)
        out["vintage"] = _describe(vintage) | {"missing": int(vintage.isna().sum())}
    return out


# --- Part II --------------------------------------------------------------------------


def _top_ngrams(texts: pd.Series, n: int, k: int = 20) -> list[tuple[str, int]]:
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(ENGLISH_STOP_WORDS), min_df=5)
    counts = np.asarray(vec.fit_transform(texts).sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    order = np.argsort(-counts, kind="stable")[:k]
    return [(str(vocab[i]), int(counts[i])) for i in order]


def log_odds_dirichlet(
    texts_a: pd.Series, texts_b: pd.Series, *, min_df: int = 10, k: int = 20
) -> dict[str, list[tuple[str, float]]]:
    """Weighted log-odds ratio with an informative Dirichlet prior
    (Monroe, Colaresi & Quinn, 2008). Positive z favours group ``a``."""
    vec = CountVectorizer(min_df=min_df, stop_words=list(ENGLISH_STOP_WORDS))
    vec.fit(pd.concat([texts_a, texts_b]))
    ya = np.asarray(vec.transform(texts_a).sum(axis=0)).ravel().astype(float)
    yb = np.asarray(vec.transform(texts_b).sum(axis=0)).ravel().astype(float)
    prior = ya + yb
    a0 = prior.sum()
    na, nb = ya.sum(), yb.sum()
    delta = np.log((ya + prior) / (na + a0 - ya - prior)) - np.log(
        (yb + prior) / (nb + a0 - yb - prior)
    )
    z = delta / np.sqrt(1.0 / (ya + prior) + 1.0 / (yb + prior))
    vocab = vec.get_feature_names_out()
    order = np.argsort(z, kind="stable")
    return {
        "favours_a": [(str(vocab[i]), float(z[i])) for i in order[::-1][:k]],
        "favours_b": [(str(vocab[i]), float(z[i])) for i in order[:k]],
    }


def modelling_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    text_column: str,
    target_column: str,
    min_class_fraction: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Training-split analyses that may inform modelling decisions."""
    texts = train[text_column]
    points = train[target_column]
    words = word_counts(texts)

    vec = CountVectorizer(lowercase=True)
    term_counts = np.asarray(vec.fit_transform(texts).sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    order = np.argsort(-term_counts, kind="stable")
    cumulative = np.cumsum(term_counts[order]) / term_counts.sum()
    sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    coverage = {str(s): float(cumulative[min(s, len(cumulative)) - 1]) for s in sizes}

    val_tokens = [w for t in validation[text_column].str.lower() for w in _WORD.findall(t)]
    oov = {}
    for s in sizes:
        kept = set(vocab[order[:s]])
        oov[str(s)] = float(np.mean([w not in kept for w in val_tokens])) if val_tokens else 0.0

    merged = merge_sparse_top_classes(
        assign_labels(points, WE_CATEGORY_V1), WE_CATEGORY_V1, min_class_fraction
    )
    deciles = pd.qcut(words, 10, duplicates="drop")
    by_decile = points.groupby(deciles, observed=True).agg(["mean", "count"])

    return {
        "n_train": len(train),
        "n_validation": len(validation),
        "vocabulary_size": len(vocab),
        "terms_with_count_ge_5": int((term_counts >= 5).sum()),
        "token_coverage_by_top_k": coverage,
        "validation_oov_rate_by_vocab_size": oov,
        "length_words_train": _describe(words),
        "spearman_words_points_train": _spearman_with_ci(
            words.to_numpy(), points.to_numpy(), seed=seed
        ),
        "mean_points_by_length_decile": [
            {"bin": str(k), "mean_points": float(r["mean"]), "n": int(r["count"])}
            for k, r in by_decile.iterrows()
        ],
        "top_unigrams": _top_ngrams(texts, 1),
        "top_bigrams": _top_ngrams(texts, 2),
        "top_trigrams": _top_ngrams(texts, 3),
        "log_odds_ge90_vs_lt90": log_odds_dirichlet(texts[points >= 90], texts[points < 90]),
        "merge_rule": {
            "min_class_fraction": min_class_fraction,
            "derived_scheme": merged.name,
            "classes": list(merged.class_names),
            "matches_frozen_scheme": merged.class_names == WE_CATEGORY_MERGED_V1.class_names,
        },
    }
