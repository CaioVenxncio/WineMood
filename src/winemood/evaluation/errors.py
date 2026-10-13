"""Error analysis keyed by ``review_id`` (never by array position; audit A-03).

All functions take a predictions table (review_id, y_true, y_pred) and the
deduplicated corpus indexed by review_id, and return plain records suitable for
JSON. Tasters are anonymised by rank of review count.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

_WORD = re.compile(r"\w+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
EXCERPT_CHARS = 90  # keep published review text minimal (copyright, CC BY-NC-SA)


def attach(pred: pd.DataFrame, corpus: pd.DataFrame, id_column: str = "review_id") -> pd.DataFrame:
    rows = corpus.loc[pred[id_column].to_numpy()]
    out = pred.copy()
    out["residual"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["residual"].abs()
    text = rows["description"].astype(str).to_numpy()
    out["words"] = [len(_WORD.findall(t)) for t in text]
    out["has_year"] = [bool(_YEAR.search(t)) for t in text]
    for col in ("taster_name", "country", "price", "variety"):
        out[col] = rows[col].to_numpy()
    return out


def _summary(frame: pd.DataFrame, by: str | pd.Series, min_n: int = 1) -> list[dict[str, Any]]:
    grouped = frame.groupby(by, observed=True, dropna=False)
    table = grouped.agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mean_residual=("residual", "mean"),
        mean_true=("y_true", "mean"),
        mean_pred=("y_pred", "mean"),
    ).reset_index()
    table = table[table["n"] >= min_n]
    first = table.columns[0]
    table[first] = table[first].astype(str)
    return table.rename(columns={first: "bin"}).to_dict(orient="records")


def by_true_score(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _summary(frame, "y_true")


def by_length_decile(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _summary(frame, pd.qcut(frame["words"], 10, duplicates="drop"))


def by_taster(frame: pd.DataFrame, min_n: int = 100) -> list[dict[str, Any]]:
    counts = frame["taster_name"].value_counts()
    rank = {name: f"Taster {i + 1:02d}" for i, name in enumerate(counts.index)}
    anon = frame["taster_name"].map(rank).fillna("(no taster)")
    return _summary(frame.assign(taster=anon), "taster", min_n=min_n)


def by_country(frame: pd.DataFrame, top: int = 8) -> list[dict[str, Any]]:
    keep = frame["country"].value_counts().head(top).index
    country = frame["country"].where(frame["country"].isin(keep), "Other")
    return _summary(frame.assign(c=country.fillna("Other")), "c")


def by_price_band(frame: pd.DataFrame) -> list[dict[str, Any]]:
    bands = pd.cut(
        frame["price"],
        [0, 15, 25, 40, 75, np.inf],
        labels=["<=15", "15-25", "25-40", "40-75", ">75"],
    )
    return _summary(frame.assign(band=bands.astype(str).replace("nan", "(missing)")), "band")


def by_year_mention(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _summary(
        frame.assign(year=frame["has_year"].map({True: "mentions a year", False: "no year"})),
        "year",
    )


def regression_to_mean(frame: pd.DataFrame) -> dict[str, float]:
    """Slope of prediction on truth (< 1 means compression towards the mean)."""
    slope, intercept = np.polyfit(frame["y_true"], frame["y_pred"], 1)
    return {
        "slope_pred_on_true": float(slope),
        "intercept": float(intercept),
        "pred_std": float(frame["y_pred"].std()),
        "true_std": float(frame["y_true"].std()),
    }


def boundary_errors(frame: pd.DataFrame) -> dict[str, float]:
    low = frame["y_true"] <= 83
    high = frame["y_true"] >= 94
    mid = ~(low | high)
    return {
        "mae_low_80_83": float(frame.loc[low, "abs_error"].mean()),
        "mae_mid_84_93": float(frame.loc[mid, "abs_error"].mean()),
        "mae_high_94_100": float(frame.loc[high, "abs_error"].mean()),
        "n_low": int(low.sum()),
        "n_high": int(high.sum()),
    }


def excerpt(text: str, n: int = EXCERPT_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def extreme_examples(frame: pd.DataFrame, corpus: pd.DataFrame, k: int = 5) -> dict[str, list]:
    def rows(sub: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "review_id": int(r.review_id),
                "true": float(r.y_true),
                "pred": round(float(r.y_pred), 2),
                "words": int(r.words),
                "excerpt": excerpt(corpus.at[int(r.review_id), "description"]),
            }
            for r in sub.itertuples()
        ]

    return {
        "overestimated": rows(frame.nlargest(k, "residual")),
        "underestimated": rows(frame.nsmallest(k, "residual")),
        "accurate": rows(frame.loc[frame["abs_error"] < 0.25].head(k)),
    }


def disagreements(
    a: pd.DataFrame, b: pd.DataFrame, corpus: pd.DataFrame, k: int = 5
) -> list[dict[str, Any]]:
    merged = a[["review_id", "y_true", "y_pred"]].merge(
        b[["review_id", "y_pred"]], on="review_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    merged["gap"] = (merged["y_pred_a"] - merged["y_pred_b"]).abs()
    top = merged.nlargest(k, "gap")
    return [
        {
            "review_id": int(r.review_id),
            "true": float(r.y_true),
            "pred_a": round(float(r.y_pred_a), 2),
            "pred_b": round(float(r.y_pred_b), 2),
            "excerpt": excerpt(corpus.at[int(r.review_id), "description"]),
        }
        for r in top.itertuples()
    ] + [{"correlation": float(np.corrcoef(merged["y_pred_a"], merged["y_pred_b"])[0, 1])}]


def label_noise_floor(raw: pd.DataFrame) -> dict[str, Any]:
    """Empirical disagreement in the target itself (descriptive, whole corpus).

    * exact duplicate descriptions carrying different scores;
    * repeat reviews of the same wine (same title, different text).
    """
    key = raw["description"].str.lower().str.split().str.join(" ")
    dup = raw[key.duplicated(keep=False)].assign(k=key)
    spread = dup.groupby("k")["points"].agg(["nunique", "min", "max"])
    conflicting = spread[spread["nunique"] > 1]
    title = raw["title"].str.lower().str.split().str.join(" ")
    distinct = raw.assign(t=title, k=key).drop_duplicates("k")
    pairs = []
    for _, g in distinct.groupby("t"):
        if len(g) > 1:
            pts = g["points"].to_numpy(dtype=float)
            pairs.extend(
                abs(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))
            )
    pairs_arr = np.asarray(pairs)
    return {
        "conflicting_duplicate_groups": len(conflicting),
        "conflicting_duplicate_mean_range": float((conflicting["max"] - conflicting["min"]).mean())
        if len(conflicting)
        else 0.0,
        "same_wine_pairs": int(pairs_arr.size),
        "same_wine_mean_abs_diff": float(pairs_arr.mean()) if pairs_arr.size else None,
        "same_wine_median_abs_diff": float(np.median(pairs_arr)) if pairs_arr.size else None,
    }
