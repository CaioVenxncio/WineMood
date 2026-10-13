"""Figures for the EDA report (matplotlib, Okabe-Ito colour-blind-safe palette)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from winemood.data.labels import WE_CATEGORY_BOUNDS, WE_CATEGORY_V1

# Okabe & Ito (2008) palette.
OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442")
INK = "#222222"
MUTED = "#777777"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e5e5e5",
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _save(fig: plt.Figure, out: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def score_distribution(summary: dict[str, Any], out: Path) -> Path:
    counts = {int(k): v for k, v in summary["points_counts"].items()}
    xs = np.arange(80, 101)
    ys = np.array([counts.get(int(x), 0) for x in xs])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bounds = (*WE_CATEGORY_BOUNDS, 101)
    for i in range(len(WE_CATEGORY_BOUNDS)):
        ax.axvspan(bounds[i] - 0.5, bounds[i + 1] - 0.5, color=OKABE_ITO[i % 7], alpha=0.10, lw=0)
        ax.text(
            (bounds[i] + bounds[i + 1] - 1) / 2,
            ys.max() * 1.04,
            WE_CATEGORY_V1.display_names[i].split(" (")[0],
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    ax.bar(xs, ys, color=OKABE_ITO[0], width=0.8)
    ax.set_xticks(xs)
    ax.set_xlabel("Critic score (points)")
    ax.set_ylabel("Reviews")
    ax.set_ylim(0, ys.max() * 1.12)
    ax.set_title("Score distribution with Wine Enthusiast categories (deduplicated corpus)")
    return _save(fig, out)


def target_distributions(summary: dict[str, Any], out: Path) -> Path:
    names = ("we_category_v1", "we_category_merged_v1", "points_binary_90_v1")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [6, 5, 2]})
    for ax, name in zip(axes, names, strict=True):
        rep = summary["targets"][name]
        labels = list(rep["proportions"])
        vals = [rep["proportions"][k] * 100 for k in labels]
        ax.bar(range(len(vals)), vals, color=OKABE_ITO[2])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8)
        ax.set_xticks(range(len(vals)), [lbl.replace("_", "\n") for lbl in labels], fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18)
    axes[0].set_ylabel("Share of reviews (%)")
    fig.suptitle("Candidate classification targets (deduplicated corpus)", fontweight="bold")
    return _save(fig, out)


def length_distribution(words: np.ndarray, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(words, bins=np.arange(0, words.max() + 5, 3), color=OKABE_ITO[4], edgecolor="white")
    med = float(np.median(words))
    ax.axvline(med, color=OKABE_ITO[5], lw=1.5)
    ax.text(med + 2, ax.get_ylim()[1] * 0.9, f"median = {med:.0f}", color=OKABE_ITO[5])
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("Reviews")
    ax.set_title("Review length distribution (deduplicated corpus)")
    return _save(fig, out)


def length_vs_score(
    points: np.ndarray, words: np.ndarray, rho: dict[str, float], out: Path
) -> Path:
    xs = np.arange(80, 101)
    groups = [words[points == x] for x in xs]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.boxplot(
        [g if g.size else [np.nan] for g in groups],
        positions=xs,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#cfe3f3", "edgecolor": OKABE_ITO[0]},
        medianprops={"color": OKABE_ITO[5]},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
    )
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("Critic score (points)")
    ax.set_ylabel("Review length (words)")
    ax.set_title(
        f"Length vs score, training split (Spearman rho = {rho['rho']:.2f}, "
        f"95% CI {rho['ci95_low']:.2f}-{rho['ci95_high']:.2f}; associational)"
    )
    return _save(fig, out)


def missingness(summary: dict[str, Any], out: Path) -> Path:
    miss = {k: v for k, v in summary["missing_fraction"].items() if k != "review_id"}
    items = sorted(miss.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([k for k, _ in items], [v * 100 for _, v in items], color=OKABE_ITO[1])
    ax.set_xlabel("Missing values (%)")
    ax.grid(axis="x", color="#e5e5e5")
    ax.grid(axis="y", visible=False)
    ax.set_title("Missingness by column (raw file)")
    return _save(fig, out)


def taster_distribution(summary: dict[str, Any], out: Path) -> Path:
    t = summary["metadata"]["tasters"]
    counts = t["rows_per_taster_sorted"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(1, len(counts) + 1), counts, color=OKABE_ITO[3])
    ax.set_xlabel("Taster (anonymised, ranked by number of reviews)")
    ax.set_ylabel("Reviews")
    ax.set_xticks(range(1, len(counts) + 1))
    ax.set_title(f"Reviews per taster ({t['rows_missing_taster']:,} reviews have no taster)")
    return _save(fig, out)


def log_odds_terms(model: dict[str, Any], out: Path, k: int = 15) -> Path:
    lo = model["log_odds_ge90_vs_lt90"]
    terms = lo["favours_b"][:k][::-1] + lo["favours_a"][:k][::-1]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    zs = [z for _, z in terms]
    colours = [OKABE_ITO[0] if z > 0 else OKABE_ITO[1] for z in zs]
    ax.barh(range(len(terms)), zs, color=colours)
    ax.set_yticks(range(len(terms)), [w for w, _ in terms], fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="x", color="#e5e5e5")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("z-scored log-odds (positive: more frequent in >= 90-point reviews)")
    ax.set_title("Terms associated with >= 90 vs < 90 points (training split)")
    return _save(fig, out)


def vocabulary_coverage(model: dict[str, Any], out: Path) -> Path:
    cov = model["token_coverage_by_top_k"]
    xs = [int(k) for k in cov]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(xs, [cov[str(x)] * 100 for x in xs], marker="o", color=OKABE_ITO[2])
    ax.set_xscale("log")
    ax.set_xlabel("Vocabulary size (most frequent training terms)")
    ax.set_ylabel("Training tokens covered (%)")
    ax.set_title("Vocabulary coverage (training split)")
    return _save(fig, out)


def make_all(
    summary: dict[str, Any],
    model: dict[str, Any],
    dedup_words: np.ndarray,
    train_points: np.ndarray,
    train_words: np.ndarray,
    out_dir: Path,
) -> list[Path]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        score_distribution(summary, out_dir / "score_distribution.png"),
        target_distributions(summary, out_dir / "target_distributions.png"),
        length_distribution(dedup_words, out_dir / "length_distribution.png"),
        length_vs_score(
            train_points,
            train_words,
            model["spearman_words_points_train"],
            out_dir / "length_vs_score_train.png",
        ),
        missingness(summary, out_dir / "missingness.png"),
        taster_distribution(summary, out_dir / "taster_distribution.png"),
        log_odds_terms(model, out_dir / "log_odds_terms_train.png"),
        vocabulary_coverage(model, out_dir / "vocabulary_coverage_train.png"),
    ]
