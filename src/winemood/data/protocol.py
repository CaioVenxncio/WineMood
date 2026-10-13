"""Research split protocol: dedup -> near-dup and wine-identity groups -> grouped,
stratified train/validation/test assignment, with fingerprints for reproducibility.

See docs/experimental-protocol.md §2 for the rationale and tradeoffs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from winemood.data.neardup import UnionFind, find_near_duplicates
from winemood.data.split import normalize_for_dedup
from winemood.logging_utils import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "validation", "test")


class SplitLeakageError(AssertionError):
    """Raised when an isolation invariant between splits is violated."""


# --- Deduplication ---------------------------------------------------------------


@dataclass
class ExactDedupStats:
    n_before: int
    n_after: int
    n_duplicate_groups: int
    n_groups_with_conflicting_scores: int

    @property
    def n_removed(self) -> int:
        return self.n_before - self.n_after


def drop_exact_duplicates(
    df: pd.DataFrame, text_column: str, target_column: str, id_column: str
) -> tuple[pd.DataFrame, ExactDedupStats]:
    """Keep the lowest-id row of each normalised text (deterministic).

    Duplicate groups whose scores disagree are counted: they are label noise
    that a model cannot resolve, and their count is reported, not hidden.
    """
    ordered = df.sort_values(id_column, kind="stable")
    key = ordered[text_column].map(normalize_for_dedup)
    sizes = key.value_counts()
    conflicts = int((ordered.groupby(key)[target_column].nunique() > 1).sum())
    kept = ordered.loc[~key.duplicated()].copy()
    stats = ExactDedupStats(len(df), len(kept), int((sizes > 1).sum()), conflicts)
    logger.info("exact deduplication", extra=asdict(stats))
    return kept, stats


# --- Grouping --------------------------------------------------------------------


@dataclass
class GroupStats:
    n_rows: int
    n_groups: int
    n_multi_row_groups: int
    largest_group: int
    n_near_dup_pairs: int
    n_rows_in_near_dup_clusters: int
    n_rows_sharing_title: int


def build_groups(
    df: pd.DataFrame,
    text_column: str,
    *,
    title_column: str | None = "title",
    near_dup_cluster: np.ndarray | None = None,
) -> np.ndarray:
    """Union-find over rows linked by equal normalised text, equal normalised
    title (same wine and vintage) or a shared near-duplicate cluster.

    Returns a group id per row (positional, aligned with ``df``).
    """
    n = len(df)
    uf = UnionFind(n)

    def link(keys: Iterable[Any]) -> None:
        first: dict[Any, int] = {}
        for i, k in enumerate(keys):
            if k is None or (isinstance(k, float) and np.isnan(k)):
                continue
            j = first.setdefault(k, i)
            if j != i:
                uf.union(i, j)

    link(df[text_column].map(normalize_for_dedup))
    if title_column and title_column in df.columns:
        link(df[title_column].map(lambda t: normalize_for_dedup(t) if isinstance(t, str) else None))
    if near_dup_cluster is not None:
        if len(near_dup_cluster) != n:
            raise ValueError("near_dup_cluster must align with df")
        link(near_dup_cluster.tolist())
    return uf.labels()


# --- Assignment ------------------------------------------------------------------


def grouped_stratified_assignment(
    groups: np.ndarray,
    strata: np.ndarray,
    *,
    n_folds: int = 10,
    test_folds: int = 2,
    validation_folds: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Assign each row to train/validation/test with whole groups kept together.

    ``StratifiedGroupKFold`` balances the strata across folds as far as group
    constraints allow; the first ``test_folds`` folds form the test set and the
    next ``validation_folds`` the validation set.
    """
    if test_folds + validation_folds >= n_folds:
        raise ValueError("test_folds + validation_folds must be < n_folds")
    fold = np.full(len(groups), -1, dtype=np.int64)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(cv.split(np.zeros(len(groups)), strata, groups)):
        fold[test_idx] = k
    assignment = np.where(
        fold < test_folds,
        "test",
        np.where(fold < test_folds + validation_folds, "validation", "train"),
    )
    return assignment.astype(object)


def taster_holdout_assignment(
    tasters: pd.Series,
    groups: np.ndarray,
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Hold out whole tasters: validation/test tasters never appear in train.

    Rows without a taster are left unassigned (``"excluded"``). Because one wine
    can be reviewed by several tasters, a group may straddle partitions; such
    groups are excluded entirely to keep the leakage invariants. Partition sizes
    are approximate because tasters contribute very different numbers of rows.
    """
    known = tasters.notna().to_numpy()
    names = sorted(tasters[known].unique())
    order = np.random.default_rng(seed).permutation(len(names))
    counts = tasters[known].value_counts()
    total = int(known.sum())
    part_of: dict[str, str] = {}
    acc = {"test": 0, "validation": 0}
    for idx in order:
        name = names[idx]
        if acc["test"] < test_fraction * total:
            target = "test"
        elif acc["validation"] < validation_fraction * total:
            target = "validation"
        else:
            target = "train"
        part_of[name] = target
        if target in acc:
            acc[target] += int(counts[name])
    assignment = np.array(
        [part_of[t] if isinstance(t, str) else "excluded" for t in tasters], dtype=object
    )
    frame = pd.DataFrame({"g": groups, "s": assignment})
    straddling = frame[frame.s != "excluded"].groupby("g").s.nunique()
    bad_groups = set(straddling[straddling > 1].index)
    assignment[np.isin(groups, list(bad_groups))] = "excluded"
    return assignment


def size_matched_random_assignment(
    groups: np.ndarray, reference: np.ndarray, *, seed: int = 42
) -> np.ndarray:
    """Randomly reassign exactly the rows assigned in ``reference`` (whole groups),
    matching each partition's size to within one group.

    This is the comparison split for H3: same rows, same sizes, but test rows are
    drawn at random instead of by taster. Rows excluded in ``reference`` stay
    excluded.
    """
    eligible = reference != "excluded"
    targets = {p: int((reference == p).sum()) for p in ("test", "validation")}
    unique_groups = np.unique(groups[eligible])
    order = np.random.default_rng(seed).permutation(unique_groups)
    sizes = pd.Series(groups[eligible]).value_counts()
    part_of: dict[int, str] = {}
    filled = {"test": 0, "validation": 0}
    for g in order:
        if filled["test"] < targets["test"]:
            target = "test"
        elif filled["validation"] < targets["validation"]:
            target = "validation"
        else:
            target = "train"
        part_of[int(g)] = target
        if target in filled:
            filled[target] += int(sizes[g])
    out = np.full(len(groups), "excluded", dtype=object)
    out[eligible] = [part_of[int(g)] for g in groups[eligible]]
    return out


# --- Fingerprints and invariants -------------------------------------------------


def fingerprint_ids(ids: Iterable[int]) -> str:
    payload = "\n".join(str(int(i)) for i in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def assert_split_isolation(
    df: pd.DataFrame,
    assignment: np.ndarray,
    *,
    id_column: str,
    text_column: str,
    groups: np.ndarray,
) -> None:
    """Raise ``SplitLeakageError`` if ids, normalised texts or groups cross splits."""
    frame = pd.DataFrame(
        {
            "id": df[id_column].to_numpy(),
            "text": df[text_column].map(normalize_for_dedup).to_numpy(),
            "group": groups,
            "split": assignment,
        }
    )
    frame = frame[frame.split.isin(SPLITS)]
    if frame.id.duplicated().any():
        raise SplitLeakageError("duplicate review ids in the assigned rows")
    for column in ("text", "group"):
        crossing = frame.groupby(column).split.nunique()
        n_bad = int((crossing > 1).sum())
        if n_bad:
            raise SplitLeakageError(f"{n_bad} {column} value(s) appear in more than one split")


@dataclass
class SplitManifest:
    name: str
    dataset_sha256: str
    config: dict[str, Any]
    config_hash: str
    counts: dict[str, int]
    id_sha256: dict[str, str]
    stratum_proportions: dict[str, dict[str, float]] = field(default_factory=dict)
    dedup: dict[str, int] = field(default_factory=dict)
    groups: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_manifest(
    name: str,
    df: pd.DataFrame,
    assignment: np.ndarray,
    *,
    id_column: str,
    dataset_sha256: str,
    config: dict[str, Any],
    strata: np.ndarray | None = None,
    strata_names: tuple[str, ...] | None = None,
    dedup: ExactDedupStats | None = None,
    group_stats: GroupStats | None = None,
) -> SplitManifest:
    ids = df[id_column].to_numpy()
    parts = [p for p in (*SPLITS, "excluded") if (assignment == p).any()]
    proportions: dict[str, dict[str, float]] = {}
    if strata is not None:
        for p in parts:
            values = pd.Series(strata[assignment == p]).value_counts(normalize=True)
            proportions[p] = {
                (strata_names[int(k)] if strata_names else str(k)): round(float(v), 6)
                for k, v in sorted(values.items())
            }
    dedup_dict = {**asdict(dedup), "n_removed": dedup.n_removed} if dedup is not None else {}
    return SplitManifest(
        name=name,
        dataset_sha256=dataset_sha256,
        config=config,
        config_hash=config_hash(config),
        counts={p: int((assignment == p).sum()) for p in parts},
        id_sha256={p: fingerprint_ids(ids[assignment == p]) for p in parts},
        stratum_proportions=proportions,
        dedup=dedup_dict,
        groups=asdict(group_stats) if group_stats else {},
    )


def write_manifest_ids(
    out_dir: Path,
    df: pd.DataFrame,
    assignment: np.ndarray,
    id_column: str,
    groups: np.ndarray | None = None,
) -> None:
    """Write one sorted review-id file per split (git-ignored location).

    With ``groups``, also write ``groups.csv`` mapping each assigned review id to
    a stable group id (the smallest review id in its group), which the cluster
    bootstrap uses as its resampling unit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = df[id_column].to_numpy()
    for p in (*SPLITS, "excluded"):
        selected = np.sort(ids[assignment == p])
        if p == "excluded" and selected.size == 0:
            continue
        (out_dir / f"{p}.txt").write_text(
            "\n".join(str(int(i)) for i in selected) + "\n", encoding="utf-8"
        )
    if groups is not None:
        stable = pd.Series(ids).groupby(groups).transform("min").to_numpy()
        table = pd.DataFrame({"review_id": ids, "group": stable}).sort_values("review_id")
        table.to_csv(out_dir / "groups.csv", index=False)


def load_split_ids(split_dir: Path, split: str) -> np.ndarray:
    path = split_dir / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}. Run 'winemood data split'.")
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def load_groups(split_dir: Path) -> pd.Series:
    """Stable group id per review id (written by ``write_manifest_ids``)."""
    path = split_dir / "groups.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Group file not found: {path}. Re-run 'winemood data split'.")
    table = pd.read_csv(path)
    return pd.Series(table["group"].to_numpy(), index=table["review_id"].to_numpy(), name="group")


# --- Orchestration -----------------------------------------------------------------


@dataclass
class ProtocolResult:
    data: pd.DataFrame  # deduplicated rows, with 'group' and 'split' columns
    manifest: SplitManifest
    near_dup_pairs: list[tuple[int, int, float]]


def run_split_protocol(
    df: pd.DataFrame,
    *,
    name: str,
    text_column: str,
    target_column: str,
    id_column: str,
    strata: np.ndarray | None,
    strata_names: tuple[str, ...] | None,
    dataset_sha256: str,
    config: dict[str, Any],
    seed: int,
    neardup: dict[str, Any] | None,
    group_by_title: bool = True,
    n_folds: int = 10,
    test_folds: int = 2,
    validation_folds: int = 1,
    strategy: str = "grouped",
    taster_column: str = "taster_name",
) -> ProtocolResult:
    """Full protocol. ``strata`` must align with ``df`` (before deduplication)."""
    frame = df.copy()
    if strata is not None:
        frame["_stratum"] = strata
    frame, dedup = drop_exact_duplicates(frame, text_column, target_column, id_column)
    frame = frame.reset_index(drop=True)

    cluster = None
    pairs: list[tuple[int, int, float]] = []
    if neardup is not None:
        result = find_near_duplicates(frame[text_column].tolist(), seed=seed, **neardup)
        cluster, pairs = result.cluster, result.pairs
    title_column = "title" if group_by_title else None
    groups = build_groups(frame, text_column, title_column=title_column, near_dup_cluster=cluster)

    _, group_sizes = np.unique(groups, return_counts=True)
    title_shared = 0
    if title_column and title_column in frame.columns:
        title_shared = int(
            frame[title_column].map(normalize_for_dedup).duplicated(keep=False).sum()
        )
    group_stats = GroupStats(
        n_rows=len(frame),
        n_groups=len(group_sizes),
        n_multi_row_groups=int((group_sizes > 1).sum()),
        largest_group=int(group_sizes.max()) if len(group_sizes) else 0,
        n_near_dup_pairs=len(pairs),
        n_rows_in_near_dup_clusters=(
            int(pd.Series(cluster).duplicated(keep=False).sum()) if cluster is not None else 0
        ),
        n_rows_sharing_title=title_shared,
    )

    frame_strata = frame["_stratum"].to_numpy() if strata is not None else None
    if strategy == "grouped":
        if frame_strata is None:
            raise ValueError("grouped strategy requires strata")
        assignment = grouped_stratified_assignment(
            groups,
            frame_strata,
            n_folds=n_folds,
            test_folds=test_folds,
            validation_folds=validation_folds,
            seed=seed,
        )
    elif strategy == "taster_holdout":
        assignment = taster_holdout_assignment(frame[taster_column], groups, seed=seed)
    elif strategy == "taster_random":
        reference = taster_holdout_assignment(frame[taster_column], groups, seed=seed)
        assignment = size_matched_random_assignment(groups, reference, seed=seed)
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'")

    assert_split_isolation(
        frame, assignment, id_column=id_column, text_column=text_column, groups=groups
    )
    manifest = make_manifest(
        name,
        frame,
        assignment,
        id_column=id_column,
        dataset_sha256=dataset_sha256,
        config=config,
        strata=frame_strata,
        strata_names=strata_names,
        dedup=dedup,
        group_stats=group_stats,
    )
    frame = frame.drop(columns=["_stratum"], errors="ignore")
    frame["group"] = groups
    frame["split"] = assignment
    logger.info("split protocol complete", extra={"split_name": name, "counts": manifest.counts})
    return ProtocolResult(frame, manifest, pairs)
