"""Split-aware data access for experiments.

Every load verifies the dataset checksum and that the split manifests on disk
still match the committed fingerprints. The test partition is only returned
when the caller explicitly asks for it (the final-evaluation command).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from winemood.config import Config
from winemood.data.acquire import verify_checksum
from winemood.data.ingest import load_reviews
from winemood.data.protocol import drop_exact_duplicates, fingerprint_ids, load_groups, load_split_ids


class SplitIntegrityError(RuntimeError):
    """Manifests on disk do not match the committed fingerprints."""


class TestAccessError(PermissionError):
    """Raised when test data is requested outside the final-evaluation path."""


@dataclass
class SplitData:
    name: str
    frames: dict[str, pd.DataFrame]
    groups: pd.Series  # group id per review_id
    dataset_sha256: str
    id_sha256: dict[str, str]


_CACHE: dict[str, pd.DataFrame] = {}


def deduplicated_corpus(cfg: Config) -> pd.DataFrame:
    key = str(cfg.data.raw_csv)
    if key not in _CACHE:
        verify_checksum(cfg.data.raw_csv, cfg.data.sha256 or None)
        raw = load_reviews(cfg.data.raw_csv, cfg.data.id_column)
        dedup, _ = drop_exact_duplicates(
            raw, cfg.data.text_column, cfg.data.target_column, cfg.data.id_column
        )
        _CACHE[key] = dedup.set_index(cfg.data.id_column, drop=False)
    return _CACHE[key]


def committed_fingerprints(cfg: Config) -> dict:
    if not cfg.data.fingerprints.is_file():
        raise SplitIntegrityError(f"No committed fingerprints at {cfg.data.fingerprints}")
    return json.loads(cfg.data.fingerprints.read_text(encoding="utf-8"))


def load_split(
    cfg: Config, name: str, parts: tuple[str, ...] = ("train", "validation"), *, allow_test: bool = False
) -> SplitData:
    if "test" in parts and not allow_test:
        raise TestAccessError(
            "The test partition is only available to 'winemood exp final' (frozen selection)."
        )
    manifest = committed_fingerprints(cfg).get(name)
    if manifest is None:
        raise SplitIntegrityError(f"Split '{name}' has no committed fingerprint")
    if manifest["dataset_sha256"] != cfg.data.sha256:
        raise SplitIntegrityError(f"Split '{name}' was built from a different dataset")
    split_dir: Path = cfg.data.processed_dir / "splits" / name
    corpus = deduplicated_corpus(cfg)
    frames: dict[str, pd.DataFrame] = {}
    for part in parts:
        ids = load_split_ids(split_dir, part)
        if fingerprint_ids(ids) != manifest["id_sha256"][part]:
            raise SplitIntegrityError(
                f"Manifest {split_dir / part}.txt does not match the committed fingerprint"
            )
        missing = set(ids) - set(corpus.index)
        if missing:
            raise SplitIntegrityError(f"{len(missing)} manifest ids are not in the corpus")
        frames[part] = corpus.loc[ids].reset_index(drop=True)
    return SplitData(
        name=name,
        frames=frames,
        groups=load_groups(split_dir),
        dataset_sha256=manifest["dataset_sha256"],
        id_sha256={p: manifest["id_sha256"][p] for p in parts},
    )
