"""Typed configuration loaded from TOML files."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    """Locate the project root: $WINEMOOD_ROOT, the source checkout, or the CWD.

    The CWD fallback covers non-editable installs (e.g. Streamlit Cloud), where
    the package lives in site-packages but the app runs from the repo root.
    """
    if env := os.environ.get("WINEMOOD_ROOT"):
        return Path(env).resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "configs").is_dir():
        return checkout
    return Path.cwd()


REPO_ROOT = _find_repo_root()
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.toml"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True)
class DataConfig:
    raw_csv: Path = Path("data/raw/winemag-data-130k-v2.csv")
    text_column: str = "description"
    target_column: str = "points"
    id_column: str = "review_id"
    url: str = ""
    sha256: str = ""
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports")
    # Committed record of split fingerprints (counts + hashes, no review text).
    fingerprints: Path = Path("data/split_fingerprints.json")


@dataclass(frozen=True)
class LabelConfig:
    # Label scheme served by the legacy model in the UI (historical compatibility).
    scheme: str = "points_band_v0"


@dataclass(frozen=True)
class TargetConfig:
    primary: str = "points_regression_v1"
    secondary: tuple[str, ...] = ("we_category_v1", "points_binary_90_v1")
    # Pre-declared rule for we_category_merged_v1: merge adjacent top bands until
    # every class holds at least this fraction of the TRAINING split.
    min_class_fraction: float = 0.01


@dataclass(frozen=True)
class NearDupConfig:
    enabled: bool = True
    shingle_size: int = 3
    num_perm: int = 128
    bands: int = 16
    threshold: float = 0.8


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    validation_size: float = 0.1
    deduplicate: bool = True
    name: str = "grouped_v1"
    n_folds: int = 10
    test_folds: int = 2
    validation_folds: int = 1
    stratify_on: str = "we_category_v1"
    group_by_title: bool = True


@dataclass(frozen=True)
class PreprocessingConfig:
    lowercase: bool = True
    strip_punctuation: bool = True
    remove_stopwords: bool = True
    keep_negations: bool = False
    max_sequence_length: int = 100
    mask_numbers: str = "none"  # "none" | "years" | "all" (Phase 3, H4)


@dataclass(frozen=True)
class ExperimentsConfig:
    spec: Path = Path("configs/experiments.toml")
    dir: Path = Path("experiments")  # git-ignored: models, predictions, histories
    results_dir: Path = Path("results")  # committed: small metrics/selection/test log
    bootstrap_resamples: int = 2000


@dataclass(frozen=True)
class ArtifactConfig:
    dir: Path = Path("artifacts/legacy_lstm")
    model_file: str = "model.h5"
    tokenizer_json: str = "tokenizer.json"
    tokenizer_pickle: str = "tokenizer.pickle"

    @property
    def model_path(self) -> Path:
        return self.dir / self.model_file

    @property
    def tokenizer_json_path(self) -> Path:
        return self.dir / self.tokenizer_json

    @property
    def tokenizer_pickle_path(self) -> Path:
        return self.dir / self.tokenizer_pickle


@dataclass(frozen=True)
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    neardup: NearDupConfig = field(default_factory=NearDupConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{section}]: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None, root: Path = REPO_ROOT) -> Config:
    """Load a TOML config file. Relative paths inside it resolve against ``root``."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    data = _build(DataConfig, dict(raw.get("data", {})), "data")
    data = replace(
        data,
        raw_csv=_resolve(data.raw_csv, root),
        processed_dir=_resolve(data.processed_dir, root),
        reports_dir=_resolve(data.reports_dir, root),
        fingerprints=_resolve(data.fingerprints, root),
    )
    artifacts = _build(ArtifactConfig, dict(raw.get("artifacts", {})), "artifacts")
    artifacts = replace(artifacts, dir=_resolve(artifacts.dir, root))

    split = _build(SplitConfig, dict(raw.get("split", {})), "split")
    if not 0 < split.test_size < 1 or not 0 <= split.validation_size < 1:
        raise ConfigError("split.test_size must be in (0, 1) and validation_size in [0, 1)")
    if split.test_folds + split.validation_folds >= split.n_folds:
        raise ConfigError("split.test_folds + split.validation_folds must be < split.n_folds")

    experiments = _build(ExperimentsConfig, dict(raw.get("experiments", {})), "experiments")
    experiments = replace(
        experiments,
        spec=_resolve(experiments.spec, root),
        dir=_resolve(experiments.dir, root),
        results_dir=_resolve(experiments.results_dir, root),
    )

    targets = dict(raw.get("targets", {}))
    if "secondary" in targets:
        targets["secondary"] = tuple(targets["secondary"])

    return Config(
        seed=int(raw.get("project", {}).get("seed", 42)),
        data=data,
        labels=_build(LabelConfig, dict(raw.get("labels", {})), "labels"),
        targets=_build(TargetConfig, targets, "targets"),
        neardup=_build(NearDupConfig, dict(raw.get("neardup", {})), "neardup"),
        split=split,
        preprocessing=_build(
            PreprocessingConfig, dict(raw.get("preprocessing", {})), "preprocessing"
        ),
        artifacts=artifacts,
        experiments=experiments,
    )
