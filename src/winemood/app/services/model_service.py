"""Locate, verify and load the selected score model for the application.

Resolution order: ``$WINEMOOD_MODEL_DIR`` (exclusive) -> ``$WINEMOOD_MODEL_URL``
(exclusive; the release archive is downloaded once into a directory named after
its SHA-256, so changing the configured checksum always installs that exact
release) -> ``models/e04`` (installed release archive) ->
``experiments/grouped_v1/E04_tfidf_word`` (local Phase 3 build).
A directory with a ``manifest.json`` must pass checksum verification; a local
build without a manifest is accepted but reported as unverified.

The selected model must be served with the preprocessing it was trained with
(``norm_min_v1``, no number masking). Anything else is refused, so the A-31
correction (``norm_min_v2``) can never be applied silently to E04.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from winemood.inference.predictor import ArtifactError
from winemood.models.artifact import ArtifactVerificationError, fetch_and_install, verify

SELECTED_EXPERIMENT = "E04_tfidf_word"
SUPPORTED_PREPROCESSING = {"norm_min_v1"}
SUPPORTED_MASK = {"none"}

SETUP_INSTRUCTIONS = (
    "Install the packaged model (CC BY-NC-SA 4.0 release asset):\n"
    "    winemood model install path/to/E04_tfidf_word.zip\n"
    "or rebuild it from the dataset:\n"
    "    winemood data download && winemood data split && winemood exp run E04_tfidf_word\n"
    "or point WINEMOOD_MODEL_DIR at an installed model directory. See docs/deployment.md."
)


@dataclass
class ModelState:
    ok: bool
    predictor: Any = None  # ScorePredictor when ok
    directory: Path | None = None
    source: str = ""
    verified: bool = False
    manifest: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    load_seconds: float = 0.0
    tried: list[str] = field(default_factory=list)


def release_dir(root: Path, archive_sha256: str) -> Path:
    """Install location of a downloaded release archive, keyed by its checksum."""
    return root / "models" / f"release-{archive_sha256.strip().lower()[:16]}"


def candidate_dirs(root: Path, env: Mapping[str, str] | None = None) -> list[tuple[str, Path]]:
    """An explicit WINEMOOD_MODEL_DIR or WINEMOOD_MODEL_URL is used exclusively (never
    silently replaced by another model); otherwise the installed release, then the
    local build."""
    env = os.environ if env is None else env
    if env.get("WINEMOOD_MODEL_DIR"):
        return [("WINEMOOD_MODEL_DIR", Path(env["WINEMOOD_MODEL_DIR"]))]
    if env.get("WINEMOOD_MODEL_URL"):
        return [("downloaded release", release_dir(root, env.get("WINEMOOD_MODEL_SHA256", "")))]
    return [
        ("installed release", root / "models" / "e04"),
        ("local Phase 3 build", root / "experiments" / "grouped_v1" / SELECTED_EXPERIMENT),
    ]


def load_model_state(root: Path, env: Mapping[str, str] | None = None) -> ModelState:
    env = os.environ if env is None else env
    tried = []
    for source, directory in candidate_dirs(root, env):
        tried.append(f"{source}: {directory}")
        if not (directory / "run.json").is_file():
            continue
        state = _load(directory, source, tried)
        if state.ok or source != "downloaded release":
            return state
        tried.append(f"reinstalling damaged download: {state.error}")
    # Hosted deployments: fetch the release archive once, verified by checksum.
    if env.get("WINEMOOD_MODEL_URL") and not env.get("WINEMOOD_MODEL_DIR"):
        digest = env.get("WINEMOOD_MODEL_SHA256", "").strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            return ModelState(
                ok=False,
                error="WINEMOOD_MODEL_SHA256 must be the 64-character hex SHA-256 of the "
                "release archive.",
                tried=tried,
            )
        target = release_dir(root, digest)
        tried.append(f"download: {env['WINEMOOD_MODEL_URL']}")
        if target.exists():
            # Left over from an interrupted or damaged install; the directory is
            # app-managed and named after the checksum, so it is safe to replace.
            shutil.rmtree(target)
        try:
            fetch_and_install(env["WINEMOOD_MODEL_URL"], digest, target)
        except ArtifactVerificationError as exc:
            return ModelState(ok=False, error=f"Model download was refused: {exc}", tried=tried)
        return _load(target, "downloaded release", tried)
    return ModelState(ok=False, error="No model artifact was found.", tried=tried)


def _load(directory: Path, source: str, tried: list[str]) -> ModelState:
    started = time.perf_counter()
    manifest: dict[str, Any] = {}
    verified = False
    if (directory / "manifest.json").is_file():
        try:
            manifest = verify(directory)
            verified = True
        except ArtifactVerificationError as exc:
            return ModelState(
                ok=False,
                directory=directory,
                source=source,
                tried=tried,
                error=f"Model artifact failed verification: {exc}",
            )
    try:
        from winemood.inference.regression import ScorePredictor

        predictor = ScorePredictor(directory)
    except ArtifactError as exc:
        return ModelState(
            ok=False,
            directory=directory,
            source=source,
            tried=tried,
            error=f"Model artifact could not be loaded: {exc}",
        )
    if predictor.experiment != SELECTED_EXPERIMENT or predictor.family != "linear":
        return ModelState(
            ok=False,
            directory=directory,
            source=source,
            tried=tried,
            error=f"Expected {SELECTED_EXPERIMENT} (linear), found "
            f"{predictor.experiment} ({predictor.family}).",
        )
    if (
        predictor.preprocessing_version not in SUPPORTED_PREPROCESSING
        or predictor.mask_numbers not in SUPPORTED_MASK
        or (manifest and manifest.get("preprocessing_version") != predictor.preprocessing_version)
    ):
        return ModelState(
            ok=False,
            directory=directory,
            source=source,
            tried=tried,
            error="Preprocessing mismatch: the selected model must be served with "
            f"norm_min_v1, found '{predictor.preprocessing_version}'.",
        )
    return ModelState(
        ok=True,
        predictor=predictor,
        directory=directory,
        source=source,
        verified=verified,
        manifest=manifest,
        load_seconds=time.perf_counter() - started,
        tried=tried,
    )
