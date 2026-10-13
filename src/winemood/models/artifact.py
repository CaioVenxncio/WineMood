"""Packaging, installation and verification of trained model artifacts.

A packaged model is a zip archive containing:

* ``run.json`` - the experiment record (selected config, validation metrics, metadata);
* ``model/`` - the pickle-free model files (JSON + .npy);
* ``manifest.json`` - provenance and a SHA-256 for every file;
* ``LICENSE-MODEL.md`` - CC BY-NC-SA 4.0 notice (the model is a derived work).

``install`` verifies every checksum before anything is used, and refuses
archives with unsafe paths. Nothing here downloads or uploads anything.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_FORMAT = 1

LICENSE_TEXT = """# Model licence

This model was trained on the *Wine Reviews* dataset (Kaggle `zynicide/wine-reviews`,
WineEnthusiast reviews scraped in 2017), which is licensed under
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)**.

As a derived work, this model artifact is distributed under the same licence:
non-commercial use only, with attribution, and derivatives shared alike.
https://creativecommons.org/licenses/by-nc-sa/4.0/

The WineMood source code is separately licensed under MIT.

The model predicts critic scores from review text. It does not measure wine
quality or sentiment. See docs/model-card.md.
"""


class ArtifactVerificationError(RuntimeError):
    """An artifact is missing files, has wrong checksums or an unsafe layout."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(directory: Path) -> list[Path]:
    files = [directory / "run.json"] + sorted((directory / "model").glob("*"))
    missing = [str(f) for f in files if not f.is_file()]
    if missing or not (directory / "model").is_dir():
        raise ArtifactVerificationError(f"Artifact incomplete in {directory}: {missing or 'model/'}")
    return files


def build_manifest(experiment_dir: Path, results_dir: Path | None = None) -> dict[str, Any]:
    run = json.loads((experiment_dir / "run.json").read_text(encoding="utf-8"))
    features = json.loads((experiment_dir / "model" / "features.json").read_text(encoding="utf-8"))
    test_metrics = None
    if results_dir is not None:
        test_path = results_dir / "test" / f"{run['experiment']}.json"
        if test_path.is_file():
            tm = json.loads(test_path.read_text(encoding="utf-8"))["test_metrics"]
            test_metrics = {k: tm[k] for k in ("n", "mae", "rmse", "r2") if k in tm}
    files = {
        str(p.relative_to(experiment_dir).as_posix()): _sha256(p)
        for p in _artifact_files(experiment_dir)
    }
    return {
        "format": MANIFEST_FORMAT,
        "experiment": run["experiment"],
        "task": run["task"],
        "family": run["family"],
        "split": run["split"],
        "config_hash": run["config_hash"],
        "preprocessing_version": features["spec"]["preprocessing"],
        "mask_numbers": features["spec"].get("mask_numbers", "none"),
        "dataset_sha256": run["meta"].get("dataset_sha256"),
        "git_commit": run["meta"].get("git", {}).get("commit"),
        "test_metrics": test_metrics,
        "license": "CC BY-NC-SA 4.0",
        "packaged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }


def package(experiment_dir: Path, out_path: Path, results_dir: Path | None = None) -> dict:
    """Write a verified zip archive of an experiment's selected model."""
    manifest = build_manifest(experiment_dir, results_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in manifest["files"]:
            zf.write(experiment_dir / rel, rel)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("LICENSE-MODEL.md", LICENSE_TEXT)
    return manifest


def verify(directory: Path) -> dict[str, Any]:
    """Check the manifest's checksums for an installed artifact directory."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactVerificationError(f"No manifest.json in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ArtifactVerificationError(f"Unsupported manifest format {manifest.get('format')}")
    for rel, expected in manifest["files"].items():
        path = directory / rel
        if not path.is_file():
            raise ArtifactVerificationError(f"Missing artifact file: {rel}")
        if _sha256(path) != expected:
            raise ArtifactVerificationError(f"Checksum mismatch for {rel}")
    return manifest


def _safe(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return bool(p.name) and not p.is_absolute() and ".." not in p.parts


def fetch_and_install(
    url: str, archive_sha256: str, target: Path, *, timeout: float = 60.0, opener=None
) -> dict[str, Any]:
    """Download a packaged model, check the *archive* checksum before extracting,
    then install and verify every file. Used by hosted deployments that set
    WINEMOOD_MODEL_URL / WINEMOOD_MODEL_SHA256 (docs/deployment.md)."""
    import tempfile
    import urllib.request

    if not archive_sha256 or len(archive_sha256) != 64:
        raise ArtifactVerificationError("A 64-character archive SHA-256 must be provided")
    open_url = opener or urllib.request.urlopen
    request = urllib.request.Request(url, headers={"User-Agent": "winemood-model-fetch/1.0"})
    try:
        with open_url(request, timeout=timeout) as response:
            payload = response.read()
    except OSError as exc:
        raise ArtifactVerificationError(f"Model download failed: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != archive_sha256.lower():
        raise ArtifactVerificationError("Downloaded archive does not match WINEMOOD_MODEL_SHA256")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.zip"
        archive.write_bytes(payload)
        return install(archive, target)


def install(archive: Path, target: Path) -> dict[str, Any]:
    """Extract ``archive`` into ``target`` and verify it; nothing is kept on failure."""
    if target.exists() and any(target.iterdir()):
        raise ArtifactVerificationError(f"Target {target} is not empty")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        unsafe = [n for n in names if not _safe(n)]
        if unsafe:
            raise ArtifactVerificationError(f"Refusing archive with unsafe paths: {unsafe}")
        if "manifest.json" not in names:
            raise ArtifactVerificationError("Archive has no manifest.json")
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target)
    try:
        return verify(target)
    except ArtifactVerificationError:
        for path in sorted(target.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        raise
