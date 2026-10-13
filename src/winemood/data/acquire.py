"""Reproducible dataset acquisition with checksum verification.

The Kaggle download endpoint for this public dataset redirects to a signed
storage URL and currently needs no credentials. If that changes, use the
manual or Kaggle-CLI route documented in data/README.md; credentials are
never read or stored by this module.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from winemood.logging_utils import get_logger

logger = get_logger(__name__)

MANUAL_INSTRUCTIONS = (
    "Download 'winemag-data-130k-v2.csv' manually from "
    "https://www.kaggle.com/datasets/zynicide/wine-reviews, or run "
    "'kaggle datasets download -d zynicide/wine-reviews -f winemag-data-130k-v2.csv "
    "-p data/raw' with credentials in ~/.kaggle/kaggle.json (never in the repo). "
    "Then run 'winemood data verify'."
)


class AcquisitionError(RuntimeError):
    """Download, extraction or checksum verification failed."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts and bool(p.name)


def extract_member(archive: bytes, member: str, dest: Path) -> Path:
    """Extract a single, safely named member of a zip archive to ``dest``."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(f"Downloaded file is not a zip archive: {exc}") from exc
    with zf:
        names = zf.namelist()
        unsafe = [n for n in names if not _safe_member(n)]
        if unsafe:
            raise AcquisitionError(f"Refusing archive with unsafe member paths: {unsafe}")
        matches = [n for n in names if PurePosixPath(n).name == member]
        if len(matches) != 1:
            raise AcquisitionError(f"Expected one '{member}' in archive, found {names}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with zf.open(matches[0]) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out)
        tmp.replace(dest)
    return dest


def verify_checksum(path: Path, expected_sha256: str | None) -> str:
    """Return the file's SHA-256; raise if it differs from a pinned value."""
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256.lower():
        raise AcquisitionError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}. "
            "The upstream file may have changed; do not proceed without investigating."
        )
    if not expected_sha256:
        logger.warning(
            "no pinned checksum; record this value in configs/default.toml [data] sha256",
            extra={"sha256": actual},
        )
    return actual


def download_dataset(
    url: str,
    dest: Path,
    expected_sha256: str | None,
    *,
    timeout: float = 120.0,
    overwrite: bool = False,
) -> str:
    """Download the zipped CSV from ``url``, extract it to ``dest`` and verify it."""
    if dest.is_file() and not overwrite:
        logger.info("dataset already present; verifying", extra={"path": str(dest)})
        return verify_checksum(dest, expected_sha256)
    request = urllib.request.Request(url, headers={"User-Agent": "winemood-acquire/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AcquisitionError(f"Download failed ({exc}). {MANUAL_INSTRUCTIONS}") from exc
    logger.info("downloaded archive", extra={"bytes": len(payload)})
    extract_member(payload, dest.name, dest)
    try:
        return verify_checksum(dest, expected_sha256)
    except AcquisitionError:
        dest.unlink(missing_ok=True)
        raise
