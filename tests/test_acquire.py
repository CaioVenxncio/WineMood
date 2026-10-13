"""Acquisition tests: local archives only, no network."""

from __future__ import annotations

import io
import urllib.error
import zipfile
from pathlib import Path

import pytest

from winemood.data import acquire
from winemood.data.acquire import (
    AcquisitionError,
    download_dataset,
    extract_member,
    sha256_file,
    verify_checksum,
)

NAME = "winemag-data-130k-v2.csv"
CONTENT = b",description,points\n0,Ripe cherry.,88\n"


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_and_verify(tmp_path: Path):
    dest = tmp_path / "raw" / NAME
    extract_member(_zip({NAME: CONTENT}), NAME, dest)
    assert dest.read_bytes() == CONTENT
    digest = sha256_file(dest)
    assert verify_checksum(dest, digest) == digest
    assert verify_checksum(dest, digest.upper()) == digest


def test_checksum_mismatch(tmp_path: Path):
    dest = tmp_path / NAME
    dest.write_bytes(CONTENT)
    with pytest.raises(AcquisitionError, match="Checksum mismatch"):
        verify_checksum(dest, "0" * 64)


def test_zip_slip_rejected(tmp_path: Path):
    with pytest.raises(AcquisitionError, match="unsafe"):
        extract_member(_zip({"../evil.csv": b"x", NAME: CONTENT}), NAME, tmp_path / NAME)


def test_missing_member_and_bad_archive(tmp_path: Path):
    with pytest.raises(AcquisitionError, match="Expected one"):
        extract_member(_zip({"other.csv": b"x"}), NAME, tmp_path / NAME)
    with pytest.raises(AcquisitionError, match="not a zip"):
        extract_member(b"not a zip", NAME, tmp_path / NAME)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_with_fake_transport(tmp_path: Path, monkeypatch):
    payload = _zip({NAME: CONTENT})
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **k: _Response(payload))
    dest = tmp_path / NAME
    expected = __import__("hashlib").sha256(CONTENT).hexdigest()
    assert download_dataset("https://example.invalid/x", dest, expected) == expected
    # Already present: verified, not re-downloaded.
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    assert download_dataset("https://example.invalid/x", dest, expected) == expected


def test_download_checksum_failure_removes_file(tmp_path: Path, monkeypatch):
    payload = _zip({NAME: CONTENT})
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **k: _Response(payload))
    dest = tmp_path / NAME
    with pytest.raises(AcquisitionError):
        download_dataset("https://example.invalid/x", dest, "0" * 64)
    assert not dest.exists()


def test_offline_gives_manual_instructions(tmp_path: Path, monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(acquire.urllib.request, "urlopen", boom)
    with pytest.raises(AcquisitionError, match="kaggle datasets download"):
        download_dataset("https://example.invalid/x", tmp_path / NAME, None)
