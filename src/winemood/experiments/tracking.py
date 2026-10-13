"""Lightweight experiment tracking: run metadata, registry and test-access log.

Everything is plain JSON/CSV. Large artifacts live under the git-ignored
``experiments/`` directory; small result summaries and the test-access log
live under the committed ``results/`` directory.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from winemood.config import REPO_ROOT

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "tensorflow",
    "keras",
    "torch",
    "transformers",
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id(exp_id: str) -> str:
    return f"{exp_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # optional

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


def run_metadata(exp_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "run_id": new_run_id(exp_id),
        "experiment": exp_id,
        "started": now_iso(),
        "git": git_state(),
        "packages": package_versions(),
        "hardware": hardware(),
        **extra,
    }


class Timer:
    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def append_registry(experiments_dir: Path, row: dict[str, Any]) -> None:
    path = experiments_dir / "registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "experiment", "split", "status", "seconds", "selected_hash", "metric", "value"]
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


# --- Test-set access guard ---------------------------------------------------------


class TestGuardError(PermissionError):
    pass


def test_log_path(results_dir: Path) -> Path:
    return results_dir / "test_access_log.jsonl"


def logged_accesses(results_dir: Path) -> list[dict[str, Any]]:
    path = test_log_path(results_dir)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def guard_test_access(
    results_dir: Path, exp_id: str, cfg_hash: str, *, allow_repeat: bool = False
) -> None:
    """Refuse without a frozen selection; refuse repeats unless explicitly allowed."""
    selection = results_dir / "selection.json"
    if not selection.is_file():
        raise TestGuardError("No frozen selection (results/selection.json); run 'winemood exp select'.")
    if not read_json(selection).get("frozen"):
        raise TestGuardError("results/selection.json is not marked frozen")
    previous = [
        a for a in logged_accesses(results_dir) if a["experiment"] == exp_id and a["config_hash"] == cfg_hash
    ]
    if previous and not allow_repeat:
        raise TestGuardError(
            f"{exp_id} ({cfg_hash}) was already evaluated on test at {previous[0]['timestamp']}; "
            "pass --allow-repeat to re-evaluate (it will be logged as a repeat)."
        )


def log_test_access(
    results_dir: Path, exp_id: str, cfg_hash: str, split: str, *, repeat: bool, note: str = ""
) -> None:
    path = test_log_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": now_iso(),
        "experiment": exp_id,
        "config_hash": cfg_hash,
        "split": split,
        "repeat": repeat,
        "git_commit": git_state()["commit"],
        "note": note,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
