"""End-to-end CLI tests on a synthetic CSV (validate -> split -> check -> eda)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from winemood.cli import main
from winemood.data.acquire import sha256_file
from winemood.data.ingest import load_reviews


def _config(tmp_path: Path, csv: Path) -> str:
    cfg = tmp_path / "test.toml"
    cfg.write_text(
        f"""
[project]
seed = 1
[data]
raw_csv = "{csv.as_posix()}"
sha256 = "{sha256_file(csv)}"
processed_dir = "{(tmp_path / "processed").as_posix()}"
reports_dir = "{(tmp_path / "reports").as_posix()}"
fingerprints = "{(tmp_path / "fingerprints.json").as_posix()}"
[neardup]
num_perm = 64
bands = 16
""",
        encoding="utf-8",
    )
    return str(cfg)


def test_ingest_exposes_review_id(corpus_csv):
    df = load_reviews(corpus_csv)
    assert df.columns[0] == "review_id"
    assert df.review_id.is_unique
    assert not [c for c in df.columns if c.startswith("Unnamed")]


def test_ingest_assigns_ids_when_missing(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"description": ["a", "b"], "points": [85, 90]}).to_csv(path, index=False)
    assert load_reviews(path).review_id.tolist() == [0, 1]


def test_pipeline_end_to_end(tmp_path, corpus_csv):
    cfg = _config(tmp_path, corpus_csv)
    assert main(["--config", cfg, "data", "verify"]) == 0
    assert main(["--config", cfg, "data", "validate"]) == 0
    report = json.loads((tmp_path / "reports" / "validation.json").read_text(encoding="utf-8"))
    assert report["n_duplicate_text_normalized"] == 20

    assert main(["--config", cfg, "data", "split"]) == 0
    fingerprints = json.loads((tmp_path / "fingerprints.json").read_text(encoding="utf-8"))
    assert set(fingerprints) == {"grouped_v1", "taster_holdout_v1"}
    assert (tmp_path / "processed" / "splits" / "grouped_v1" / "test.txt").is_file()
    assert main(["--config", cfg, "data", "split", "--check"]) == 0

    docs = tmp_path / "docs"
    assert main(["--config", cfg, "data", "eda", "--publish", "--docs-dir", str(docs)]) == 0
    summary = json.loads((tmp_path / "reports" / "eda" / "summary.json").read_text("utf-8"))
    assert summary["descriptive"]["n_rows_raw"] == 630
    assert summary["descriptive"]["n_rows_dedup"] == 610
    assert summary["modelling"]["n_train"] == fingerprints["grouped_v1"]["counts"]["train"]
    report_md = (docs / "data-report.md").read_text(encoding="utf-8")
    assert report_md.startswith("<!-- AUTO-GENERATED")
    assert len(list((docs / "figures").glob("*.png"))) == 8


def test_split_check_detects_tampering(tmp_path, corpus_csv):
    cfg = _config(tmp_path, corpus_csv)
    assert main(["--config", cfg, "data", "split"]) == 0
    path = tmp_path / "fingerprints.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["grouped_v1"]["id_sha256"]["test"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["--config", cfg, "data", "split", "--check"]) == 1


def test_checksum_mismatch_fails_cleanly(tmp_path, corpus_csv):
    cfg = Path(_config(tmp_path, corpus_csv))
    text = cfg.read_text(encoding="utf-8")
    cfg.write_text(text.replace(sha256_file(corpus_csv), "0" * 64), encoding="utf-8")
    assert main(["--config", str(cfg), "data", "validate"]) == 2


def test_missing_dataset_fails_cleanly(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[data]\nraw_csv = "{(tmp_path / "nope.csv").as_posix()}"\n', "utf-8")
    assert main(["--config", str(cfg), "data", "verify"]) == 2
