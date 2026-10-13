"""Experiment framework tests on the synthetic corpus (no real data, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from winemood.cli import main
from winemood.config import load_config
from winemood.data.acquire import sha256_file
from winemood.experiments.data import SplitIntegrityError, TestAccessError, load_split
from winemood.experiments.spec import ExperimentSpec, config_hash, load_plan, merge
from winemood.experiments.tracking import TestGuardError, guard_test_access, log_test_access

SPEC = """
[tasks.T1]
target = "points_regression_v1"
kind = "regression"
selection_metric = "mae"
[tasks.T2]
target = "we_category_merged_v1"
kind = "classification"
selection_metric = "macro_f1"
higher_is_better = true
[tasks.T3]
target = "points_binary_90_v1"
kind = "classification"
selection_metric = "macro_f1"
higher_is_better = true

[[experiment]]
id = "E01_dummy_mean"
task = "T1"
config = { features = { kind = "none" }, head = { kind = "constant" } }

[[experiment]]
id = "E02_length"
task = "T1"
config = { features = { kind = "length" }, head = { kind = "ols" } }

[[experiment]]
id = "E04_tfidf_word"
task = "T1"
config.features = { kind = "tfidf", word = { ngram_range = [1, 1], min_df = 1 } }
config.head = { kind = "ridge", alpha = 1.0 }
grid = { "head.alpha" = [0.1, 1.0, 10.0], "features.word.min_df" = [1, 2] }

[[experiment]]
id = "E08_tfidf_mask_years"
task = "T1"
base = "E04_tfidf_word"
config = { features = { mask_numbers = "years" } }

[[experiment]]
id = "E21_t2_logistic"
task = "T2"
base = "E04_tfidf_word"
config = { head = { kind = "logistic", C = 1.0 } }
grid = { "head.class_weight" = ["none", "balanced"] }

[[experiment]]
id = "E40_holdout_tfidf"
task = "T1"
split = "taster_holdout_v1"
config.features = { kind = "tfidf", word = { ngram_range = [1, 1], min_df = 1 } }
config.head = { kind = "ridge", alpha = 1.0 }

[test_scope]
experiments = ["E01_dummy_mean", "E02_length", "E04_tfidf_word", "E08_tfidf_mask_years",
               "E21_t2_logistic", "E40_holdout_tfidf"]
"""


@pytest.fixture
def workspace(tmp_path: Path, corpus_csv: Path) -> tuple[str, Path]:
    spec = tmp_path / "experiments.toml"
    spec.write_text(SPEC, encoding="utf-8")
    cfg = tmp_path / "test.toml"
    cfg.write_text(
        f"""
[project]
seed = 1
[data]
raw_csv = "{corpus_csv.as_posix()}"
sha256 = "{sha256_file(corpus_csv)}"
processed_dir = "{(tmp_path / "processed").as_posix()}"
reports_dir = "{(tmp_path / "reports").as_posix()}"
fingerprints = "{(tmp_path / "fingerprints.json").as_posix()}"
[neardup]
num_perm = 64
bands = 16
[experiments]
spec = "{spec.as_posix()}"
dir = "{(tmp_path / "experiments").as_posix()}"
results_dir = "{(tmp_path / "results").as_posix()}"
bootstrap_resamples = 200
""",
        encoding="utf-8",
    )
    assert main(["--config", str(cfg), "data", "split"]) == 0
    return str(cfg), tmp_path


def test_spec_grid_and_base_merge(tmp_path):
    spec_path = tmp_path / "e.toml"
    spec_path.write_text(SPEC, encoding="utf-8")
    plan = load_plan(spec_path)
    assert len(plan.get("E04_tfidf_word").grid_configs()) == 6
    base = {
        "features": {"kind": "tfidf", "word": {"min_df": 2}},
        "head": {"kind": "ridge", "alpha": 3.0},
    }
    derived = plan.get("E21_t2_logistic").grid_configs(base)
    assert all(c["head"]["kind"] == "logistic" and "alpha" not in c["head"] for c in derived)
    assert all(c["features"]["word"]["min_df"] == 2 for c in derived)
    masked = plan.get("E08_tfidf_mask_years").grid_configs(base)[0]
    assert masked["features"]["mask_numbers"] == "years" and masked["head"]["alpha"] == 3.0
    assert merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}
    assert config_hash({"x": 1, "y": 2}) == config_hash({"y": 2, "x": 1})
    assert ExperimentSpec("x", "T1", "linear", "s", "", {"a": 1}).grid_configs() == [{"a": 1}]


def test_load_split_refuses_test_and_tampering(workspace):
    cfg_path, tmp = workspace
    cfg = load_config(cfg_path)
    with pytest.raises(TestAccessError):
        load_split(cfg, "grouped_v1", ("train", "test"))
    data = load_split(cfg, "grouped_v1")
    assert set(data.frames) == {"train", "validation"}
    assert not set(data.frames["train"].review_id) & set(data.frames["validation"].review_id)
    manifest = tmp / "processed" / "splits" / "grouped_v1" / "train.txt"
    ids = manifest.read_text().split()
    manifest.write_text("\n".join(ids[:-1]) + "\n")
    with pytest.raises(SplitIntegrityError):
        load_split(cfg, "grouped_v1")


def test_guard(tmp_path):
    results = tmp_path / "results"
    with pytest.raises(TestGuardError, match="No frozen selection"):
        guard_test_access(results, "E", "h")
    results.mkdir()
    (results / "selection.json").write_text(json.dumps({"frozen": True}))
    guard_test_access(results, "E", "h")
    log_test_access(results, "E", "h", "grouped_v1", repeat=False)
    with pytest.raises(TestGuardError, match="already evaluated"):
        guard_test_access(results, "E", "h")
    guard_test_access(results, "E", "h", allow_repeat=True)


def test_pipeline_end_to_end(workspace):
    cfg_path, tmp = workspace
    results = tmp / "results"
    assert main(["--config", cfg_path, "exp", "run", "--all"]) == 0
    e04 = json.loads((results / "validation" / "E04_tfidf_word.json").read_text())
    assert e04["n_configs"] == 6 and len(e04["grid"]) == 6
    assert e04["meta"]["dataset_sha256"] and e04["meta"]["git"] is not None
    assert "test" not in e04["meta"]["split_id_sha256"]
    for key in ("mae", "rmse", "median_ae", "r2", "bias"):
        assert key in e04["validation_metrics"]
    val_pred = pd.read_csv(
        tmp / "experiments" / "grouped_v1" / "E04_tfidf_word" / "validation_predictions.csv"
    )
    assert val_pred.y_pred.between(80, 100).all()

    # Rerunning a deterministic experiment reproduces its metrics exactly.
    before = e04["validation_metrics"]["mae"]
    assert main(["--config", cfg_path, "exp", "run", "E04_tfidf_word"]) == 0
    again = json.loads((results / "validation" / "E04_tfidf_word.json").read_text())
    assert again["validation_metrics"]["mae"] == before

    # No test access before selection.
    assert main(["--config", cfg_path, "exp", "final"]) == 2
    assert not (results / "test_access_log.jsonl").exists()

    assert main(["--config", cfg_path, "exp", "select", "--allow-missing"]) == 0
    selection = json.loads((results / "selection.json").read_text())
    assert selection["frozen"] and selection["final_model"] == "E04_tfidf_word"

    assert main(["--config", cfg_path, "exp", "final"]) == 0
    log = [
        json.loads(line) for line in (results / "test_access_log.jsonl").read_text().splitlines()
    ]
    evaluated = {entry["experiment"] for entry in log}
    assert evaluated == {
        "E01_dummy_mean", "E02_length", "E04_tfidf_word", "E08_tfidf_mask_years",
        "E21_t2_logistic", "E40_holdout_tfidf",
        "E16_E04_tfidf_word_as_T2", "E16_E04_tfidf_word_as_T3",
    }  # fmt: skip
    test_e04 = json.loads((results / "test" / "E04_tfidf_word.json").read_text())
    assert test_e04["test_metrics"]["n"] > 0

    # Re-running the final evaluation is refused without an explicit flag.
    assert main(["--config", cfg_path, "exp", "final"]) == 2

    # The saved artifact serves predictions through the inference interface.
    from winemood.inference.predictor import ArtifactIncompatibleError
    from winemood.inference.regression import ScorePredictor

    predictor = ScorePredictor(tmp / "experiments" / "grouped_v1" / "E04_tfidf_word")
    out = predictor.predict(
        ["Ripe cherry, dense tannins, long finish. Drink 2020-2030.", "Simple."]
    )
    assert len(out) == 2 and all(80 <= p.score <= 100 for p in out)
    assert out[0].category in predictor.scheme.display_names
    assert predictor.predict([]) == []
    with pytest.raises(ArtifactIncompatibleError):
        ScorePredictor(tmp / "experiments" / "grouped_v1" / "E21_t2_logistic")

    docs = tmp / "docs"
    assert main(["--config", cfg_path, "exp", "report", "--docs-dir", str(docs)]) == 0
    assert (
        (docs / "results-tables.md").read_text(encoding="utf-8").startswith("<!-- AUTO-GENERATED")
    )
    report = json.loads((results / "report.json").read_text())
    assert "H1" in report["hypotheses"]


def test_lstm_roundtrip_and_determinism(tmp_path):
    pytest.importorskip("keras")
    from winemood.models import lstm

    rng = np.random.default_rng(0)
    words = ["ripe", "cherry", "dense", "light", "simple", "long", "finish", "oak"]
    texts = [" ".join(rng.choice(words, size=12)) for _ in range(120)]
    y = np.array(
        [85 + t.count("dense") + t.count("long") - t.count("simple") for t in texts], float
    )
    params = {"embed_dim": 8, "units": 4, "dense_units": 4, "vocab_size": 50, "max_epochs": 2}
    a = lstm.train_lstm(texts[:100], y[:100], texts[100:], y[100:], params, seed=7)
    b = lstm.train_lstm(texts[:100], y[:100], texts[100:], y[100:], params, seed=7)
    pa, pb = lstm.predict(a.model, texts[100:]), lstm.predict(b.model, texts[100:])
    np.testing.assert_allclose(pa, pb, rtol=1e-6)
    assert pa.shape == (20,)
    lstm.save(a.model, tmp_path / "m")
    loaded = lstm.load(tmp_path / "m")
    np.testing.assert_allclose(lstm.predict(loaded, texts[100:]), pa, rtol=1e-6)
    assert "val_mae_points" in a.history and a.best_epoch >= 1
