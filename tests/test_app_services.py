"""Application services: model loading, artifacts, explanations, results, batch."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import make_model_dir

from winemood.app.services import batch
from winemood.app.services.explain import ExplanationError, coefficient_table, explain, feature_kind
from winemood.app.services.model_service import load_model_state
from winemood.app.services.results import Interval, load_results
from winemood.config import REPO_ROOT
from winemood.models.artifact import ArtifactVerificationError, install, package, verify
from winemood.models.linear import Head

RESULTS = REPO_ROOT / "results"


def _env(path: Path) -> dict[str, str]:
    return {"WINEMOOD_MODEL_DIR": str(path)}


# --- model loading -----------------------------------------------------------------


def test_load_local_build_unverified(tmp_path, model_dir):
    state = load_model_state(tmp_path / "nowhere", _env(model_dir))
    assert state.ok and not state.verified and state.source == "WINEMOOD_MODEL_DIR"
    preds = state.predictor.predict(["Ripe cherry and plum with a long finish.", "Thin and sour."])
    assert len(preds) == 2 and all(80 <= p.score <= 100 for p in preds)


def test_missing_artifact_reports_where_it_looked(tmp_path):
    state = load_model_state(tmp_path, _env(tmp_path / "missing"))
    assert not state.ok and "No model artifact" in state.error
    assert len(state.tried) == 1  # an explicit directory is never silently replaced
    fallback = load_model_state(tmp_path, {})
    assert not fallback.ok and len(fallback.tried) == 2


def test_packaged_artifact_verifies_and_detects_tampering(tmp_path, model_dir):
    archive = tmp_path / "e04.zip"
    manifest = package(model_dir, archive)
    assert manifest["preprocessing_version"] == "norm_min_v1"
    assert set(manifest["files"]) >= {"run.json", "model/head.json", "model/features.json"}
    target = tmp_path / "installed"
    install(archive, target)
    assert (target / "LICENSE-MODEL.md").read_text().count("CC BY-NC-SA") >= 1
    state = load_model_state(tmp_path, _env(target))
    assert state.ok and state.verified
    head = target / "model" / "head_coef.npy"
    head.write_bytes(head.read_bytes()[:-8] + b"\x00" * 8)
    state = load_model_state(tmp_path, _env(target))
    assert not state.ok and "verification" in state.error
    with pytest.raises(ArtifactVerificationError):
        verify(target)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.payload


def test_hosted_download_verifies_archive_checksum(tmp_path, model_dir, monkeypatch):
    import hashlib
    import urllib.request

    archive = tmp_path / "e04.zip"
    package(model_dir, archive)
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response(payload))
    root = tmp_path / "deploy"
    env = {"WINEMOOD_MODEL_URL": "https://example.invalid/e04.zip", "WINEMOOD_MODEL_SHA256": digest}
    state = load_model_state(root, env)
    assert state.ok and state.verified and state.source == "downloaded release"
    assert (root / "models" / "e04" / "manifest.json").is_file()
    bad = load_model_state(tmp_path / "deploy2", {**env, "WINEMOOD_MODEL_SHA256": "0" * 64})
    assert not bad.ok and "does not match" in bad.error
    assert not (tmp_path / "deploy2" / "models" / "e04").exists()


def test_install_refuses_unsafe_or_incomplete_archives(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.json", "{}")
        zf.writestr("manifest.json", "{}")
    with pytest.raises(ArtifactVerificationError, match="unsafe"):
        install(bad, tmp_path / "t1")
    no_manifest = tmp_path / "nm.zip"
    with zipfile.ZipFile(no_manifest, "w") as zf:
        zf.writestr("run.json", "{}")
    with pytest.raises(ArtifactVerificationError, match="manifest"):
        install(no_manifest, tmp_path / "t2")


def test_preprocessing_mismatch_is_refused(tmp_path, corpus):
    d = make_model_dir(tmp_path / "v2", corpus, preprocessing="norm_min_v2")
    state = load_model_state(tmp_path, _env(d))
    assert not state.ok and "Preprocessing mismatch" in state.error


def test_wrong_experiment_and_corrupt_files_are_refused(tmp_path, corpus):
    d = make_model_dir(tmp_path / "other", corpus, experiment="E42_random_tfidf")
    assert "Expected E04_tfidf_word" in load_model_state(tmp_path, _env(d)).error
    d2 = make_model_dir(tmp_path / "corrupt", corpus)
    (d2 / "model" / "features.json").write_text("{not json", encoding="utf-8")
    state = load_model_state(tmp_path, _env(d2))
    assert not state.ok and "could not be loaded" in state.error


def test_loading_the_model_never_imports_tensorflow(model_dir):
    code = "\n".join(
        [
            "import sys, pathlib",
            "from winemood.app.services.model_service import load_model_state",
            "env = {'WINEMOOD_MODEL_DIR': sys.argv[1]}",
            "s = load_model_state(pathlib.Path('.'), env)",
            "s.predictor.predict(['Ripe cherry, long finish.'])",
            "heavy = ('tensorflow', 'keras', 'torch', 'transformers')",
            "print(s.ok, any(m in sys.modules for m in heavy))",
        ]
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(model_dir)], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["True", "False"]


# --- explanations ------------------------------------------------------------------


def test_explanation_reconstructs_prediction(tmp_path, model_dir):
    state = load_model_state(tmp_path, _env(model_dir))
    pipe, head = state.predictor.linear_components()
    text = "Ripe cherry and plum, dense tannins and a long finish. Drink 2020."
    exp = explain(pipe, head, text)
    raw = float(state.predictor.predict_raw([text])[0])
    assert exp.raw_score == pytest.approx(raw, abs=1e-3)
    assert exp.intercept + sum(r.contribution for r in exp.contributions) == pytest.approx(
        exp.raw_score, abs=1e-9
    )
    assert exp.reconstruction_error < 1e-6
    assert any(r.kind == "bigram" for r in exp.contributions)
    assert exp.displayed_score == pytest.approx(np.clip(raw, 80, 100))
    contribs = [abs(r.contribution) for r in exp.contributions]
    assert contribs == sorted(contribs, reverse=True)


def test_explanation_requires_linear_regression_head(model_dir, tmp_path):
    state = load_model_state(tmp_path, _env(model_dir))
    pipe, _ = state.predictor.linear_components()
    with pytest.raises(ExplanationError):
        explain(pipe, Head("logistic"), "text")


def test_coefficient_table_and_feature_kinds(model_dir, tmp_path):
    state = load_model_state(tmp_path, _env(model_dir))
    table = coefficient_table(*state.predictor.linear_components())
    assert set(table["kind"]) <= {"unigram", "bigram", "number"}
    assert (table["direction"] == np.where(table["coefficient"] >= 0, "raises", "lowers")).all()
    assert feature_kind("92 94") == "number" and feature_kind("long finish") == "bigram"


def test_clip_policy(tmp_path, model_dir):
    state = load_model_state(tmp_path, _env(model_dir))
    pipe, head = state.predictor.linear_components()
    head.weights["intercept"] = np.array([150.0])  # force an out-of-range raw output
    exp = explain(pipe, head, "ripe cherry")
    assert exp.raw_score > 100 and exp.displayed_score == 100.0 and exp.clipped


# --- research results -------------------------------------------------------------------


def test_results_match_committed_records():
    results = load_results(RESULTS)
    assert results.available and not results.missing
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    for row in report["test_t1"]:
        m = results.models[row["experiment"]]
        assert m.mae.estimate == row["mae"]["estimate"]
        assert (m.mae.low, m.mae.high) == (row["mae"]["ci95_low"], row["mae"]["ci95_high"])
        test = json.loads((RESULTS / "test" / f"{row['experiment']}.json").read_text())
        assert m.mae.estimate == pytest.approx(test["test_metrics"]["mae"], abs=1e-12)
    assert results.selected.experiment == results.selection["final_model"] == "E04_tfidf_word"
    lstm = results.models["E10_lstm"]
    assert lstm.n_seeds == 5 and "Mean of 5 training seeds" in lstm.metric_note
    assert lstm.ensemble_mae != lstm.mae.estimate
    assert [m.experiment for m in results.main_comparison()][:2] == ["E01_dummy_mean", "E02_length"]
    assert results.test_access_count == len(report["test_access_log"]) == 19


def test_missing_results_are_reported(tmp_path):
    results = load_results(tmp_path)
    assert not results.available and results.missing


def test_interval_formatting():
    i = Interval(1.30228, 1.28911, 1.31505)
    assert i.fmt(3) == "1.302 [1.289, 1.315]"
    assert Interval(-0.0158, -0.0255, -0.0065).fmt(4, signed=True) == "-0.0158 [-0.0255, -0.0065]"


# --- batch ----------------------------------------------------------------------------------


def test_batch_parse_and_predict(model_dir, tmp_path):
    state = load_model_state(tmp_path, _env(model_dir))
    data = (
        "id,Review\n1,Ripe cherry and plum with a long finish.\n2,\n3,92-94 Barrel sample. Rich.\n"
    )
    parsed = batch.parse_csv(data.encode("utf-8-sig"))
    assert parsed.column == "Review" and len(parsed.reviews) == 2 and parsed.skipped_empty == 1
    frame = batch.predict_batch(state.predictor, parsed)
    assert list(frame.columns) == [
        "review", "predicted_critic_score", "wine_enthusiast_category",
        "raw_output_outside_80_100", "starts_with_score_range",
    ]  # fmt: skip
    assert frame["starts_with_score_range"].tolist() == [False, True]
    exported = batch.to_csv_bytes(frame).decode("utf-8")
    assert exported.startswith("# Model estimates") and "predicted_critic_score" in exported


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"x" * (batch.MAX_BYTES + 1), "larger than"),
        (b"\xff\xfe\x00bad", "UTF-8"),
        (b"notes\nsome text here\n", "No review column"),
        (b"review\n\n\n", "empty"),
        (("review\n" + "a fine wine\n" * (batch.MAX_ROWS + 1)).encode(), "Too many"),
        (("review\n" + "x" * (batch.MAX_CHARS + 1) + "\n").encode(), "exceed"),
    ],
    ids=["too-large", "not-utf8", "no-column", "empty", "too-many-rows", "too-long"],
)
def test_batch_validation_errors(data, message):
    with pytest.raises(batch.BatchError, match=message):
        batch.parse_csv(data)


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("", False),
        ("short", False),
        ("A ripe wine here.", True),
        ("x" * (batch.MAX_CHARS + 1), False),
    ],
    ids=["empty", "short", "ok", "too-long"],
)
def test_single_review_validation(text, ok):
    if ok:
        assert batch.validate_single(text) == text
    else:
        with pytest.raises(batch.BatchError):
            batch.validate_single(text)
