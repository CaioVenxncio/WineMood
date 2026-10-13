"""Headless UI tests (streamlit.testing.AppTest) against a synthetic model and the
committed research records."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from winemood.config import REPO_ROOT

APP_SRC = REPO_ROOT / "src" / "winemood" / "app"
# Misleading claims the interface must never make (it may say what it is *not*).
FORBIDDEN = re.compile(
    r"sentiment score|confidence score|% confidence|\bconfident\b|probability that|"
    r"quality score|positive review|negative review",
    re.IGNORECASE,
)


def _page(name: str) -> None:
    import importlib

    from winemood.app.ui import components as c
    from winemood.app.ui.theme import enable_chart_theme

    enable_chart_theme()
    c.inject_css()
    importlib.import_module(f"winemood.app.ui.pages.{name}").render()


def _run(name: str, monkeypatch, model: Path | None, results: Path | None = None) -> AppTest:
    monkeypatch.setenv("WINEMOOD_MODEL_DIR", str(model) if model else "/nonexistent/model")
    if results is not None:
        monkeypatch.setenv("WINEMOOD_RESULTS_DIR", str(results))
    else:
        monkeypatch.delenv("WINEMOOD_RESULTS_DIR", raising=False)
    at = AppTest.from_function(_page, args=(name,), default_timeout=120)
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    parts = [h.proto.body for h in at.get("html")]
    parts += [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += (
        [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    )
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("page", ["predict", "research", "explore"])
def test_pages_render_without_errors(page, monkeypatch, model_dir):
    at = _run(page, monkeypatch, model_dir)
    assert not at.exception
    assert not FORBIDDEN.search(_all_text(at))


def test_predict_flow_shows_score_category_and_aggregate_context(monkeypatch, model_dir):
    at = _run("predict", monkeypatch, model_dir)
    at.text_area(key="predict_text").input(
        "Ripe black cherry and plum, firm tannins and a long, spicy finish."
    ).run()
    at.button[0].click().run()
    assert not at.exception
    text = _all_text(at)
    match = re.search(r'wm-score-value">(\d+\.\d)<', text)
    assert match and 80.0 <= float(match.group(1)) <= 100.0
    assert "wm-badge" in text and "Predicted critic score" in text
    assert "not an uncertainty range for this particular estimate" in text
    assert not FORBIDDEN.search(text)
    labels = [e.label for e in at.expander]
    assert "Why this number? (exact model arithmetic)" in labels and "Limitations" in labels


def test_predict_flags_score_range_input(monkeypatch, model_dir):
    at = _run("predict", monkeypatch, model_dir)
    at.text_area(key="predict_text").input("92-94 Barrel sample. Rich, dense and ripe.").run()
    at.button[0].click().run()
    assert "Score range detected in the text" in _all_text(at)


def test_predict_rejects_empty_input(monkeypatch, model_dir):
    at = _run("predict", monkeypatch, model_dir)
    at.button[0].click().run()
    assert any("Please enter a review" in w.value for w in at.warning)


def test_missing_model_shows_instructions_not_traceback(monkeypatch):
    for page in ("predict", "explore"):
        at = _run(page, monkeypatch, None)
        assert not at.exception
        text = _all_text(at)
        assert "The score model is not available" in text and "winemood model install" in text


def test_missing_results_show_empty_state(monkeypatch, model_dir, tmp_path):
    at = _run("research", monkeypatch, model_dir, results=tmp_path)
    assert not at.exception
    assert "Research results are not available" in _all_text(at)


def test_research_numbers_come_from_records(monkeypatch, model_dir):
    import json

    at = _run("research", monkeypatch, model_dir)
    report = json.loads((REPO_ROOT / "results" / "report.json").read_text(encoding="utf-8"))
    e04 = next(r for r in report["test_t1"] if r["experiment"] == "E04_tfidf_word")
    text = _all_text(at)
    assert f"{e04['mae']['estimate']:.3f} [{e04['mae']['ci95_low']:.3f}" in text
    assert "mean over five independently trained seeds" in text


def test_full_app_starts_headless(monkeypatch, model_dir):
    monkeypatch.setenv("WINEMOOD_MODEL_DIR", str(model_dir))
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception


def test_app_code_never_touches_test_data():
    """Guard: the application reads published aggregates only."""
    offenders = []
    for path in APP_SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"test_predictions|load_split\(|allow_test|data/raw|raw_csv", source):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []
    assert os.path.isdir(APP_SRC)
