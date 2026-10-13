"""Cached access to the model and the research records.

Paths come from the project configuration and can be overridden with
``WINEMOOD_MODEL_DIR`` / ``WINEMOOD_RESULTS_DIR`` (used by deployments and tests).
Resources are cached per path, so the model is loaded once per process. A failed
model load is not cached.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from winemood.app.services.model_service import ModelState, load_model_state
from winemood.app.services.results import ResearchResults, load_results
from winemood.config import REPO_ROOT, load_config


def _root() -> Path:
    return REPO_ROOT


def results_dir() -> Path:
    if os.environ.get("WINEMOOD_RESULTS_DIR"):
        return Path(os.environ["WINEMOOD_RESULTS_DIR"])
    try:
        return load_config().experiments.results_dir
    except Exception:  # config missing in an unusual deployment: fall back to repo layout
        return _root() / "results"


@st.cache_resource(show_spinner="Loading the score model...")
def _model_state(root: str, env_items: tuple[tuple[str, str], ...]) -> ModelState:
    return load_model_state(Path(root), dict(env_items))


def model_state() -> ModelState:
    keys = ("WINEMOOD_MODEL_DIR", "WINEMOOD_MODEL_URL", "WINEMOOD_MODEL_SHA256")
    env_items = tuple((k, os.environ[k]) for k in keys if os.environ.get(k))
    state = _model_state(str(_root()), env_items)
    if not state.ok:
        # Keep only successes: a transient download error, or a model installed after
        # start-up, is retried on the next interaction instead of persisting until restart.
        _model_state.clear()
    return state


@st.cache_resource(show_spinner=False)
def _results(path: str) -> ResearchResults:
    return load_results(Path(path))


def research_results() -> ResearchResults:
    return _results(str(results_dir()))
