"""Inference for the Phase 3 score models (for the Phase 4 application).

``ScorePredictor`` loads a saved experiment artifact - a linear TF-IDF model
(JSON + .npy, no pickle) or an LSTM (``.keras``, ``safe_mode=True``) - and
returns a clipped 80-100 score plus the derived Wine Enthusiast category.
Scores are *predicted critic ratings*, not measures of wine quality or sentiment.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from winemood.data.labels import WE_CATEGORY_MERGED_V1, LabelScheme
from winemood.evaluation.metrics import clip_scores, scores_to_classes
from winemood.features.pipeline import FeaturePipeline
from winemood.features.text import sequence_text
from winemood.inference.predictor import ArtifactIncompatibleError, ArtifactNotFoundError
from winemood.models.linear import Head
from winemood.preprocessing import TextPreprocessor, preprocessing_config


@dataclass(frozen=True)
class ScorePrediction:
    text: str
    score: float  # predicted critic score, clipped to 80-100
    category: str  # derived we_category_merged_v1 display name


class ScorePredictor:
    """Loads one saved experiment directory (``experiments/<split>/<id>``)."""

    def __init__(
        self,
        directory: Path,
        *,
        seed: int | None = None,
        stopwords=None,
        scheme: LabelScheme = WE_CATEGORY_MERGED_V1,
    ) -> None:
        self.directory = Path(directory)
        self.scheme = scheme
        run_path = self.directory / "run.json"
        if not run_path.is_file():
            raise ArtifactNotFoundError(f"No experiment artifact at {self.directory}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("task") != "T1":
            raise ArtifactIncompatibleError("ScorePredictor serves T1 score-regression models only")
        self.family = run["family"]
        self.experiment = run["experiment"]
        if self.family == "lstm":
            from winemood.models import lstm

            chosen = seed if seed is not None else run["seeds"][0]
            model_dir = self.directory / f"model_seed{chosen}"
            if not (model_dir / "model.keras").is_file():
                raise ArtifactNotFoundError(f"No LSTM model for seed {chosen} in {self.directory}")
            self._model = lstm.load(model_dir)
            cfg = run["selected_config"]
            self._prep = TextPreprocessor(
                preprocessing_config(cfg["preprocessing"], cfg.get("mask_numbers", "none")),
                stopwords=stopwords,
            )
        elif self.family == "linear":
            model_dir = self.directory / "model"
            self._pipe = FeaturePipeline.load(model_dir, stopwords=stopwords)
            if self._pipe.spec.kind in ("metadata", "encoder"):
                raise ArtifactIncompatibleError(
                    "Only text-only linear models can be served from text"
                )
            self._head = Head.load(model_dir)
        else:
            raise ArtifactIncompatibleError(f"Unsupported model family '{self.family}'")

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        if self.family == "lstm":
            from winemood.models import lstm

            return clip_scores(
                lstm.predict(self._model, [sequence_text(self._prep(t)) for t in texts])
            )
        frame = pd.DataFrame({self._pipe.text_column: list(texts)})
        return clip_scores(self._head.predict(self._pipe.transform(frame)))

    def predict(self, texts: Sequence[str]) -> list[ScorePrediction]:
        scores = self.predict_scores(texts)
        classes = scores_to_classes(scores, self.scheme) if len(scores) else []
        return [
            ScorePrediction(text=t, score=float(s), category=self.scheme.display_names[int(c)])
            for t, s, c in zip(texts, scores, classes, strict=True)
        ]
