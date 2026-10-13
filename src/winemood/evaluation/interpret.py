"""Model interpretation (associational, not causal).

* Linear models: the largest positive and negative Ridge coefficients over the
  TF-IDF vocabulary. Coefficients describe what the fitted model uses; they do
  not show that a word causes a higher score, and correlated features share
  weight arbitrarily.
* LSTM: leave-one-token-out occlusion on a validation sample - the change in
  predicted score when a token is removed, averaged per token. Occlusion is a
  local, perturbation-based approximation; removing a token also changes the
  sequence context, so effects are not additive explanations.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from winemood.features.pipeline import FeaturePipeline
from winemood.models.linear import Head


def top_coefficients(
    model_dir: Path, k: int = 25, stopwords=None
) -> dict[str, list[dict[str, Any]]]:
    pipe = FeaturePipeline.load(model_dir, stopwords=stopwords)
    head = Head.load(model_dir)
    coef = np.asarray(head.weights["coef"]).ravel()
    names = pipe.feature_names()
    order = np.argsort(coef)
    rows = lambda idx: [{"term": str(names[i]), "coef": float(coef[i])} for i in idx]  # noqa: E731
    special = {
        str(n): float(coef[i])
        for i, n in enumerate(names)
        if str(n).startswith("__") or "__year__" in str(n)
    }
    return {"positive": rows(order[::-1][:k]), "negative": rows(order[:k]), "placeholders": special}


def lstm_occlusion(
    model, texts: list[str], *, min_count: int = 15, max_tokens: int = 80, k: int = 20
) -> dict[str, Any]:
    from winemood.models import lstm

    base = lstm.predict(model, texts)
    variants, owners, tokens = [], [], []
    for i, text in enumerate(texts):
        words = text.split()[:max_tokens]
        for j, w in enumerate(words):
            variants.append(" ".join(words[:j] + words[j + 1 :]))
            owners.append(i)
            tokens.append(w)
    preds = lstm.predict(model, variants)
    deltas = defaultdict(list)
    for owner, token, p in zip(owners, tokens, preds, strict=True):
        deltas[token].append(base[owner] - p)  # > 0: token raises the prediction
    stats = [
        {"token": t, "mean_effect": float(np.mean(v)), "n": len(v)}
        for t, v in deltas.items()
        if len(v) >= min_count
    ]
    stats.sort(key=lambda r: r["mean_effect"])
    return {
        "n_reviews": len(texts),
        "n_occlusions": len(variants),
        "raises_score": stats[::-1][:k],
        "lowers_score": stats[:k],
    }
