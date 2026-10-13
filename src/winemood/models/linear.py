"""Linear model heads with safe (JSON + .npy) persistence.

Training uses scikit-learn; prediction uses only the stored weights, so a
reloaded model needs no pickled estimator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

HEAD_KINDS = ("constant", "ols", "ridge", "logistic", "ordinal", "majority", "stratified")


def _dot(x, w: np.ndarray) -> np.ndarray:
    return np.asarray(x @ w).ravel() if w.ndim == 1 else np.asarray(x @ w.T)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class Head:
    """A fitted linear head: regression (``predict``) or classification
    (``predict_proba`` / ``predict``)."""

    def __init__(self, kind: str, params: dict[str, Any] | None = None, seed: int = 42) -> None:
        if kind not in HEAD_KINDS:
            raise ValueError(f"Unknown head '{kind}'")
        self.kind = kind
        self.params = dict(params or {})
        self.seed = seed
        self.weights: dict[str, np.ndarray] = {}
        self.n_classes: int | None = None

    # -- training --------------------------------------------------------------
    def fit(self, x, y: np.ndarray, n_classes: int | None = None) -> Head:
        y = np.asarray(y)
        self.n_classes = n_classes
        if self.kind == "constant":
            self.weights["value"] = np.array([float(np.mean(y))])
        elif self.kind == "ols":
            model = LinearRegression().fit(x, y)
            self._store_linear(model.coef_, model.intercept_)
        elif self.kind == "ridge":
            model = Ridge(alpha=float(self.params["alpha"]), random_state=self.seed).fit(x, y)
            self._store_linear(model.coef_, model.intercept_)
        elif self.kind == "majority":
            prior = np.bincount(y.astype(int), minlength=n_classes) / len(y)
            self.weights["prior"] = prior
        elif self.kind == "stratified":
            self.weights["prior"] = np.bincount(y.astype(int), minlength=n_classes) / len(y)
        elif self.kind == "logistic":
            model = self._logistic().fit(x, y)
            self._store_linear(model.coef_, model.intercept_)
        elif self.kind == "ordinal":
            # Frank & Hall (2001): one binary model per threshold, P(y > k).
            coefs, intercepts = [], []
            for k in range(n_classes - 1):
                model = self._logistic().fit(x, (y > k).astype(int))
                coefs.append(model.coef_.ravel())
                intercepts.append(float(model.intercept_[0]))
            self.weights["coef"] = np.vstack(coefs).astype(np.float32)
            self.weights["intercept"] = np.asarray(intercepts)
        return self

    def _logistic(self) -> LogisticRegression:
        return LogisticRegression(
            C=float(self.params.get("C", 1.0)),
            class_weight=self.params.get("class_weight"),
            max_iter=int(self.params.get("max_iter", 1000)),
            random_state=self.seed,
        )

    def _store_linear(self, coef, intercept) -> None:
        self.weights["coef"] = np.asarray(coef, dtype=np.float32)
        self.weights["intercept"] = np.atleast_1d(np.asarray(intercept, dtype=float))

    # -- prediction ------------------------------------------------------------
    def predict(self, x) -> np.ndarray:
        n = x.shape[0]
        if self.kind == "constant":
            return np.full(n, self.weights["value"][0])
        if self.kind in ("ols", "ridge"):
            return _dot(x, self.weights["coef"]) + self.weights["intercept"][0]
        if self.kind == "stratified":
            rng = np.random.default_rng(self.seed)
            return rng.choice(len(self.weights["prior"]), size=n, p=self.weights["prior"])
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_proba(self, x) -> np.ndarray:
        n = x.shape[0]
        if self.kind in ("majority", "stratified"):
            return np.tile(self.weights["prior"], (n, 1))
        if self.kind == "logistic":
            z = _dot(x, self.weights["coef"]) + self.weights["intercept"]
            if self.weights["coef"].shape[0] == 1:  # binary
                p1 = _sigmoid(np.asarray(z).ravel())
                return np.column_stack([1 - p1, p1])
            return _softmax(np.asarray(z))
        if self.kind == "ordinal":
            greater = _sigmoid(_dot(x, self.weights["coef"]) + self.weights["intercept"])
            greater = np.atleast_2d(greater)
            # Enforce monotone P(y > k) so class probabilities are non-negative.
            greater = np.minimum.accumulate(greater, axis=1)
            cum = np.column_stack([np.ones(n), greater, np.zeros(n)])
            proba = cum[:, :-1] - cum[:, 1:]
            return proba / proba.sum(axis=1, keepdims=True)
        raise ValueError(f"Head '{self.kind}' has no probabilities")

    # -- persistence -----------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind,
            "params": self.params,
            "seed": self.seed,
            "n_classes": self.n_classes,
            "weights": sorted(self.weights),
        }
        (directory / "head.json").write_text(json.dumps(meta), encoding="utf-8")
        for name, arr in self.weights.items():
            np.save(directory / f"head_{name}.npy", arr, allow_pickle=False)

    @classmethod
    def load(cls, directory: Path) -> Head:
        meta = json.loads((directory / "head.json").read_text(encoding="utf-8"))
        head = cls(meta["kind"], meta["params"], meta["seed"])
        head.n_classes = meta["n_classes"]
        head.weights = {
            name: np.load(directory / f"head_{name}.npy", allow_pickle=False)
            for name in meta["weights"]
        }
        return head
