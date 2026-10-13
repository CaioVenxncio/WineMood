import numpy as np
import pytest
import scipy.sparse as sp

from winemood.models.linear import Head


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6))
    y = 88 + x @ np.array([1.5, -1.0, 0.5, 0, 0, 0]) + rng.normal(scale=0.3, size=300)
    return x, y


@pytest.mark.parametrize(
    ("kind", "params"), [("constant", {}), ("ols", {}), ("ridge", {"alpha": 1.0})]
)
def test_regression_heads_roundtrip(tmp_path, xy, kind, params):
    x, y = xy
    head = Head(kind, params).fit(x, y)
    head.save(tmp_path)
    assert not list(tmp_path.glob("*.pkl"))
    loaded = Head.load(tmp_path)
    np.testing.assert_allclose(loaded.predict(x), head.predict(x))
    if kind == "constant":
        assert np.allclose(head.predict(x), y.mean())
    else:
        assert np.mean(np.abs(head.predict(x) - y)) < 0.5


def test_ridge_on_sparse(xy):
    x, y = xy
    head = Head("ridge", {"alpha": 1.0}).fit(sp.csr_matrix(x), y)
    np.testing.assert_allclose(head.predict(sp.csr_matrix(x)), head.predict(x), rtol=1e-5)


def _classes(x):
    score = x[:, 0] + 0.5 * x[:, 1]
    return np.digitize(score, [-1.0, 0.0, 1.0])  # 4 ordered classes


@pytest.mark.parametrize("kind", ["logistic", "ordinal", "majority", "stratified"])
def test_classification_heads(tmp_path, xy, kind):
    x, _ = xy
    y = _classes(x)
    head = Head(kind, {"C": 1.0}).fit(x, y, n_classes=4)
    proba = head.predict_proba(x)
    assert proba.shape == (300, 4)
    assert np.all(proba >= -1e-12)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    head.save(tmp_path)
    loaded = Head.load(tmp_path)
    np.testing.assert_allclose(loaded.predict_proba(x), proba)
    if kind in ("logistic", "ordinal"):
        assert np.mean(head.predict(x) == y) > 0.8
    if kind == "majority":
        assert set(head.predict(x)) == {np.bincount(y).argmax()}


def test_binary_logistic(xy):
    x, _ = xy
    y = (x[:, 0] > 0).astype(int)
    head = Head("logistic", {"C": 1.0, "class_weight": "balanced"}).fit(x, y, n_classes=2)
    assert head.predict_proba(x).shape == (300, 2)
    assert np.mean(head.predict(x) == y) > 0.9


def test_unknown_head():
    with pytest.raises(ValueError):
        Head("forest")
