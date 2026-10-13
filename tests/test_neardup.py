import numpy as np
import pytest

from winemood.data.neardup import (
    MinHasher,
    UnionFind,
    find_near_duplicates,
    jaccard,
    shingles,
)

BASE = (
    "Ripe black cherry and plum aromas lead to a dense palate with firm tannins, "
    "toasted oak and a long peppery finish that lingers on the palate"
)


def test_shingles_and_jaccard():
    s = shingles("A b c d", 3)
    assert s == frozenset({"a b c", "b c d"})
    assert shingles("Two words", 3) == frozenset({"two words"})
    assert shingles("", 3) == frozenset()
    assert jaccard(s, s) == 1.0
    assert jaccard(frozenset({"x"}), frozenset({"y"})) == 0.0


def test_union_find():
    uf = UnionFind(5)
    uf.union(3, 1)
    uf.union(1, 4)
    assert uf.labels().tolist() == [0, 1, 2, 1, 1]


def test_minhash_is_deterministic_and_estimates_similarity():
    a, b = MinHasher(128, seed=1), MinHasher(128, seed=1)
    s = shingles(BASE)
    assert np.array_equal(a.signature(s), b.signature(s))
    t = shingles(BASE.replace("lingers", "stays"))
    est = float(np.mean(a.signature(s) == a.signature(t)))
    assert abs(est - jaccard(s, t)) < 0.15


def test_identical_near_and_unrelated_texts():
    texts = [
        BASE,
        BASE.upper() + "  ",  # identical after normalisation
        BASE.replace("lingers on the palate", "lingers"),  # near-duplicate
        "Crisp green apple, lime zest and wet stone; light bodied and refreshing.",
        "Simple, soft and fruity; drink now.",
    ]
    result = find_near_duplicates(texts, threshold=0.8)
    c = result.cluster
    assert c[0] == c[1] == c[2]
    assert len({c[0], c[3], c[4]}) == 3
    assert all(sim >= 0.8 for _, _, sim in result.pairs)
    assert result.n_clusters_multi == 1
    assert result.n_texts_in_clusters == 3


def test_threshold_is_respected():
    texts = [BASE, " ".join(BASE.split()[:10]) + " entirely different ending words here now"]
    assert jaccard(shingles(texts[0]), shingles(texts[1])) < 0.8
    result = find_near_duplicates(texts, threshold=0.8)
    assert result.cluster[0] != result.cluster[1]


def test_deterministic(corpus):
    texts = corpus["description"].tolist()
    a = find_near_duplicates(texts, seed=3)
    b = find_near_duplicates(texts, seed=3)
    assert np.array_equal(a.cluster, b.cluster)


def test_planted_near_duplicates_are_found(corpus):
    n = 600
    result = find_near_duplicates(corpus["description"].tolist(), threshold=0.8)
    for j in range(20, 30):
        assert result.cluster[j] == result.cluster[n + 20 + (j - 20)]


def test_bands_must_divide_permutations():
    with pytest.raises(ValueError):
        find_near_duplicates(["a b c"], num_perm=100, bands=16)
