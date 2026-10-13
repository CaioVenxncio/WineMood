"""Near-duplicate detection with MinHash + LSH (NumPy only).

Texts are represented as sets of word n-gram shingles. MinHash signatures
estimate Jaccard similarity; LSH banding proposes candidate pairs, which are
then *verified* with the exact Jaccard similarity of the shingle sets, so the
reported pairs never depend on the approximation.

This is a grouping step used to keep near-identical reviews in the same split.
It does not learn any feature transformation, so running it on the full
corpus does not leak information into a model.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from winemood.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN = re.compile(r"\w+")
_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_BUCKET = 200  # buckets larger than this are almost always exact duplicates


def shingles(text: str, size: int = 3) -> frozenset[str]:
    tokens = _TOKEN.findall(text.lower())
    if len(tokens) < size:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _stable_hash(s: str) -> int:
    # Python's hash() is salted per process; blake2b is stable across runs.
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, int(_MERSENNE), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE), size=num_perm, dtype=np.uint64)

    def signature(self, shingle_set: Iterable[str]) -> np.ndarray:
        hashes = np.fromiter((_stable_hash(s) for s in shingle_set), dtype=np.uint64)
        if hashes.size == 0:
            return np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        hashes %= _MERSENNE
        # (a*x + b) mod p with uint64 arithmetic: the product wraps modulo 2**64, so
        # this is not an exactly universal family. That only affects candidate
        # recall, never correctness, because every candidate is verified exactly.
        values = (np.outer(self.a, hashes) + self.b[:, None]) % _MERSENNE
        return values.min(axis=1)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)


@dataclass
class NearDupResult:
    cluster: np.ndarray  # cluster id per input text (smallest member position)
    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # verified pairs

    @property
    def n_clusters_multi(self) -> int:
        _, counts = np.unique(self.cluster, return_counts=True)
        return int((counts > 1).sum())

    @property
    def n_texts_in_clusters(self) -> int:
        _, inverse, counts = np.unique(self.cluster, return_inverse=True, return_counts=True)
        return int((counts[inverse] > 1).sum())


def find_near_duplicates(
    texts: Sequence[str],
    *,
    shingle_size: int = 3,
    num_perm: int = 128,
    bands: int = 16,
    threshold: float = 0.8,
    seed: int = 42,
) -> NearDupResult:
    """Cluster texts whose shingle-set Jaccard similarity is >= ``threshold``.

    Exact (whitespace/case-normalised) duplicates share identical shingle sets
    and are always clustered; they are handled without MinHash.
    """
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    sets = [shingles(t, shingle_size) for t in texts]
    uf = UnionFind(len(sets))
    pairs: list[tuple[int, int, float]] = []

    # Identical shingle sets: union directly, then MinHash one representative each.
    first_of: dict[frozenset[str], int] = {}
    reps: list[int] = []
    for i, s in enumerate(sets):
        j = first_of.setdefault(s, i)
        if j == i:
            reps.append(i)
        else:
            uf.union(i, j)
            pairs.append((j, i, 1.0))

    hasher = MinHasher(num_perm, seed)
    sigs = np.stack([hasher.signature(sets[i]) for i in reps]) if reps else np.empty((0, num_perm))
    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = sigs[:, band * rows : (band + 1) * rows]
        for k, row in enumerate(block):
            buckets[row.tobytes()].append(reps[k])
        for members in buckets.values():
            if 1 < len(members) <= _MAX_BUCKET:
                candidates.update(combinations(members, 2))
            elif len(members) > _MAX_BUCKET:
                logger.warning("skipping oversized LSH bucket", extra={"size": len(members)})

    for i, j in sorted(candidates):
        sim = jaccard(sets[i], sets[j])
        if sim >= threshold:
            uf.union(i, j)
            pairs.append((i, j, sim))

    result = NearDupResult(cluster=uf.labels(), pairs=pairs)
    logger.info(
        "near-duplicate detection done",
        extra={
            "texts": len(sets),
            "candidates": len(candidates),
            "verified_pairs": len(pairs),
            "multi_clusters": result.n_clusters_multi,
        },
    )
    return result
