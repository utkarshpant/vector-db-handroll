# tests/test_ball_tree.py
import numpy as np
import pytest
from uuid import uuid4

from ..indexes.BallTreeIndex import BallTreeIndex
from ..core.Chunk import EMBEDDING_DIM


# ---------------------------------------------------------------- helpers
def _make_dataset(n: int = 200, d: int = EMBEDDING_DIM):
    """
    Generate n random unit-length vectors and parallel UUID list.
    """
    vecs = np.random.randn(n, d).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [uuid4() for _ in range(n)]
    return vecs, ids


# ---------------------------------------------------------------- tests
def test_build_and_simple_query():
    """
    • Build succeeds.
    • Query returns correct nearest neighbour (matches brute force).
    """
    vecs, ids = _make_dataset()
    bt = BallTreeIndex(leaf_size=8)
    bt.build(list(vecs), ids)

    # choose a query close to the first point
    q = vecs[0] + 0.05
    brute_cos = vecs @ q / (np.linalg.norm(q))
    brute_nn = ids[int(brute_cos.argmax())]
    top = bt.search(q, k=1)
    assert top[0][0] == brute_nn
    # returned similarity should be >= brute-force best minus tiny epsilon
    assert pytest.approx(top[0][1], rel=1e-6) == float(brute_cos.max())


def test_k_larger_than_dataset():
    """
    Asking for more neighbours than points should gracefully return all points.
    """
    vecs, ids = _make_dataset(n=10, d=32)
    bt = BallTreeIndex()
    bt.build(list(vecs), ids)

    results = bt.search(vecs[0], k=20)   # k > n
    assert len(results) == 10            # not 20


def test_empty_index_behaviour():
    """
    Building with an empty list should not crash; searching should raise.
    """
    bt = BallTreeIndex()
    bt.build([], [])
    with pytest.raises(RuntimeError):
        bt.search(np.zeros(8, dtype=np.float32), k=1)
