import numpy as np
from uuid import uuid4
from pytest import approx
from ..classes.BruteForceIndex import BruteForceIndex

def test_bruteforce_index_basic():
    ids   = [uuid4() for _ in range(3)]
    vecs  = [np.array([1, 0, 0], dtype=np.float32),
             np.array([0, 1, 0], dtype=np.float32),
             np.array([0, 0, 1], dtype=np.float32)]

    ix = BruteForceIndex(normalize=False)
    ix.build(vecs, ids)

    q = np.array([0.9, 0.1, 0], dtype=np.float32)
    top = ix.search(q, k=2)

    # first neighbour should be the first vector (cos ~0.9)
    # TODO: examine the approx tolerance again
    assert top[0][0] == ids[0] and approx(top[0][1], rel=1e-1) == 0.9
    # second neighbour should be the second vector (cos ~0.1)
    assert top[1][0] == ids[1]
