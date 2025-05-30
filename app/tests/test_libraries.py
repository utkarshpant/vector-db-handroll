import pytest
import numpy as np
from uuid import uuid4

from ..core.Chunk import Chunk, EMBEDDING_DIM
from ..core.Library import Library
from ..indexes.BruteForceIndex import BruteForceIndex

def make_chunk(fill: float = 1.0) -> Chunk:
    """Return a simple 1536-D vector filled with `fill`."""
    vec = np.full((EMBEDDING_DIM,), fill, dtype=np.float32).tolist()
    return Chunk(embedding=vec, metadata={"text": f"text {fill}"})

def test_library_creation_defaults():
    lib = Library(name="My Lib")
    assert lib.id and lib.name == "My Lib"
    assert lib.metadata == {}


def test_remove_chunk():
    lib = Library(name="Remove chunk")
    ch1, ch2 = make_chunk(1.0), make_chunk(2.0)
    lib.upsert_chunks([ch1, ch2])

    lib.delete_chunks([ch1.id])
    assert len(lib.get_all_chunks()) == 1

def test_build_index_and_search():
    lib = Library(name="Index search", index=BruteForceIndex(normalize=False))
    
    v1 = np.zeros((EMBEDDING_DIM,), dtype=np.float32).tolist()
    v1[0] = 1.0
    v2 = np.zeros((EMBEDDING_DIM,), dtype=np.float32).tolist()
    v2[1] = 1.0

    c1 = Chunk(embedding=v1, metadata={"text": "x"})
    c2 = Chunk(embedding=v2, metadata={"text": "y"})
    lib.upsert_chunks([c1, c2])

    ix = BruteForceIndex(normalize=False)
    lib.build_index(ix)

    # query near v1
    q = np.zeros((EMBEDDING_DIM,), dtype=np.float32).tolist()
    q[0] = 0.9
    q[1] = 0.1

    top = lib.search(q, k=1)
    assert top[0][0] == c1.id # nearest neighbor should be chunk 1
    # TODO: revisit this tolerance
    assert pytest.approx(top[0][1], rel=1e-1) == 0.9
