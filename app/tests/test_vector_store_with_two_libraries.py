# tests/test_vector_store_two_libs.py
import numpy as np
import pytest
from uuid import uuid4

from ..classes.vector_store import VectorStore
from ..classes.BruteForceIndex import BruteForceIndex
from ..classes.BallTreeIndex import BallTreeIndex
from ..classes.Document import Document
from ..classes.Chunk import Chunk

@pytest.fixture(scope="session")
def fake_embed():
    """
    Deterministic embedding generator. Hashes text into a PRNG seed.
    """
    def _embed(text: str, dim: int = 1536) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        # small noise to avoid identical norms
        vec = rng.standard_normal(dim).astype(np.float32)
        return vec / np.linalg.norm(vec)
    return _embed

@pytest.fixture
def vector_store():
    return VectorStore()

def _populate_library(store, lib_id, embed, doc_prefix):
    """
    3 docs with 2 chunks each = 6 chunks in total
    Returns a dict; `{chunk_id: text}`.
    """
    lib = store.get_library(lib_id)
    chunk_map = {}
    for doc in range(3):
        lib.add_document(Document(title=f"{doc_prefix}_doc_{d}"))
        for chunk in range(2):
            text = f"{doc_prefix}_doc_{d}_chunk_{c}"
            chunk = Chunk(
                text=text,
                embedding=embed(text),
                metadata={"role": doc_prefix},
            )
            chunk_id = lib.add_chunk(chunk)
            chunk_map[chunk_id] = text
    return chunk_map

# parameters for this test: brute force index and ball tree index
@pytest.mark.parametrize("index_cls", [BruteForceIndex, BallTreeIndex])
def test_two_libraries_isolated(vector_store, fake_embed, index_cls):
    """
    • Build one library with `index_cls`, another with the default.
    • Ensure search returns only chunks from the queried lib.
    """
    # lib A uses the injected index_cls
    lib_a = vector_store.create_library("A")
    chunks_A = _populate_library(vector_store, lib_a, fake_embed, "A")
    vector_store.build_index(lib_a, index_cls=index_cls)

    lib_b = vector_store.create_library("B")
    chunks_B = _populate_library(vector_store, lib_b, fake_embed, "B")
    vector_store.build_index(lib_b)
    print(chunks_A, chunks_B, "chunks A and B populated");
    target_text = next(iter(chunks_A.values()))
    # perturb the target text slightly;
    query_vec = fake_embed(target_text) + 0.01

    results = vector_store.search(lib_a, query_vec, k=4)

    # all results that match should be from library A
    result_ids = {chunk.id for chunk, _ in results}
    assert result_ids.issubset(set(chunks_A.keys()))
    assert result_ids.isdisjoint(set(chunks_B.keys()))

    # First hit should be the chunk we perturbed
    top_chunk, score = results[0]
    assert chunks_A[top_chunk.id] == target_text
    assert score > 0.9
