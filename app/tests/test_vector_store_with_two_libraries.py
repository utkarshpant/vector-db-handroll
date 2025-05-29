import numpy as np
import pytest
from uuid import UUID
from typing import Callable

from app.services.vector_store import VectorStore
from app.indexes.BruteForceIndex import BruteForceIndex
from app.indexes.BallTreeIndex import BallTreeIndex
from app.core.Document import Document
from app.core.Chunk import Chunk


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


def _populate_library(store: VectorStore, lib_id: UUID, embed: Callable[[str, int], np.ndarray], doc_prefix):
    """
    3 docs with 2 chunks each = 6 chunks in total
    Returns a dict; `{chunk_id: text}`.
    """
    lib = store.get_library(lib_id)
    for d in range(3):
        doc = Document(title=f"{doc_prefix}_doc_{d}")
        for c in range(2):
            text = f"{doc_prefix}_doc_{d}_chunk_{c}"
            doc.add_chunk(Chunk(text=text, embedding=embed(text)))
        lib.add_document(doc)
    chunk_map = {chunk.id: chunk.text for doc in lib.get_all_documents()
                 for chunk in doc.get_all_chunks()}
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
    target_text = next(iter(chunks_A.values()))
    # perturb the target text slightly;
    query_vec = fake_embed(target_text) + 0.01

    results = vector_store.search(lib_a, query_vec, k=4)

    # all results that match should be from library A
    result_ids = {chunk.id for chunk, _ in results}
    assert result_ids.issubset(set(chunks_A.keys()))
    assert result_ids.isdisjoint(set(chunks_B.keys()))

    # first hit should be the chunk we perturbed
    top_chunk, score = results[0]
    assert chunks_A[top_chunk.id] == target_text
    assert score > 0.9
