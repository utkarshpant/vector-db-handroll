import numpy as np
import pytest
from uuid import UUID

from ..classes.Chunk import Chunk, EMBEDDING_DIM  # adjust the import path as necessary


def _random_embedding(dim: int = EMBEDDING_DIM):
    """Utility: create a deterministic random vector for reproducible tests."""
    rng = np.random.default_rng(seed=42)
    return rng.random(dim).astype(np.float32)


def test_chunk_valid_creation():
    """Chunk with correct embedding length should be created successfully."""
    chunk = Chunk(text="hello world", embedding=_random_embedding())
    assert chunk.text == "hello world"
    assert len(chunk.embedding) == EMBEDDING_DIM
    assert isinstance(chunk.id, UUID)


def test_chunk_invalid_embedding_length():
    """Creating a chunk with a wrong‑sized embedding must raise a ValueError."""
    short_emb = np.array([0.1, 0.2, 0.3])  # obviously too short
    with pytest.raises(ValueError):
        Chunk(text="bad", embedding=short_emb)


def test_embedding_np_property():
    """`embedding_np` should expose a numpy array of shape (dim,)."""
    chunk = Chunk(text="numeric", embedding=_random_embedding())
    arr = chunk.embedding
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (EMBEDDING_DIM,)


def test_cosine_similarity_static():
    """Static method should return 1.0 for identical vectors."""
    vec = _random_embedding()
    Chunk1 = Chunk(text="vec1", embedding=vec)
    Chunk2 = Chunk(text="vec2", embedding=vec)
    sim_identical = Chunk.cosine_similarity(Chunk1, Chunk2.embedding)
    
    assert pytest.approx(sim_identical, rel=1e-6) == 1.0
    