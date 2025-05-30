import numpy as np
import pytest
from uuid import UUID

from ..core.Chunk import Chunk, EMBEDDING_DIM  # adjust the import path as necessary


def _random_embedding(dim: int = EMBEDDING_DIM):
    """Utility: create a deterministic random vector for reproducible tests."""
    rng = np.random.default_rng(seed=42)
    return rng.random(dim).astype(np.float32).tolist()


def test_chunk_valid_creation():
    """Chunk with correct embedding length should be created successfully."""
    chunk = Chunk(embedding=_random_embedding(), metadata={"text": "hello world"})
    assert chunk.metadata['text'] == "hello world"
    assert len(chunk.embedding) == EMBEDDING_DIM
    assert isinstance(chunk.id, UUID)


def test_chunk_invalid_embedding_length():
    """Creating a chunk with a wrong‑sized embedding must raise a ValueError."""
    short_emb = np.array([0.1, 0.2, 0.3]).tolist()  # obviously too short
    with pytest.raises(ValueError):
        Chunk(embedding=short_emb, metadata={"text": "invalid embedding"})


def test_cosine_similarity_static():
    """Static method should return 1.0 for identical vectors."""
    vec = _random_embedding()
    Chunk1 = Chunk(embedding=vec, metadata={"text": "vec1"})
    Chunk2 = Chunk(embedding=vec, metadata={"text": "vec2"})
    sim_identical = Chunk.cosine_similarity(Chunk1, Chunk2.embedding)
    
    assert pytest.approx(sim_identical, rel=1e-6) == 1.0
    