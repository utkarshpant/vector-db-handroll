import os, time, textwrap, pytest, numpy as np
from typing import List
from app.utils.openai import client
from uuid import uuid4

from app.services.VectorStore import VectorStore
from app.indexes.BallTreeIndex import BallTreeIndex
from app.indexes.BruteForceIndex import BruteForceIndex
from app.core.Chunk import Chunk

OPENAI_MODEL = "text-embedding-3-small"
PARA_SPLIT   = "\n"                      # paragraph splitter


def embed_openai(texts: list[str]) -> list[List[float]]:
    """Batch embed; returns list[np.ndarray]."""
    res = client.embeddings.create(
        model=OPENAI_MODEL,
        input=texts,
        encoding_format="float"
    )
    return [np.asarray(e.embedding, dtype=np.float32).tolist() for e in res.data]


@pytest.mark.integration
def test_end_to_end_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    store = VectorStore()

    doc_text_a = textwrap.dedent("""
        The quick brown fox jumps over the lazy dog.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Vector databases enable efficient similarity search in high-dimensional space.
    """).strip()

    doc_text_b = textwrap.dedent("""
        Pack my box with five dozen liquor jugs.
        Natural-language embeddings map text to points in ℝᵈ.
        Ball-tree and KD-tree are classical space-partitioning indexes.
    """).strip()

    lib_a = store.create_library("A")
    lib_a_obj = store.get_library(lib_a)

    paragraphs_a = doc_text_a.split(PARA_SPLIT)
    embeds_a = embed_openai(paragraphs_a)

    lib_a_obj.upsert_chunks([Chunk(embedding=emb, metadata={"text": text}) for text, emb in zip(paragraphs_a, embeds_a)])
    store.build_index(lib_a, BallTreeIndex)

    lib_b = store.create_library("B")
    lib_b_obj = store.get_library(lib_b)

    paragraphs_b = doc_text_b.split(PARA_SPLIT)
    embeds_b = embed_openai(paragraphs_b)

    lib_b_obj.upsert_chunks([Chunk(embedding=emb, metadata={"text": text}) for text, emb in zip(paragraphs_b, embeds_b)])
    store.build_index(lib_b, BruteForceIndex)

    query = "Pack how many liquor jugs?"
    q_emb  = embed_openai([query])[0]

    hits_a = store.search(lib_a, q_emb, k=2)
    hits_b = store.search(lib_b, q_emb, k=2)

    top_chunk_a, sim_a = hits_a[0]
    top_chunk_b, sim_b = hits_b[0]
    assert "five dozen" in top_chunk_b.metadata['text'] and sim_b > sim_a
    # isolation: no chunk from B in A’s results
    ids_b = {chunk.id for chunk, _ in hits_b}
    assert all(chunk.id not in ids_b for chunk, _ in hits_a)