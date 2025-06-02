import numpy as np
import pytest
import os
import textwrap
from typing import List
from uuid import uuid4
from app.services.VectorStore import VectorStore
from app.indexes.BallTreeIndex import BallTreeIndex
from app.indexes.BruteForceIndex import BruteForceIndex
from app.core.Chunk import EMBEDDING_DIM, Chunk
import cohere
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PARA_SPLIT   = "\n" # paragraph split on \n

cohere_client = cohere.ClientV2(COHERE_API_KEY) if COHERE_API_KEY else None

def embed_cohere(texts: list[str]) -> list[List[float]]:
    """Batch embed; returns list[np.ndarray]."""
    if not cohere_client:
        raise RuntimeError("COHERE_API_KEY not set or Cohere client not initialized.")
    res = cohere_client.embed(
        texts=texts,
        model="embed-v4.0",
        embedding_types=["float"],
        output_dimension=EMBEDDING_DIM,
        input_type="search_query"
    )
    # Cohere returns a .embeddings attribute with .float_ as a list of lists
    if not hasattr(res.embeddings, 'float_') or res.embeddings.float_ is None:
        raise RuntimeError("Cohere did not return embeddings. Check your API key and quota.")
    return [np.asarray(e, dtype=np.float32).tolist() for e in res.embeddings.float_]


@pytest.mark.integration
def test_end_to_end_cohere():
    if not COHERE_API_KEY:
        pytest.skip("COHERE_API_KEY not set")

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

    lib_a = store.create_library("A", index_name="BallTreeIndex")
    lib_a_obj = store.get_library(lib_a)

    paragraphs_a = doc_text_a.split(PARA_SPLIT)
    embeds_a = embed_cohere(paragraphs_a)

    lib_a_obj.upsert_chunks([Chunk(embedding=emb, metadata={"text": text}) for text, emb in zip(paragraphs_a, embeds_a)])
    store.build_index(lib_a, BallTreeIndex)

    lib_b = store.create_library("B", index_name="BruteForceIndex")
    lib_b_obj = store.get_library(lib_b)

    paragraphs_b = doc_text_b.split(PARA_SPLIT)
    embeds_b = embed_cohere(paragraphs_b)

    lib_b_obj.upsert_chunks([Chunk(embedding=emb, metadata={"text": text}) for text, emb in zip(paragraphs_b, embeds_b)])
    store.build_index(lib_b, BruteForceIndex)

    query = "Pack how many liquor jugs?"
    q_emb  = embed_cohere([query])[0]

    hits_a = store.search(lib_a, q_emb, k=2)
    hits_b = store.search(lib_b, q_emb, k=2)

    top_chunk_a, sim_a = hits_a[0]
    top_chunk_b, sim_b = hits_b[0]
    assert "five dozen" in top_chunk_b.metadata['text'] and sim_b > sim_a
    # isolation: no chunk from B in A’s results
    ids_b = {chunk.id for chunk, _ in hits_b}
    assert all(chunk.id not in ids_b for chunk, _ in hits_a)