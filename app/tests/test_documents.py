import numpy as np
import pytest
from uuid import UUID, uuid4

from app.core.Document import Document
from app.core.Chunk import Chunk, EMBEDDING_DIM

def create_test_chunk(text="test", dim=EMBEDDING_DIM):
    emb = np.ones((dim,), dtype=np.float32)
    return Chunk(text=text, embedding=emb, metadata={"foo": "bar"})

def test_document_creation_default():
    doc = Document(title="My Doc")
    assert doc.id is not None
    assert doc.title == "My Doc"
    assert doc.metadata == {}
    assert doc.chunks == []
    assert hasattr(doc, "created_at")

def test_add_chunk():
    doc = Document(title="Doc A")
    chunk = create_test_chunk()
    doc.add_chunk(chunk)
    assert len(doc.chunks) == 1
    assert doc.chunks[0].id == chunk.id

def test_add_duplicate_chunk_raises():
    doc = Document(title="Doc B")
    chunk = create_test_chunk()
    doc.add_chunk(chunk)
    with pytest.raises(ValueError) as exc:
        doc.add_chunk(chunk)
    assert "already exists" in str(exc.value)

def test_remove_chunk():
    doc = Document(title="Doc C")
    chunk1 = create_test_chunk(text="c1")
    chunk2 = create_test_chunk(text="c2")
    doc.add_chunk(chunk1)
    doc.add_chunk(chunk2)
    doc.remove_chunk(chunk1.id)
    assert len(doc.chunks) == 1
    assert doc.chunks[0].id == chunk2.id

def test_remove_nonexistent_chunk_raises():
    doc = Document(title="Doc D")
    nonexistent_id = uuid4()
    with pytest.raises(KeyError) as exc:
        doc.remove_chunk(nonexistent_id)
    assert "No chunk with id" in str(exc.value)

# def test_get_all_vectors_returns_correct_list():
#     doc = Document(title="Doc E")
#     chunks = [create_test_chunk(text=f"c{i}") for i in range(3)]
#     for ch in chunks:
#         doc.add_chunk(ch)
#     vectors = doc.get_all_vectors()
#     assert isinstance(vectors, list)
#     assert len(vectors) == 3
#     for vec in vectors:
#         assert isinstance(vec, np.ndarray)
#         assert vec.shape == (EMBEDDING_DIM,)

def test_to_dict_serializes_correctly():
    doc = Document(title="Doc F", metadata={"author": "alice"})
    chunk = create_test_chunk(text="hello")
    doc.add_chunk(chunk)
    d = doc.to_dict()
    assert d["id"] == str(doc.id)
    assert d["title"] == "Doc F"
    assert d["metadata"] == {"author": "alice"}
    assert isinstance(d["created_at"], str)
    assert isinstance(d["chunks"], list)
    assert d["chunks"][0]["text"] == "hello"