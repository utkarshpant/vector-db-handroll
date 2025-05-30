import pytest
import numpy as np
from uuid import uuid4

from ..core.Chunk import Chunk, EMBEDDING_DIM
from ..core.Document import Document
from ..core.Library import Library
from ..indexes.BruteForceIndex import BruteForceIndex

def make_chunk(fill: float = 1.0) -> Chunk:
    """Return a simple 1536-D vector filled with `fill`."""
    vec = np.full((EMBEDDING_DIM,), fill, dtype=np.float32).tolist()
    return Chunk(embedding=vec, metadata={"text": f"text {fill}"})

def make_document(title="doc") -> Document:
    return Document(title=title)

def test_library_creation_defaults():
    lib = Library(name="My Lib")
    assert lib.id and lib.name == "My Lib"
    assert lib.documents == []
    assert lib.metadata == {}

def test_add_and_remove_document():
    lib = Library(name="Adds docs")
    doc = make_document()
    lib.add_document(doc)
    assert len(lib.documents) == 1

    lib.remove_document(doc.id)
    assert lib.documents == []

    # removing again raises
    with pytest.raises(KeyError):
        lib.remove_document(doc.id)

def test_add_duplicate_document_raises():
    lib = Library(name="Dup docs")
    doc = make_document()
    lib.add_document(doc)
    with pytest.raises(ValueError):
        lib.add_document(doc)

def test_add_chunk_with_explicit_document():
    lib = Library(name="Chunks w doc")
    doc = make_document()
    lib.add_document(doc)

    chunk = make_chunk()
    lib.add_chunk(chunk, document_id=doc.id)

    assert len(lib.documents[0].chunks) == 1
    assert lib.documents[0].chunks[0].id == chunk.id

def test_add_chunk_without_document_creates_default():
    lib = Library(name="Chunks default doc")
    chunk = make_chunk()
    lib.add_chunk(chunk)

    assert len(lib.documents) == 1
    assert lib.documents[0].title == "__default__"
    assert lib.documents[0].chunks[0].id == chunk.id

def test_remove_chunk():
    lib = Library(name="Remove chunk")
    doc = make_document()
    lib.add_document(doc)
    ch1, ch2 = make_chunk(1.0), make_chunk(2.0)
    lib.add_chunk(ch1, doc.id)
    lib.add_chunk(ch2, doc.id)

    lib.remove_chunk(ch1.id)
    assert len(lib.documents[0].chunks) == 1
    assert lib.documents[0].chunks[0].id == ch2.id

    with pytest.raises(KeyError):
        lib.remove_chunk(ch1.id)

def test_build_index_and_search():
    lib = Library(name="Index search", index=BruteForceIndex(normalize=False))
    doc = make_document()
    lib.add_document(doc)

    v1 = np.zeros((EMBEDDING_DIM,), dtype=np.float32).tolist()
    v1[0] = 1.0
    v2 = np.zeros((EMBEDDING_DIM,), dtype=np.float32).tolist()
    v2[1] = 1.0

    c1 = Chunk(embedding=v1, metadata={"text": "x"})
    c2 = Chunk(embedding=v2, metadata={"text": "y"})
    lib.add_chunk(c1, doc.id)
    lib.add_chunk(c2, doc.id)

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
