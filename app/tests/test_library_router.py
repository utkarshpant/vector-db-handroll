import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from uuid import UUID, uuid4
from datetime import datetime
from app.main import app
from app.core.Chunk import Chunk
from app.core.Document import Document
from app.core.Library import Library

client = TestClient(app)


@pytest.fixture
def mock_vector_store():
    with patch('app.api.library_router.vector_store') as mock:
        yield mock


@pytest.fixture
def sample_library_id():
    return str(uuid4())


@pytest.fixture
def sample_library(sample_library_id):
    return Library(
        id=UUID(sample_library_id),
        name="Test Library",
        metadata={"description": "Test library for unit tests"},
        created_at=datetime.now(),
        documents=[]
    )


@pytest.fixture
def sample_library_with_documents(sample_library_id):
    doc = Document(
        id=uuid4(),
        title="Test document content",
        metadata={},
        chunks=[
            Chunk(
                id=uuid4(),
                metadata={
                    "text": "Chunk 1 content",
                    "created_at": datetime.now().isoformat()
                },
                embedding=np.random.rand(1536).tolist()  # Example embedding
            ),
            Chunk(
                id=uuid4(),
                metadata={
                    "text": "Chunk 2 content",
                    "created_at": datetime.now().isoformat()
                },
                embedding=np.random.rand(1536).tolist()
            )
        ]
    )

    return Library(
        id=UUID(sample_library_id),
        name="Test Library with Documents",
        metadata={"description": "Test library with documents"},
        created_at=datetime.now(),
        documents=[doc]
    )


def test_list_libraries(mock_vector_store, sample_library):
    # Setup
    mock_vector_store.get_all_libraries.return_value = [sample_library]

    # Execute
    response = client.get("/library/")

    # Assert
    assert response.status_code == 200
    libraries = response.json()
    assert len(libraries) == 1
    assert libraries[0]["name"] == "Test Library"
    assert libraries[0]["id"] == str(sample_library.id)


def test_get_library_by_id_success(mock_vector_store, sample_library_id, sample_library):
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library

    # Execute
    response = client.get(f"/library/{sample_library_id}")

    # Assert
    assert response.status_code == 200
    lib = response.json()
    assert lib["name"] == "Test Library"
    assert lib["id"] == sample_library_id
    assert lib["total_documents"] == 0
    assert lib["total_chunks"] == 0


def test_get_library_by_id_not_found(mock_vector_store, sample_library_id):
    # Setup
    mock_vector_store.has_library.return_value = False

    # Execute
    response = client.get(f"/library/{sample_library_id}")

    # Assert
    assert response.status_code == 404
    assert "Library not found" in response.json()["detail"]


def test_create_library(mock_vector_store, sample_library_id, sample_library):
    # Setup
    mock_vector_store.create_library.return_value = UUID(sample_library_id)
    mock_vector_store.get_library.return_value = sample_library

    # Execute
    response = client.post("/library/", json={
        "name": "Test Library",
        "metadata": {"description": "Test library for unit tests"}
    })

    # Assert
    assert response.status_code == 200
    lib = response.json()
    assert lib["name"] == "Test Library"
    assert lib["id"] == sample_library_id
    mock_vector_store.create_library.assert_called_once_with(
        "Test Library", {"description": "Test library for unit tests"}
    )


def test_delete_library_success(mock_vector_store, sample_library_id):
    # Setup
    mock_vector_store.has_library.return_value = True

    # Execute
    response = client.delete(f"/library/{sample_library_id}")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_vector_store.delete_library.assert_called_once_with(
        UUID(sample_library_id))


def test_delete_library_not_found(mock_vector_store, sample_library_id):
    # Setup
    mock_vector_store.has_library.return_value = False

    # Execute
    response = client.delete(f"/library/{sample_library_id}")

    # Assert
    assert response.status_code == 404
    assert "Library not found" in response.json()["detail"]


def test_library_exists_true(mock_vector_store, sample_library_id, sample_library):
    # Setup
    mock_vector_store.get_library.return_value = sample_library

    # Execute
    response = client.get(f"/library/{sample_library_id}/exists")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"exists": True}


def test_library_exists_false(mock_vector_store, sample_library_id):
    # Setup
    mock_vector_store.get_library.side_effect = KeyError()

    # Execute
    response = client.get(f"/library/{sample_library_id}/exists")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"exists": False}


def test_get_chunks_by_library_success(mock_vector_store, sample_library_id, sample_library_with_documents):
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_documents

    # Execute
    response = client.get(f"/library/{sample_library_id}/chunks")

    # Assert
    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) == 2
    assert chunks[0]['metadata']['text'] == "Chunk 1 content"
    assert chunks[1]['metadata']["text"] == "Chunk 2 content"


def test_get_chunks_by_library_not_found(mock_vector_store, sample_library_id):
    # Setup
    mock_vector_store.has_library.return_value = False

    # Execute
    response = client.get(f"/library/{sample_library_id}/chunks")

    # Assert
    assert response.status_code == 404
    assert "Library not found" in response.json()["detail"]


def test_upsert_chunks_success(mock_vector_store, sample_library_id, sample_library_with_documents):
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_documents

    new_chunks = [
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "created_at": datetime.now().isoformat(), "text": "test chunk"},
            "embedding": np.random.rand(1536).tolist()  # Example embedding
        }
    ]
    # Execute
    response = client.put(
        f"/library/{sample_library_id}/chunks", json={"chunks": new_chunks})
    # Assert
    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) == 2  # The original chunks from the sample library
    mock_vector_store.upsert_chunks.assert_called_once()
    call_args = mock_vector_store.upsert_chunks.call_args

    # Check the arguments
    assert call_args[0][0] == UUID(sample_library_id)  # lib_id
    assert call_args[0][1] is None  # document_id
    assert len(call_args[0][2]) == 1  # chunks list
    # First chunk is a Chunk object
    assert isinstance(call_args[0][2][0], Chunk)
    assert call_args[0][2][0].metadata['text'] == "test chunk"


def test_upsert_chunks_library_not_found(mock_vector_store, sample_library_id):
    # Setup
    mock_vector_store.has_library.return_value = False

    new_chunks = [
        {
            "id": str(uuid4()),
            "metadata": {
                "source": "test",
                "created_at": datetime.now().isoformat(),
                "text": "test chunk"
            },
            # Example embedding
            "embedding": np.random.rand(1536).tolist()
        }
    ]

    # Execute
    response = client.put(
        f"/library/{sample_library_id}/chunks", json={"chunks": new_chunks})

    # Assert
    assert response.status_code == 404
    assert "Library not found" in response.json()["detail"]
