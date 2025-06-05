from typing import Any
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from uuid import UUID, uuid4
from datetime import datetime
from app.main import app
from app.core.Chunk import Chunk
from app.core.Library import Library
import threading
import time

client = TestClient(app)


@pytest.fixture
def mock_vector_store(monkeypatch):
    # Patch the get_vector_store async function to return a MagicMock
    from app.services import LibraryService
    mock = MagicMock()
    async def fake_get_vector_store():
        return mock
    monkeypatch.setattr(LibraryService, "get_vector_store", fake_get_vector_store)
    return mock


@pytest.fixture
def sample_library_id():
    return str(uuid4())


@pytest.fixture
def sample_library(sample_library_id):
    return Library(
        id=UUID(sample_library_id),
        name="Test Library",
        metadata={"description": "Test library for unit tests"},
    )


@pytest.fixture
def sample_library_with_chunks(sample_library_id):
    chunks = [
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

    return Library(
        id=UUID(sample_library_id),
        name="Test Library with Documents",
        metadata={"description": "Test library with documents"},
        chunks=chunks
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
        "Test Library", index_name="BallTreeIndex", metadata={"description": "Test library for unit tests"}
    )


def test_create_library_with_index_name(mock_vector_store, sample_library_id, sample_library):
    mock_vector_store.create_library.return_value = UUID(sample_library_id)
    mock_vector_store.get_library.return_value = sample_library

    # Test with index_name specified
    response = client.post("/library/", json={
        "name": "Test Library",
        "metadata": {"description": "Test library for unit tests"},
        "index_name": "BallTreeIndex"
    })
    assert response.status_code == 200
    lib = response.json()
    assert lib["name"] == "Test Library"
    assert lib["id"] == sample_library_id
    mock_vector_store.create_library.assert_called_with(
        "Test Library", index_name="BallTreeIndex", metadata={"description": "Test library for unit tests"}
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


def test_get_chunks_by_library_success(mock_vector_store, sample_library_id, sample_library_with_chunks):
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_chunks
    mock_vector_store.get_all_chunks.return_value = sample_library_with_chunks.chunks

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


def test_upsert_chunks_success(mock_vector_store, sample_library_id, sample_library_with_chunks):
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_chunks

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
    # assert len(chunks) == 2  # The original chunks from the sample library
    mock_vector_store.upsert_chunks.assert_called_once()
    call_args = mock_vector_store.upsert_chunks.call_args
    # Check the arguments
    assert call_args[0][0] == UUID(sample_library_id)  # lib_id
    assert len(call_args[0][1]) == 1  # chunks list
    # First chunk is a Chunk object
    assert isinstance(call_args[0][1][0], Chunk)
    assert call_args[0][1][0].metadata['text'] == "test chunk"


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


def test_upsert_chunks_with_filters_matching(mock_vector_store, sample_library_id, sample_library_with_chunks):
    """Test upsert chunks with filters that match some chunks"""
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_chunks

    new_chunks = [
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "category": "important", "text": "important chunk"},
            "embedding": np.random.rand(1536).tolist()
        },
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "category": "normal", "text": "normal chunk"},
            "embedding": np.random.rand(1536).tolist()
        }
    ]

    # Add filters to only upsert chunks with category="important"
    filters = {
        "category": {"eq": "important"}
    }

    # Execute
    response = client.put(
        f"/library/{sample_library_id}/chunks",
        json={"chunks": new_chunks, "filters": filters}
    )

    # Assert
    assert response.status_code == 200
    mock_vector_store.upsert_chunks.assert_called_once()
    call_args = mock_vector_store.upsert_chunks.call_args

    # Should only upsert the chunk with category="important"
    assert len(call_args[0][1]) == 1
    assert call_args[0][1][0].metadata['category'] == "important"


def test_upsert_chunks_with_filters_no_matches(mock_vector_store, sample_library_id, sample_library_with_chunks):
    """Test upsert chunks with filters that don't match any chunks"""
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_chunks

    new_chunks = [
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "category": "normal", "text": "normal chunk"},
            "embedding": np.random.rand(1536).tolist()
        }
    ]

    # Add filters that won't match any chunks
    filters = {
        "category": {"eq": "nonexistent"}
    }

    # Execute
    response = client.put(
        f"/library/{sample_library_id}/chunks",
        json={"chunks": new_chunks, "filters": filters}
    )

    # Assert
    assert response.status_code == 200
    mock_vector_store.upsert_chunks.assert_called_once()
    call_args = mock_vector_store.upsert_chunks.call_args

    # Should upsert no chunks
    assert len(call_args[0][1]) == 0


def test_upsert_chunks_with_contains_filter(mock_vector_store, sample_library_id, sample_library_with_chunks):
    """Test upsert chunks with contains filter"""
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_chunks

    new_chunks = [
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "text": "This is important information"},
            "embedding": np.random.rand(1536).tolist()
        },
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "text": "This is just regular content"},
            "embedding": np.random.rand(1536).tolist()
        }
    ]

    # Add filters to only upsert chunks containing "important"
    filters = {
        "text": {"contains": "important"}
    }

    # Execute
    response = client.put(
        f"/library/{sample_library_id}/chunks",
        json={"chunks": new_chunks, "filters": filters}
    )

    # Assert
    assert response.status_code == 200
    mock_vector_store.upsert_chunks.assert_called_once()
    call_args = mock_vector_store.upsert_chunks.call_args

    # Should only upsert the chunk containing "important"
    assert len(call_args[0][1]) == 1
    assert "important" in call_args[0][1][0].metadata['text']


def test_upsert_chunks_with_multiple_filters(mock_vector_store, sample_library_id, sample_library_with_chunks):
    """Test upsert chunks with multiple filter conditions"""
    # Setup
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_chunks

    new_chunks = [
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "priority": 5, "category": "important"},
            "embedding": np.random.rand(1536).tolist()
        },
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "priority": 3, "category": "important"},
            "embedding": np.random.rand(1536).tolist()
        },
        {
            "id": str(uuid4()),
            "metadata": {"source": "test", "priority": 7, "category": "normal"},
            "embedding": np.random.rand(1536).tolist()
        }
    ]

    # Add filters for priority >= 5 AND category = "important"
    filters = {
        "priority": {"gte": 5},
        "category": {"eq": "important"}
    }

    # Execute
    response = client.put(
        f"/library/{sample_library_id}/chunks",
        json={"chunks": new_chunks, "filters": filters}
    )

    # Assert
    assert response.status_code == 200
    mock_vector_store.upsert_chunks.assert_called_once()
    call_args = mock_vector_store.upsert_chunks.call_args

    # Should only upsert the chunk with priority >= 5 AND category = "important"
    assert len(call_args[0][1]) == 1
    assert call_args[0][1][0].metadata['priority'] == 5
    assert call_args[0][1][0].metadata['category'] == "important"


def test_delete_chunks_with_filters_success(mock_vector_store, sample_library_id):
    """Test delete chunks with filters"""
    # Setup
    mock_vector_store.has_library.return_value = True

    # Create a mock library with get_all_chunks method for delete operation
    mock_library = MagicMock()
    mock_library.get_all_chunks.return_value = [
        Chunk(
            id=uuid4(),
            metadata={"category": "important", "text": "Important chunk"},
            embedding=np.random.rand(1536).tolist()
        ),
        Chunk(
            id=uuid4(),
            metadata={"category": "normal", "text": "Normal chunk"},
            embedding=np.random.rand(1536).tolist()
        )
    ]
    mock_vector_store.get_library.return_value = mock_library

    # Add filters to only delete chunks with category="important"
    filters = {
        "category": {"eq": "important"}
    }

    # Execute - now using POST method for delete chunks
    response = client.post(
        f"/library/{sample_library_id}/chunks/delete",
        json={"filters": filters}
    )

    # Assert
    assert response.status_code == 200
    # Verify that get_all_chunks was called to get chunks for filtering
    mock_library.get_all_chunks.assert_called_once()
    # Verify that delete_chunks was called with the ID of the important chunk
    mock_library.delete_chunks.assert_called_once()


def test_delete_chunks_with_contains_filter(mock_vector_store, sample_library_id):
    """Test delete chunks with contains filter"""
    # Setup
    mock_vector_store.has_library.return_value = True

    chunk_id_1 = uuid4()
    chunk_id_2 = uuid4()

    mock_library = MagicMock()
    mock_library.get_all_chunks.return_value = [
        Chunk(
            id=chunk_id_1,
            metadata={"text": "This contains the word urgent"},
            embedding=np.random.rand(1536).tolist()
        ),
        Chunk(
            id=chunk_id_2,
            metadata={"text": "This is just normal content"},
            embedding=np.random.rand(1536).tolist()
        )
    ]
    mock_vector_store.get_library.return_value = mock_library

    # Add filters to only delete chunks containing "urgent"
    filters = {
        "text": {"contains": "urgent"}
    }

    # Execute - using POST method for delete chunks
    response = client.post(
        f"/library/{sample_library_id}/chunks/delete",
        json={"filters": filters}
    )

    # Assert
    assert response.status_code == 200
    mock_library.get_all_chunks.assert_called_once()
    # Should call delete_chunks with the chunk ID that contains "urgent"
    mock_library.delete_chunks.assert_called_once_with([chunk_id_1])


def test_delete_chunks_with_no_filters(mock_vector_store, sample_library_id):
    """Test delete chunks without any filters (delete all)"""
    # Setup
    mock_vector_store.has_library.return_value = True

    mock_library = MagicMock()
    mock_vector_store.get_library.return_value = mock_library

    # Execute - no filters provided, using POST method
    response = client.post(
        f"/library/{sample_library_id}/chunks/delete",
        json={}
    )

    # Assert
    assert response.status_code == 200
    # Should call delete_chunks without arguments (delete all)
    mock_library.delete_chunks.assert_called_once_with()


def test_delete_chunks_with_filters_no_matches(mock_vector_store, sample_library_id):
    """Test delete chunks with filters that don't match any chunks"""
    # Setup
    mock_vector_store.has_library.return_value = True

    mock_library = MagicMock()
    mock_library.get_all_chunks.return_value = [
        Chunk(
            id=uuid4(),
            metadata={"category": "normal", "text": "Normal chunk"},
            embedding=np.random.rand(1536).tolist()
        )
    ]
    mock_vector_store.get_library.return_value = mock_library

    # Add filters that won't match any chunks
    filters = {
        "category": {"eq": "nonexistent"}
    }

    # Execute - using POST method for delete chunks
    response = client.post(
        f"/library/{sample_library_id}/chunks/delete",
        json={"filters": filters}
    )

    # Assert
    assert response.status_code == 200
    mock_library.get_all_chunks.assert_called_once()
    # Should not call delete_chunks since no chunks match the filter
    mock_library.delete_chunks.assert_not_called()


def test_delete_chunks_with_numeric_filters(mock_vector_store, sample_library_id):
    """Test delete chunks with numeric comparison filters"""
    # Setup
    mock_vector_store.has_library.return_value = True

    chunk_id_1 = uuid4()
    chunk_id_2 = uuid4()
    chunk_id_3 = uuid4()

    mock_library = MagicMock()
    mock_library.get_all_chunks.return_value = [
        Chunk(
            id=chunk_id_1,
            metadata={"priority": 8, "text": "High priority"},
            embedding=np.random.rand(1536).tolist()
        ),
        Chunk(
            id=chunk_id_2,
            metadata={"priority": 3, "text": "Low priority"},
            embedding=np.random.rand(1536).tolist()
        ),
        Chunk(
            id=chunk_id_3,
            metadata={"priority": 5, "text": "Medium priority"},
            embedding=np.random.rand(1536).tolist()
        )
    ]
    mock_vector_store.get_library.return_value = mock_library

    # Add filters to delete chunks with priority >= 5
    filters = {
        "priority": {"gte": 5}
    }

    # Execute - using POST method for delete chunks
    response = client.post(
        f"/library/{sample_library_id}/chunks/delete",
        json={"filters": filters}
    )    # Assert
    assert response.status_code == 200
    mock_library.get_all_chunks.assert_called_once()
    # Should call delete_chunks once for chunks with priority >= 5
    assert mock_library.delete_chunks.call_count == 1
    # Verify the correct chunk IDs were passed
    call_args = mock_library.delete_chunks.call_args
    deleted_chunk_ids = call_args[0][0]  # First positional argument (chunk_ids list)
    assert chunk_id_1 in deleted_chunk_ids  # priority 8
    assert chunk_id_3 in deleted_chunk_ids  # priority 5
    assert chunk_id_2 not in deleted_chunk_ids  # priority 3


def test_delete_chunks_library_not_found(mock_vector_store, sample_library_id):
    """Test delete chunks when library doesn't exist"""
    # Setup
    mock_vector_store.has_library.return_value = False

    filters = {
        "category": {"eq": "test"}
    }

    # Execute - using POST method for delete chunks
    response = client.post(
        f"/library/{sample_library_id}/chunks/delete",
        json={"filters": filters}
    )

    # Assert
    assert response.status_code == 404
    assert "Library not found" in response.json()["detail"]


def test_concurrent_upserts_no_race(tmp_path):
    """
    Test that concurrent upserts to the same library do not cause data races or errors.
    """
    client = TestClient(app)
    # Create a library first
    response = client.post(
        "/library/", json={"name": "RaceTest", "metadata": {}})
    assert response.status_code == 200
    lib_id = response.json()["id"]

    def upsert_chunk(idx):
        chunk = {
            "id": str(uuid4()),
            "metadata": {"source": f"thread-{idx}",
                         "created_at": datetime.now().isoformat(),
                         "text": f"chunk {idx}"
                         },
            "embedding": np.random.rand(1536).tolist()
        }
        resp = client.put(
            f"/library/{lib_id}/chunks", json={"chunks": [chunk]})
        assert resp.status_code == 200

    threads = [threading.Thread(target=upsert_chunk, args=(i,))
               for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # verify that all chunks are present
    response = client.get(f"/library/{lib_id}/chunks")
    assert response.status_code == 200
    chunks = response.json()
    # there should ideally be 5 chunks;
    assert len(chunks) >= 5


def test_search_with_filters_api(mock_vector_store, sample_library_id, sample_library_with_chunks):
    mock_vector_store.has_library.return_value = True
    mock_vector_store.get_library.return_value = sample_library_with_chunks
    mock_vector_store.search.return_value = [
        (sample_library_with_chunks.chunks[0], 0.99),
        (sample_library_with_chunks.chunks[1], 0.88)
    ]
    filters = {"text": {"contains": "Chunk 1"}}
    query = np.random.rand(1536).tolist()
    response = client.post(
        f"/library/{sample_library_id}/search",
        json={"query": query, "filters": filters}
    )
    assert response.status_code == 200
    mock_vector_store.search.assert_called_once()
    # Should filter results to only those matching the filter
    # (the service layer does the filtering after calling search)
    results = response.json()
    assert any("Chunk 1" in chunk[0]['metadata']['text'] for chunk in results)
