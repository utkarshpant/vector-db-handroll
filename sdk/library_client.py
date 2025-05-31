import requests
from typing import Any, Dict, List, Optional, Tuple

from app.api.dto.Library import UpsertChunksDto

class LibraryClient:
    """
    Python client for the Stack-AI Vector Database REST API.
    Provides methods to manage libraries and chunks, and to perform vector queries.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        :param base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")

    def list_libraries(self) -> List[Dict[str, Any]]:
        """
        List all libraries.
        :return: List of library metadata dicts.
        """
        resp = requests.get(f"{self.base_url}/library")
        resp.raise_for_status()
        return resp.json()

    def get_library(self, library_id: str) -> Dict[str, Any]:
        """
        Get details of a specific library.
        :param library_id: Library UUID
        """
        resp = requests.get(f"{self.base_url}/library/{library_id}")
        resp.raise_for_status()
        return resp.json()

    def create_library(self, name: str, metadata: Optional[dict] = None) -> Dict[str, Any]:
        """
        Create a new library.
        :param name: Name of the library
        :param metadata: Optional metadata dict
        :return: Created library info
        """
        data = {"name": name}
        if metadata:
            data["metadata"] = metadata
        resp = requests.post(f"{self.base_url}/library", json=data)
        resp.raise_for_status()
        return resp.json()

    def delete_library(self, library_id: str) -> None:
        """
        Delete a library by its ID.
        :param library_id: Library UUID
        """
        resp = requests.delete(f"{self.base_url}/library/{library_id}")
        resp.raise_for_status()

    def upsert_chunks(self, library_id: str, chunks: List[dict]) -> Dict[str, Any]:
        """
        Upsert (insert or update) chunks in a library.
        :param library_id: Library UUID
        :param chunks: List of chunk dicts
        :return: API response dict
        """
        data = {"chunks": chunks}
        resp = requests.put(f"{self.base_url}/library/{library_id}/chunks", json=data)
        resp.raise_for_status()
        return resp.json()

    def list_chunks(self, library_id: str) -> List[Dict[str, Any]]:
        """
        List all chunks in a library.
        :param library_id: Library UUID
        :return: List of chunk dicts
        """
        resp = requests.get(f"{self.base_url}/library/{library_id}/chunks")
        resp.raise_for_status()
        return resp.json()

    def count_chunks(self, library_id: str) -> int:
        """
        Get the number of chunks in a library.
        :param library_id: Library UUID
        :return: Number of chunks
        """
        resp = requests.get(f"{self.base_url}/library/{library_id}/count")
        resp.raise_for_status()
        return resp.json().get("count", 0)

    def search(self, library_id: str, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search the vector index for the top-k most similar chunks.
        :param library_id: Library UUID
        :param query_vector: List of floats (embedding)
        :param k: Number of results to return
        :return: List of (chunk_id, similarity) tuples
        """
        data = {"query": query_vector}
        resp = requests.post(f"{self.base_url}/library/{library_id}/search?k={k}", json=data)
        resp.raise_for_status()
        return resp.json()

    def library_exists(self, library_id: str) -> bool:
        """
        Check if a library exists by its ID.
        :param library_id: Library UUID
        :return: True if the library exists, False otherwise
        """
        resp = requests.get(f"{self.base_url}/library/{library_id}/exists")
        resp.raise_for_status()
        return resp.json().get("exists", False)
