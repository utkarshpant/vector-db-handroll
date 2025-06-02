import httpx
from typing import Any, Dict, List, Optional, Tuple

class LibraryClient:
    """
    Async Python client for the Stack Vector Database's REST API.
    Provides async methods to manage libraries and chunks, and to perform vector queries with optional filters.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        :param base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient()

    async def list_libraries(self) -> List[Dict[str, Any]]:
        """
        List all libraries.
        :return: List of library metadata dicts.
        """
        resp = await self._client.get(f"{self.base_url}/library/")
        resp.raise_for_status()
        return resp.json()

    async def get_library(self, library_id: str) -> Dict[str, Any]:
        """
        Get details of a specific library.
        :param library_id: Library UUID
        """
        resp = await self._client.get(f"{self.base_url}/library/{library_id}/")
        resp.raise_for_status()
        return resp.json()

    async def create_library(self, name: str, metadata: Optional[Dict[str, Any]] = None, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new library.
        :param name: Name of the library
        :param metadata: Optional metadata dict
        :param index_name: Optional index name (e.g., 'BruteForceIndex', 'BallTreeIndex')
        :return: Created library info
        """
        data: Dict[str, Any] = {"name": name}
        if metadata:
            data["metadata"] = metadata
        if index_name:
            data["index_name"] = index_name
        print(data, "data")
        resp = await self._client.post(f"{self.base_url}/library/", json=data)
        resp.raise_for_status()
        return resp.json()

    async def delete_library(self, library_id: str) -> None:
        """
        Delete a library by its ID.
        :param library_id: Library UUID
        """
        resp = await self._client.delete(f"{self.base_url}/library/{library_id}/")
        resp.raise_for_status()

    async def upsert_chunks(self, library_id: str, chunks: List[Dict[str, Any]], filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upsert (insert or update) chunks in a library, with optional filters.
        :param library_id: Library UUID
        :param chunks: List of chunk dicts
        :param filters: Optional filters dict
        :return: API response dict
        """
        data: Dict[str, Any] = {"chunks": chunks}
        if filters:
            data["filters"] = filters
        resp = await self._client.put(f"{self.base_url}/library/{library_id}/chunks", json=data)
        resp.raise_for_status()
        return resp.json()

    async def list_chunks(self, library_id: str) -> List[Dict[str, Any]]:
        """
        List all chunks in a library.
        :param library_id: Library UUID
        :return: List of chunk dicts
        """
        resp = await self._client.get(f"{self.base_url}/library/{library_id}/chunks")
        resp.raise_for_status()
        return resp.json()

    async def count_chunks(self, library_id: str) -> int:
        """
        Get the number of chunks in a library.
        :param library_id: Library UUID
        :return: Number of chunks
        """
        resp = await self._client.get(f"{self.base_url}/library/{library_id}/count")
        resp.raise_for_status()
        return resp.json().get("count", 0)

    async def search(self, library_id: str, query_vector: List[float], k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        Search the vector index for the top-k most similar chunks, with optional filters.
        :param library_id: Library UUID
        :param query_vector: List of floats (embedding)
        :param k: Number of results to return
        :param filters: Optional filters dict
        :return: List of (chunk_id, similarity) tuples
        """
        data: Dict[str, Any] = {"query": query_vector}
        if filters:
            data["filters"] = filters
        resp = await self._client.post(f"{self.base_url}/library/{library_id}/search?k={k}", json=data)
        resp.raise_for_status()
        return resp.json()

    async def library_exists(self, library_id: str) -> bool:
        """
        Check if a library exists by its ID.
        :param library_id: Library UUID
        :return: True if the library exists, False otherwise
        """
        resp = await self._client.get(f"{self.base_url}/library/{library_id}/exists")
        resp.raise_for_status()
        return resp.json().get("exists", False)

    async def aclose(self):
        """
        Close the HTTP client session. Useful to close the pool of connections after all work is done.
        """
        await self._client.aclose()
