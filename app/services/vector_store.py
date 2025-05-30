from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from app.core.Chunk import Chunk
from app.core.Document import Document
from app.core.Library import Library
from app.indexes.BaseIndex import BaseIndex
from app.indexes.BruteForceIndex import BruteForceIndex


class VectorStore:
    """
    A simple in-memory vector store that manages multiple `Libraries` and exposes a CRUD API to interact with them.
    """

    def __init__(self, index_factory=BruteForceIndex):
        self._libraries: Dict[UUID, Library] = {}
        self._index_factory = index_factory
        # per-library helper: id -> Chunk (populated when index is (re)built)
        self._chunk_lookup: Dict[UUID, Dict[UUID, Chunk]] = {}

    def create_library(self, name: str, metadata: dict | None = None) -> UUID:
        lib = Library(name=name, metadata=metadata or {})
        self._libraries[lib.id] = lib
        return lib.id

    def get_library(self, lib_id: UUID) -> Library:
        if lib_id not in self._libraries:
            raise KeyError(f"Library with ID {lib_id} does not exist.")
        return self._libraries[lib_id]

    def upsert_chunks(self, library_id: UUID, document_id: Optional[UUID], chunks: List[Chunk]) -> None:
        if library_id not in self._libraries:
            raise KeyError(f"Library with ID {library_id} does not exist.")
        library = self._libraries[library_id]
        library.add_chunks(chunks, document_id)

        # Rebuild index and chunk lookup after upsert
        self.build_index(library_id)

    def delete_library(self, lib_id: UUID) -> None:
        if lib_id not in self._libraries:
            raise KeyError(f"Library with ID {lib_id} does not exist.")
        self._libraries.pop(lib_id)
        self._chunk_lookup.pop(lib_id, None)

    def get_all_libraries(self) -> Tuple[Library, ...]:
        """
        Return *tuples* of all libraries in the vector store. Tuples are returned to ensure immutability.
        """
        return tuple(self._libraries.values())

    def has_library(self, lib_id: UUID) -> bool:
        """
        Check if a library with the given ID exists in the vector store.
        """
        return lib_id in self._libraries

    def build_index(self, lib_id: UUID, index_cls: type[BaseIndex] | None = None) -> None:
        """
        (Re)build the index for one library and refresh its chunk-lookup table.
        """
        lib = self._libraries[lib_id]
        index = (index_cls or self._index_factory)()
        lib.build_index(index)

        # rebuild quick lookup
        self._chunk_lookup[lib_id] = {
            chunk.id: chunk
            for doc in lib.documents
            for chunk in doc.chunks
        }

    def search(
        self, lib_id: UUID, query_vec: np.ndarray, k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Return [(Chunk, similarity)] sorted by similarity desc.
        """
        hits = self._libraries[lib_id].search(query_vec, k)
        lookup = self._chunk_lookup.get(lib_id)  # populated by build_index()
        if lookup is None:
            raise RuntimeError("Index has not been built for this library")

        return [(lookup[cid], score) for cid, score in hits]
