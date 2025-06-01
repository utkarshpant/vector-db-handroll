from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import threading
import pickle
import aiofiles
import asyncio
import os

from app.api.dto.Library import Chunk
from app.core.Chunk import Chunk
from app.core.Library import Library
from app.indexes.BallTreeIndex import BallTreeIndex
from app.indexes.BaseIndex import BaseIndex
from app.indexes.BruteForceIndex import BruteForceIndex


class VectorStore:
    """
    A simple in-memory vector store that manages multiple `Libraries` and exposes a CRUD API to interact with them.
    """
    SNAPSHOT_PATH = os.getenv('SNAPSHOT_PATH') or './vectorstore_snapshot.pkl'
    SNAPSHOT_INTERVAL = 10  # seconds

    def __init__(self, index_factory=BruteForceIndex):
        self._libraries: Dict[UUID, Library] = {}
        self._index_factory = index_factory
        # per-library helper: id -> Chunk (populated when index is (re)built)
        self._chunk_lookup: Dict[UUID, Dict[UUID, Chunk]] = {}
        self._snapshot_lock = asyncio.Lock()
    
    @classmethod
    async def create(cls, index_factory=BruteForceIndex):
        store = cls(index_factory)
        await store.load_from_disk_async()
        store._start_snapshot_thread()
        return store

    def create_library(self, name: str, index: BallTreeIndex | BruteForceIndex = BruteForceIndex(), metadata: dict | None = None) -> UUID:
        lib = Library(name=name, metadata=metadata or {})
        lib.build_index(index)
        self._libraries[lib.id] = lib
        return lib.id

    def get_library(self, lib_id: UUID) -> Library:
        if lib_id not in self._libraries:
            raise KeyError(f"Library with ID {lib_id} does not exist.")
        return self._libraries[lib_id]

    def upsert_chunks(self, library_id: UUID, chunks: List[Chunk]) -> None:
        if library_id not in self._libraries:
            raise KeyError(f"Library with ID {library_id} does not exist.")
        library = self._libraries[library_id]
        library.upsert_chunks(chunks)

        # rebuild index and chunk lookup after upsert
        self.build_index(library_id)

    def get_all_chunks(self, lib_id: UUID) -> List[Chunk]:
        """
        Return all chunks in the library as a list.
        """
        if lib_id not in self._libraries:
            raise KeyError(f"Library with ID {lib_id} does not exist.")
        return self._libraries[lib_id].chunks

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
            for chunk in lib.chunks
        }

    def search(
        self, lib_id: UUID, query_vec: List[float], k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Return [(Chunk, similarity)] sorted by similarity desc.
        """
        hits = self._libraries[lib_id].search(query_vec, k)
        lookup = self._chunk_lookup.get(lib_id)  # populated by build_index()
        if lookup is None:
            raise RuntimeError("Index has not been built for this library")
        return [(lookup[cid], score) for cid, score in hits]

    async def save_to_disk_async(self):
        async with self._snapshot_lock:
            async with aiofiles.open(self.SNAPSHOT_PATH + '.tmp', 'wb') as f:
                await f.write(pickle.dumps({
                    'libraries': self._libraries,
                    'chunk_lookup': self._chunk_lookup
                }))
            os.replace(self.SNAPSHOT_PATH + '.tmp', self.SNAPSHOT_PATH)

    async def load_from_disk_async(self):
        if os.path.exists(self.SNAPSHOT_PATH):
            async with self._snapshot_lock:
                async with aiofiles.open(self.SNAPSHOT_PATH, 'rb') as f:
                    data = pickle.loads(f)
                    self._libraries = data.get('libraries', {})
                    self._chunk_lookup = data.get('chunk_lookup', {})

    def _start_snapshot_thread(self):
        async def snapshot_loop():
            while True:
                await asyncio.sleep(self.SNAPSHOT_INTERVAL)
                await self.save_to_disk_async()
        asyncio.create_task(snapshot_loop())
