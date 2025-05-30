# app/domain/library.py

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Union
from uuid import uuid4, UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.indexes.BallTreeIndex import BallTreeIndex
from app.indexes.BruteForceIndex import BruteForceIndex

from .Chunk import EMBEDDING_DIM, Chunk
from ..indexes.BaseIndex import BaseIndex


class Library(BaseModel):
    """
    Aggregate root: owns Chunks a vector-index instance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this Library"
    )
    name: str = Field(
        ...,
        description="Human-readable name of the library"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for the Library"
    )
    chunks: List[Chunk] = Field(
        default_factory=list,
        description="Ordered list of Chunks belonging to this Library"
    )
    index: Optional[BaseIndex | BruteForceIndex | BallTreeIndex] = Field(
        default=BruteForceIndex(),
        description="In-memory vector index for this Library"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the library was created"
    )

    def upsert_chunks(self, chunks_to_upsert: List[Chunk]) -> None:
        """
        Upsert (insert or update) Chunks in the Library's chunk list.
        If a chunk with the same ID exists, it is replaced; otherwise, it is appended.
        """
        if not chunks_to_upsert:
            return
        if not all(len(chunk.embedding) == EMBEDDING_DIM for chunk in chunks_to_upsert):
            raise ValueError(f"All chunks must have {EMBEDDING_DIM} dimensions")
        
        # Build a mapping from chunk ID to index for efficient lookup
        id_to_index = {chunk.id: idx for idx, chunk in enumerate(self.chunks)}
        for chunk in chunks_to_upsert:
            if chunk.id in id_to_index:
                self.chunks[id_to_index[chunk.id]] = chunk
            else:
                self.chunks.append(chunk)
        
        # Rebuild index
        if self.index is not None:
            self.build_index(self.index.__class__())
        else:
            self.build_index(BallTreeIndex())
        

    def delete_chunks(self, chunk_ids: List[UUID] | None = None) -> None:
        """
        Unified method to delete Chunks by ID, list of IDs, or all Chunks.
        """
        if chunk_ids is None:
            self.chunks.clear()
        elif isinstance(chunk_ids, list):
            self.chunks = [chunk for chunk in self.chunks if chunk.id not in chunk_ids]
        self.build_index(self.index.__class__() if self.index else BallTreeIndex())

    def get_all_chunks(self) -> Tuple[Chunk, ...]:
        """Get all chunks in this Library as immutable tuples."""
        return tuple(self.chunks)
        
    # TODO: use delete_chunks and pass an array with a single ID
    # def _remove_chunk(self, chunk_id: UUID) -> None:
    #     """
    #     Remove the given Chunk from whichever Document it lives in.
    #     """
    #     for d in self.documents:
    #         try:
    #             d.remove_chunk(chunk_id)
    #             return
    #         except KeyError:
    #             continue
    #     raise KeyError(f"No chunk with id={chunk_id} in any document")

    # def _delete_all_chunks(self) -> None:
    #     """
    #     Remove all Chunks from all Documents in this Library.
    #     """
    #     for document in self.documents:
    #         document.chunks.clear()
    #     self.index = None

    def build_index(self, index: BaseIndex) -> None:
        """
        (Re)build the in-memory index for this Library.
        """
        all_chunks = self.get_all_chunks()
        all_embeddings = [chunk.embedding for chunk in all_chunks]
        all_ids = [chunk.id for chunk in all_chunks]
        index.build(all_embeddings, all_ids)
        self.index = index

    def search(
        self,
        query_vector: List[float],
        k: int
    ) -> List[Tuple[UUID, float]]:
        """
        k-NN search over the built index. Raises if index is None.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built yet")
        return self.index.search(query_vector, k)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation of the Library."""
        return {
            "id": str(self.id),
            "name": self.name,
            "metadata": self.metadata,
            # "created_at": self.created_at.isoformat(),
            # "documents": [d.to_dict() for d in self.documents],
        }
