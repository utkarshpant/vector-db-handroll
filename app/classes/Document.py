from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, List, Optional
from uuid import uuid4, UUID

from pydantic import BaseModel, ConfigDict, Field, validator
from .Chunk import Chunk

class Document(BaseModel):
    """
    A Document is made out of multiple Chunks; it also contains metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the Document"
    )
    title: str = Field(
        ...,
        description="The title of the document."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for the Document"
    )
    chunks: List[Chunk] = Field(
        default_factory=list,
        description="Ordered list of Chunks belonging to this Document"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the Document was created"
    )

    @validator("chunks", pre=True)
    def _ensure_chunk_list(cls, v):
        """
        Allow initializing from list of dicts or list of Chunk instances.
        """
        if not isinstance(v, list):
            raise ValueError("`chunks` must be a list of Chunk instances")
        return v

    def add_chunk(self, chunk: Chunk) -> None:
        """
        Append a new Chunk to this document.
        """
        # Example invariant: no duplicate chunk IDs
        if any(c.id == chunk.id for c in self.chunks):
            raise ValueError(f"Chunk with id={chunk.id} already exists in document")
        self.chunks.append(chunk)

    def remove_chunk(self, chunk_id: UUID) -> None:
        """
        Remove the Chunk with the given ID.
        """
        original_len = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.id != chunk_id]
        if len(self.chunks) == original_len:
            raise KeyError(f"No chunk with id={chunk_id} found")

    def to_dict(self) -> dict[str, Any]:
        """
        JSON-serialisable dict: flattens each chunk via its to_dict().
        """
        return {
            "id": str(self.id),
            "title": self.title,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "chunks": [c.to_dict() for c in self.chunks],
        }