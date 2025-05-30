from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4, UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np

EMBEDDING_DIM = 1536


class Chunk(BaseModel):
    """
    A Chunk is a piece of text with an associated embedding and metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: UUID = Field(default_factory=uuid4,
                     description="Unique identifier for the Chunk")
    text: str = Field(..., description="The raw text of the Chunk")
    embedding: np.ndarray = Field(default_factory=lambda: np.zeros(
        (EMBEDDING_DIM,), dtype=np.float32), description="The embedding vector of the Chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the Chunk"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the chunk was created",
    )

    @field_validator('embedding')
    def _validate_embedding(cls, v: np.ndarray) -> np.ndarray:
        """
        Validate that the embedding is a numpy array of the correct shape.
        """
        if not isinstance(v, np.ndarray):
            raise ValueError("Embedding must be a numpy array")
        if v.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Embedding must have shape ({EMBEDDING_DIM},)")
        return v

    @property
    def vector(self) -> np.ndarray:
        """Return the embedding as an immutable `np.ndarray`."""
        return np.asarray(self.embedding, dtype=np.float32)

    def cosine_similarity(self, other_vector: np.ndarray) -> float:
        """Cosine similarity between this chunk and `other_vector`."""
        a = self.embedding.astype(np.float32, copy=False)
        b = other_vector.astype(np.float32, copy=False)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def to_dict(self) -> dict[str, Any]:
        """JSON‑serialisable representation (numpy arrays → list)."""
        return self.model_dump() | {"embedding": self.embedding}


class SerializableChunk(Chunk):
    """
    A serializable version of Chunk for use in APIs.
    """
    id: UUID = Field(default_factory=uuid4,
                     description="Unique identifier for the Chunk")
    document_id: Optional[UUID] = Field(
        None, description="ID of the Document this Chunk belongs to")
    text: str = Field(..., description="The text content of the Chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with the Chunk")
    embedding: Optional[List[float]] = Field(
        [i for i in range(EMBEDDING_DIM)], description="Vector embeddings as a list of floats")


    @field_validator('embeddings')
    def validate_embeddings(cls, v):
        # Validate that embeddings are provided as a list of floats if present
        if v is not None and not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embeddings must be a list of float values")
        return v

    class Config:
        from_attributes = True
