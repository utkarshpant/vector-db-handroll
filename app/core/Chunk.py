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
    # text: str = Field(..., description="The raw text of the Chunk")
    embedding: List[float] = Field(default_factory=lambda: [0 for i in range(EMBEDDING_DIM)], description="The embedding vector of the Chunk")
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text": "",
        },
        description="Additional metadata associated with the Chunk"
    )
    # created_at: datetime = Field(
    #     default_factory=lambda: datetime.now(timezone.utc),
    #     description="UTC timestamp when the chunk was created",
    # )

    @field_validator('embedding')
    def _validate_embedding(cls, v: List[float]) -> List[float]:
        """
        Validate that the embedding is a numpy array of the correct shape.
        """
        if len(v) != EMBEDDING_DIM:
            raise ValueError(f"Embedding must have shape ({EMBEDDING_DIM},)")
        return v

    @property
    def vector(self) -> List[float]:
        """Return the embedding as an immutable tuple."""
        return (self.embedding)

    @property
    def created_at(self) -> str | None:
        """Return the creation timestamp from metadata."""
        if "created_at" not in self.metadata:
            return None
        return self.metadata.get("created_at")


    def cosine_similarity(self, other_vector: List[float]) -> float:
        """Cosine similarity between this chunk and `other_vector`."""
        a = self.embedding
        b = other_vector
        return float(np.dot(self.embedding, other_vector) / (np.linalg.norm(self.embedding) * np.linalg.norm(other_vector)))

    def to_dict(self) -> dict[str, Any]:
        """JSON‑serialisable representation (numpy arrays → list)."""
        return self.model_dump() | {"embedding": self.embedding}
