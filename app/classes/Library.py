# app/domain/library.py

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple
from uuid import uuid4, UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .Document import Document
from .Chunk import Chunk
from ..classes.BaseIndex import BaseIndex


class Library(BaseModel):
    """
    Aggregate root: owns Documents and a vector-index instance.
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
    documents: List[Document] = Field(
        default_factory=list,
        description="Documents contained in this Library"
    )
    index: Optional[BaseIndex] = Field(
        default=None,
        description="In-memory vector index for this Library"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the library was created"
    )

    @field_validator("documents", mode='before')
    def _ensure_documents_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("`documents` must be a list of Document instances")
        return v

    def add_document(self, doc: Document) -> UUID:
        """Add a new Document to this Library."""
        if any(d.id == doc.id for d in self.documents):
            raise ValueError(f"Document with id={doc.id} already exists")
        self.documents.append(doc)
        return doc.id

    def remove_document(self, doc_id: UUID) -> None:
        """Remove a Document and all its Chunks."""
        original_len = len(self.documents)
        self.documents = [d for d in self.documents if d.id != doc_id]
        if len(self.documents) == original_len:
            raise KeyError(f"No document with id={doc_id} found")

    def add_chunk(
        self,
        chunk: Chunk,
        document_id: Optional[UUID] = None
    ) -> None:
        """
        Add a Chunk into a specific Document (or into a default 'root' doc).
        """
        if document_id:
            # find the document
            for d in self.documents:
                if d.id == document_id:
                    d.add_chunk(chunk)
                    break
            else:
                raise KeyError(f"No document with id={document_id} found")
        else:
            # if no doc specified, create a default one
            if not self.documents:
                default_doc = Document(title="__default__")
                self.documents.append(default_doc)
            self.documents[-1].add_chunk(chunk)

    def remove_chunk(self, chunk_id: UUID) -> None:
        """
        Remove the given Chunk from whichever Document it lives in.
        """
        for d in self.documents:
            try:
                d.remove_chunk(chunk_id)
                return
            except KeyError:
                continue
        raise KeyError(f"No chunk with id={chunk_id} in any document")

    def get_all_embeddings(self) -> List[np.ndarray]:
        """Flatten all chunk embeddings in document order."""
        vectors: List[np.ndarray] = []
        for d in self.documents:
            vectors.extend(d.get_all_vectors())
        return vectors

    def build_index(self, index: BaseIndex) -> None:
        """
        (Re)build the in-memory index for this Library.
        """
        all_vectors = self.get_all_embeddings()
        all_ids = [chunk.id for d in self.documents for chunk in d.chunks]
        index.build(all_vectors, all_ids)
        self.index = index

    def search(
        self,
        query_vector: np.ndarray,
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
            "created_at": self.created_at.isoformat(),
            "documents": [d.to_dict() for d in self.documents],
        }
