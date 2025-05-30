from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field
from typing import Any, Optional

from app.core.Chunk import Chunk, SerializableChunk

class LibraryListItem(BaseModel):
    """
    Response object sent back when all libraries are listed.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the Library")
    name: str = Field(..., description="Name of the Library")
    # metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the Library")
    # created_at: datetime = Field(..., description="UTC timestamp when the library was created")
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True

class LibraryCreate(BaseModel):
    """
    Data Transfer Object (DTO) for creating a library.
    """
    name: str = Field(..., description="Name of the Library")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Metadata associated with the Library")
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True

class LibraryResponse(BaseModel):
    """
    Response object sent back when a Library is created or a given Library is fetched.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the Library")
    name: str = Field(..., description="Name of the Library")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the Library")
    created_at: datetime = Field(..., description="UTC timestamp when the library was created")
    total_documents: int = Field(..., description="Total number of documents in the Library")
    total_chunks: int = Field(..., description="Total number of chunks in the Library")
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True

class UpsertChunksDto(BaseModel):
    """
    Data Transfer Object (DTO) for upserting chunks into a library.
    """
    library_id: UUID = Field(..., description="ID of the Library to which chunks will be added")
    document_id: Optional[UUID] = Field(None, description="ID of the Document to which chunks will be added")
    chunks: list[SerializableChunk] = Field(..., description="List of chunks to be upserted, each chunk is a dict with its properties")
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True