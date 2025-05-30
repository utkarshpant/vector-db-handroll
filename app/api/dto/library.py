from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.core.Chunk import Chunk
from app.core.Filter import Condition

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
    total_chunks: int = Field(..., description="Total number of chunks in the Library")
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True

# class MinimalChunk(BaseModel):
#     """
#     A minimal, serializable version of Chunk that has only text, embedding and metadata information.
#     """
#     id: UUID = Field(default_factory=uuid4, description="ID for the chunk")
#     # text: str = Field(..., description="Text corresponding to the Chunk")
#     embedding: List[float] = Field(default_factory=lambda: [0.0] * 1536, description="Embedding vector of the Chunk")
#     metadata: Dict[str, Any] = Field(default_factory=lambda: {
#         "created_at": datetime.now().isoformat(),
#         "text": ""
#     }, description="Metadata associated with the Chunk")

class UpsertChunksDto(BaseModel):
    """
    Data Transfer Object (DTO) for upserting chunks into a library.
    """
    chunks: list[Chunk] = Field(..., description="List of chunks to be upserted, each chunk is a dict with its properties")
    filters: Optional[Dict[str, Condition]] = Field(
        None, description="Optional filters to apply when upserting chunks"
    )
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True

class DeleteChunksDto(BaseModel):
    """
    Data Transfer Object (DTO) for deleting chunks from a library.
    """
    filters: Optional[Dict[str, Condition]] = Field(
        None, description="Optional filters to apply when deleting chunks"
    )
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True