from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field
from typing import Any, Optional

class LibraryListItem(BaseModel):
    """
    Response object sent back when all libraries are listed.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the Library")
    name: str = Field(..., description="Name of the Library")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the Library")
    created_at: datetime = Field(..., description="UTC timestamp when the library was created")
    
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

# class LibraryCreateResponse(BaseModel):
#     """
#     Response object sent back when a Library is created.
#     """
#     id: UUID = Field(default_factory=uuid4, description="Unique identifier for the Library")
#     name: str = Field(..., description="Name of the Library")
#     metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the Library")
    
#     class Config:
#         from_attributes = True
#         arbitrary_types_allowed = True