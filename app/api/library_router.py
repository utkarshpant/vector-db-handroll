from uuid import UUID
from fastapi import APIRouter, HTTPException
from app.core.Chunk import Chunk, SerializableChunk
from app.core.Document import Document
from app.core.Library import Library
from app.services.vector_store import VectorStore

# DTOs for different operations on a Library
from app.api.dto.Library import LibraryListItem, LibraryCreate, LibraryResponse, UpsertChunksDto

vector_store = VectorStore()
router = APIRouter()


@router.get("/", response_model=list[LibraryListItem])
def list_libraries():
    """
    Index route handler for `/library`. List all libraries in the vector store.
    """
    libraries = vector_store.get_all_libraries()
    return [LibraryListItem.model_validate(lib) for lib in libraries]


@router.get("/{lib_id}", response_model=LibraryResponse)
def get_library_by_id(lib_id: str):
    """
    Get a specific library by its ID.
    """
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")
    library = vector_store.get_library(UUID(lib_id))
    library = LibraryResponse(
        id=library.id,
        name=library.name,
        metadata=library.metadata,
        created_at=library.created_at,
        total_documents=len(library.documents),
        total_chunks=sum(len(doc.chunks) for doc in library.documents)
    )
    return LibraryResponse.model_validate(library)


@router.post('/', response_model=LibraryResponse)
def createLibrary(libraryData: LibraryCreate):
    """
    Create a new library with the given name.
    """
    lib_id = vector_store.create_library(
        libraryData.name, libraryData.metadata)
    library = vector_store.get_library(lib_id)
    library = LibraryResponse(
        id=library.id,
        name=library.name,
        metadata=library.metadata,
        created_at=library.created_at,
        total_documents=len(library.documents),
        total_chunks=sum(len(doc.chunks) for doc in library.documents)
    )
    return LibraryResponse.model_validate(library)


@router.delete("/{lib_id}")
def delete_library(lib_id: str):
    """
    Delete a library by its ID.
    """
    if (not vector_store.has_library(UUID(lib_id))):
        raise HTTPException(status_code=404, detail="Library not found")
    vector_store.delete_library(UUID(lib_id))
    return {"status": "ok"}


@router.get("/{lib_id}/exists")
def library_exists(lib_id: str):
    """
    Check if a library exists by its ID.
    """
    try:
        vector_store.get_library(UUID(lib_id))
        return {"exists": True}
    except KeyError:
        return {"exists": False}


@router.get("/{lib_id}/chunks", response_model=list[SerializableChunk])
def get_chunks_by_library(lib_id: str):
    """
    Get all chunks in a library by its ID.
    """
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")

    library = vector_store.get_library(UUID(lib_id))
    chunks = [chunk.model_dump()
              for doc in library.documents for chunk in doc.chunks]
    return chunks


@router.put("/{lib_id}/chunks", response_model=list[Chunk])
def upsert_chunks(upsertChunksDto: UpsertChunksDto, lib_id: str):
    """
    Upsert chunks into a library by its ID.
    """
    # print(upsertChunksDto.chunks)
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")

    vector_store.upsert_chunks(UUID(lib_id), None, upsertChunksDto.chunks)
    library = vector_store.get_library(UUID(lib_id))
    all_chunks = [chunk.model_dump()
                  for doc in library.documents for chunk in doc.chunks]
    return all_chunks
