from uuid import UUID, uuid4
from fastapi import APIRouter, HTTPException
from app.core.Filter import Filter
from app.core.Chunk import Chunk
from app.core.Library import Library
from app.services.vector_store import VectorStore

# DTOs for different operations on a Library
from app.api.dto.Library import DeleteChunksDto, LibraryListItem, LibraryCreate, LibraryResponse, UpsertChunksDto
from app.utils.filters import passes_filter

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
        total_chunks=len(library.chunks)
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
        total_chunks=len(library.chunks)
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


@router.get("/{lib_id}/chunks", response_model=list[Chunk])
def get_chunks_by_library(lib_id: str):
    """
    Get all chunks in a library by its ID.
    """
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")

    library = vector_store.get_library(UUID(lib_id))
    chunks = [chunk.model_dump() for chunk in library.get_all_chunks()]
    return chunks


@router.put("/{lib_id}/chunks", response_model=list[Chunk])
def upsert_chunks(upsertChunksDto: UpsertChunksDto, lib_id: str):
    """
    Upsert chunks into a library by its ID.
    """
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")

    hydrated_chunks = [
        Chunk.model_validate(obj=chunk) for chunk in upsertChunksDto.chunks
    ]
    # if filters are present, only update those Chunks that match the filter criteria
    if (upsertChunksDto.filters):
        filters = Filter(root=upsertChunksDto.filters)
        print(filters)
        hydrated_chunks = [
            chunk for chunk in hydrated_chunks if passes_filter(chunk.metadata, filters)
        ]
    vector_store.upsert_chunks(UUID(lib_id), hydrated_chunks)
    return [chunk.model_dump() for chunk in hydrated_chunks]


@router.post("/{lib_id}/chunks/delete")
def delete_chunks_by_library(deleteChunksDto: DeleteChunksDto, lib_id: str):
    """
    Delete all chunks in a library by its ID, optionally delete only those that match a filter.
    """
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")

    library = vector_store.get_library(UUID(lib_id))

    # If no filters, delete all chunks
    if not deleteChunksDto.filters:
        library.delete_chunks()
        return {"deleted": "all"}

    # Otherwise, delete only chunks matching the filter
    filters = Filter(root=deleteChunksDto.filters)
    filtered_chunks = [
        chunk for chunk in library.get_all_chunks()
        if passes_filter(chunk.metadata, filters)
    ]
    chunk_ids_to_delete = [chunk.id for chunk in filtered_chunks]
    if chunk_ids_to_delete:
        library.delete_chunks(chunk_ids_to_delete)
    return {"deleted": len(chunk_ids_to_delete)}
