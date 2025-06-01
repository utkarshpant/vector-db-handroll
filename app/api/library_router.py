from uuid import UUID, uuid4
from fastapi import APIRouter, HTTPException
from app.core.Filter import Filter
from app.core.Chunk import EMBEDDING_DIM, Chunk
from app.core.Library import Library
from app.services.VectorStore import VectorStore
from app.services.LibraryService import (
    list_libraries_service,
    get_library_by_id_service,
    create_library_service,
    delete_library_service,
    library_exists_service,
    get_chunks_by_library_service,
    upsert_chunks_service,
    delete_chunks_by_library_service,
    count_chunks_by_library_service,
    search_chunks_by_library_service,
)

# DTOs for different operations on a Library
from app.api.dto.Library import DeleteChunksDto, LibraryListItem, LibraryCreate, LibraryResponse, QueryDto, UpsertChunksDto

router = APIRouter()


@router.get("/", response_model=list[LibraryListItem])
async def list_libraries():
    """
    Index route handler for `/library`. List all libraries in the vector store.
    """
    return await list_libraries_service()


@router.get("/{lib_id}", response_model=LibraryResponse)
async def get_library_by_id(lib_id: str):
    """
    Get a specific def by its ID.
    """
    return await get_library_by_id_service(lib_id)


@router.post('/', response_model=LibraryResponse)
async def createLibrary(libraryData: LibraryCreate):
    """
    Create a new library with the given name.
    """
    return await create_library_service(libraryData)


@router.delete("/{lib_id}")
async def delete_library(lib_id: str):
    """
    Delete a library by its ID.
    """
    return await delete_library_service(lib_id)


@router.get("/{lib_id}/exists")
async def library_exists(lib_id: str):
    """
    Check if a library exists by its ID.
    """
    return await library_exists_service(lib_id)


@router.get("/{lib_id}/chunks", response_model=list[Chunk])
async def get_chunks_by_library(lib_id: str):
    """
    Get all chunks in a library by its ID.
    """
    return await get_chunks_by_library_service(lib_id)


@router.put("/{lib_id}/chunks", response_model=list[Chunk])
async def upsert_chunks(upsertChunksDto: UpsertChunksDto, lib_id: str):
    """
    Upsert chunks into a library by its ID.
    """
    return await upsert_chunks_service(upsertChunksDto, lib_id)


@router.post("/{lib_id}/chunks/delete")
async def delete_chunks_by_library(deleteChunksDto: DeleteChunksDto, lib_id: str):
    """
    Delete all chunks in a library by its ID, optionally delete only those that match a filter.
    """
    return await delete_chunks_by_library_service(deleteChunksDto, lib_id)


@router.get("/{lib_id}/count")
async def count_chunks_by_library(lib_id: str):
    """
    Count the number of chunks in a library by its ID.
    """
    return await count_chunks_by_library_service(lib_id)


@router.post("/{lib_id}/search")
async def search_chunks_by_library(lib_id: str, queryDto: QueryDto, k: int = 5):
    """
    Search for chunks in a library by its ID using a query string. Optionally specify `k`, the number of results to return await (default is 5).
    """
    return await search_chunks_by_library_service(lib_id, queryDto, k)
