from uuid import UUID
from fastapi import HTTPException
from app.core.Filter import Filter
from app.core.Chunk import EMBEDDING_DIM, Chunk
from app.core.Library import Library
from app.services.vector_store import VectorStore
from app.api.dto.Library import DeleteChunksDto, LibraryListItem, LibraryCreate, LibraryResponse, QueryDto, UpsertChunksDto
from app.utils.filters import passes_filter

# use this vector store to save everything in
vector_store = VectorStore()

def list_libraries_service():
    libraries = vector_store.get_all_libraries()
    return [LibraryListItem.model_validate(lib) for lib in libraries]

def get_library_by_id_service(lib_id: str):
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

def create_library_service(libraryData: LibraryCreate):
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

def delete_library_service(lib_id: str):
    if (not vector_store.has_library(UUID(lib_id))):
        raise HTTPException(status_code=404, detail="Library not found")
    vector_store.delete_library(UUID(lib_id))
    return {"status": "ok"}

def library_exists_service(lib_id: str):
    try:
        vector_store.get_library(UUID(lib_id))
        return {"exists": True}
    except KeyError:
        return {"exists": False}

def get_chunks_by_library_service(lib_id: str):
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = vector_store.get_library(UUID(lib_id))
    chunks = [chunk.model_dump() for chunk in library.get_all_chunks()]
    return chunks

def upsert_chunks_service(upsertChunksDto: UpsertChunksDto, lib_id: str):
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")
    # deserialize chunks from DTO and validate;
    hydrated_chunks = [
        Chunk.model_validate(obj=chunk) for chunk in upsertChunksDto.chunks
    ]

    # if any filters were sent, only a subset of chunks will be upserted, so validate them
    if (upsertChunksDto.filters):
        filters = Filter(root=upsertChunksDto.filters)
        hydrated_chunks = [
            chunk for chunk in hydrated_chunks if passes_filter(chunk.metadata, filters)
        ]
    vector_store.upsert_chunks(UUID(lib_id), hydrated_chunks)
    # return all the chunks that were upserted;
    return [chunk.model_dump() for chunk in hydrated_chunks]

def delete_chunks_by_library_service(deleteChunksDto: DeleteChunksDto, lib_id: str):
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")
    library = vector_store.get_library(UUID(lib_id))
    
    # we wither delete all or a filtered subset; similar implementation to upsert
    if not deleteChunksDto.filters:
        library.delete_chunks()
        return {"deleted": "all"}
    filters = Filter(root=deleteChunksDto.filters)
    filtered_chunks = [
        chunk for chunk in library.get_all_chunks()
        if passes_filter(chunk.metadata, filters)
    ]
    chunk_ids_to_delete = [chunk.id for chunk in filtered_chunks]
    if chunk_ids_to_delete:
        library.delete_chunks(chunk_ids_to_delete)
    return {"deleted": len(chunk_ids_to_delete)}

def count_chunks_by_library_service(lib_id: str):
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")
    library = vector_store.get_library(UUID(lib_id))
    return {"count": len(library.get_all_chunks())}

def search_chunks_by_library_service(lib_id: str, queryDto: QueryDto, k: int = 5):
    if (len(queryDto.query) != EMBEDDING_DIM):
        raise HTTPException(
            status_code=400, detail=f"Query vector must be of length {EMBEDDING_DIM}")
    if not vector_store.has_library(UUID(lib_id)):
        raise HTTPException(status_code=404, detail="Library not found")
    results = vector_store.search(
        UUID(lib_id), queryDto.query, k=k
    )
    return results
