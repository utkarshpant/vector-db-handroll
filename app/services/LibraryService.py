from uuid import UUID
from fastapi import HTTPException
from app.core.Filter import Filter
from app.core.Chunk import EMBEDDING_DIM, Chunk
from app.core.Library import Library
from app.indexes.BallTreeIndex import BallTreeIndex
from app.indexes.BruteForceIndex import BruteForceIndex
from app.services.VectorStore import VectorStore
from app.api.dto.Library import DeleteChunksDto, LibraryListItem, LibraryCreate, LibraryResponse, QueryDto, UpsertChunksDto
from app.utils.filters import passes_filter
from app.utils.read_write_lock import ReadWriteLock
# use this vector store to save everything in
rw_lock = ReadWriteLock()

async def get_vector_store():
    return await VectorStore.get_instance()

async def get_library_lock(lib_id: str):
    vector_store = await get_vector_store()
    return vector_store.get_library_lock(UUID(lib_id))

async def list_libraries_service():
    try:
        rw_lock.acquire_read()
        vector_store = await get_vector_store()
        libraries = vector_store.get_all_libraries()
        return [LibraryListItem.model_validate(lib) for lib in libraries]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        rw_lock.release_read()

async def get_library_by_id_service(lib_id: str):
    if not lib_id:
        raise HTTPException(status_code=400, detail="Library ID must be provided.")
    lock = await get_library_lock(lib_id)
    try:
        lock.acquire_read()
        vector_store = await get_vector_store()
        if not vector_store.has_library(UUID(lib_id)):
            raise HTTPException(status_code=404, detail="Library not found")
        library = vector_store.get_library(UUID(lib_id))
        library = LibraryResponse(
            id=library.id,
            name=library.name,
            metadata=library.metadata,
            total_chunks=len(library.chunks),
            index_name=library.index_name
        )
        return LibraryResponse.model_validate(library)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        lock.release_read()

async def create_library_service(libraryData: LibraryCreate):
    if not libraryData or not libraryData.name:
        raise HTTPException(status_code=422, detail="Library name is required.")
    rw_lock.acquire_write()
    try:
        vector_store = await get_vector_store()
        # Use the index_name if provided, else default
        # index_instance = None
        # index_name = None
        # if hasattr(libraryData, 'index_name') and libraryData.index_name:
        #     # Support both Enum and string
        #     if hasattr(libraryData.index_name, 'value'):
        #         index_name = libraryData.index_name.value
        #     else:
        #         index_name = str(libraryData.index_name)
        #     if index_name == LibraryCreate.IndexNameEnum.BallTreeIndex.value:
        #         index_instance = BallTreeIndex()
        #     else:
        #         index_instance = BruteForceIndex()
        # else:
        #     index_instance = BruteForceIndex()
        lib_id = vector_store.create_library(
            libraryData.name, index_name="BallTreeIndex", metadata=libraryData.metadata)
        library = vector_store.get_library(lib_id)
        library = LibraryResponse(
            id=library.id,
            name=library.name,
            metadata=library.metadata,
            total_chunks=len(library.chunks),
            index_name=library.index_name
        )
        return LibraryResponse.model_validate(library)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        rw_lock.release_write()

async def delete_library_service(lib_id: str):
    if not lib_id:
        raise HTTPException(status_code=400, detail="Library ID must be provided.")
    rw_lock.acquire_write()
    try:
        vector_store = await get_vector_store()
        if not vector_store.has_library(UUID(lib_id)):
            raise HTTPException(status_code=404, detail="Library not found")
        vector_store.delete_library(UUID(lib_id))
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        rw_lock.release_write()

async def library_exists_service(lib_id: str):
    if not lib_id:
        raise HTTPException(status_code=400, detail="Library ID must be provided.")
    rw_lock.acquire_read()
    try:
        vector_store = await get_vector_store()
        try:
            vector_store.get_library(UUID(lib_id))
            return {"exists": True}
        except KeyError:
            return {"exists": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        rw_lock.release_read()

async def get_chunks_by_library_service(lib_id: str):
    if not lib_id:
        raise HTTPException(status_code=400, detail="Library ID must be provided.")
    lock = await get_library_lock(lib_id)
    try:
        lock.acquire_read()
        vector_store = await get_vector_store()
        if not vector_store.has_library(UUID(lib_id)):
            raise HTTPException(status_code=404, detail="Library not found")
        library = vector_store.get_library(UUID(lib_id))
        chunks = [chunk.model_dump() for chunk in library.get_all_chunks()]
        return chunks
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        lock.release_read()

async def upsert_chunks_service(upsertChunksDto: UpsertChunksDto, lib_id: str):
    if not lib_id or not upsertChunksDto or not upsertChunksDto.chunks:
        raise HTTPException(status_code=422, detail="Library ID and chunks are required.")
    lock = await get_library_lock(lib_id)
    try:
        lock.acquire_write()
        vector_store = await get_vector_store()
        if not vector_store.has_library(UUID(lib_id)):
            raise HTTPException(status_code=404, detail="Library not found")
        hydrated_chunks = [
            Chunk.model_validate(obj=chunk) for chunk in upsertChunksDto.chunks
        ]
        filters = getattr(upsertChunksDto, 'filters', None)
        if filters:
            filter_obj = Filter(root=filters)
            hydrated_chunks = [
                chunk for chunk in hydrated_chunks if passes_filter(chunk.metadata, filter_obj)
            ]
        vector_store.upsert_chunks(UUID(lib_id), hydrated_chunks)
        return [chunk.model_dump() for chunk in hydrated_chunks]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        lock.release_write()

async def delete_chunks_by_library_service(deleteChunksDto: DeleteChunksDto, lib_id: str):
    if not lib_id:
        raise HTTPException(status_code=400, detail="Library ID must be provided.")
    lock = await get_library_lock(lib_id)
    try:
        lock.acquire_write()
        vector_store = await get_vector_store()
        if not vector_store.has_library(UUID(lib_id)):
            raise HTTPException(status_code=404, detail="Library not found")
        library = vector_store.get_library(UUID(lib_id))
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        lock.release_write()

async def count_chunks_by_library_service(lib_id: str):
    if not lib_id:
        raise HTTPException(status_code=400, detail="Library ID must be provided.")
    lock = await get_library_lock(lib_id)
    try:
        lock.acquire_read()
        vector_store = await get_vector_store()
        if not vector_store.has_library(UUID(lib_id)):
            raise HTTPException(status_code=404, detail="Library not found")
        library = vector_store.get_library(UUID(lib_id))
        return {"count": len(library.get_all_chunks())}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        lock.release_read()

async def search_chunks_by_library_service(lib_id: str, queryDto: QueryDto, k: int = 5):
    if not lib_id or not queryDto or not queryDto.query:
        raise HTTPException(status_code=422, detail="Library ID and query are required.")
    lock = await get_library_lock(lib_id)
    try:
        lock.acquire_read()
        vector_store = await get_vector_store()
        if len(queryDto.query) != EMBEDDING_DIM:
            raise HTTPException(
                status_code=400, detail=f"Query vector must be of length {EMBEDDING_DIM}")
        if not vector_store.has_library(UUID(lib_id)):
            raise HTTPException(status_code=404, detail="Library not found")
        filters = getattr(queryDto, 'filters', None)
        data = {"query": queryDto.query}
        if filters:
            data["filters"] = filters
        results = vector_store.search(
            UUID(lib_id), queryDto.query, k=k
        )
        if not filters:
            return results
        filter_obj = Filter(root=filters)
        results = [
            (chunk, score) for chunk, score in results
            if passes_filter(chunk.metadata, filter_obj)
        ]
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        lock.release_read()
