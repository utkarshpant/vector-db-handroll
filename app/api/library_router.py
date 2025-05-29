from fastapi import APIRouter
from app.core.Chunk import Chunk
from app.core.Document import Document
from app.core.Library import Library
from app.services.vector_store import VectorStore

# DTOs for different operations on a Library
from app.api.dto.library import LibraryListItem, LibraryCreate

vector_store = VectorStore()
router = APIRouter()

@router.get("/", response_model=list[LibraryListItem])
def list_libraries():
    """
    List all libraries in the vector store.
    """
    libraries = vector_store.get_all_libraries()
    return [LibraryListItem.model_validate(lib) for lib in libraries]

@router.post('/', response_model=LibraryListItem)
def createLibrary(libraryData: LibraryCreate):
    """
    Create a new library with the given name.
    """
    lib_id = vector_store.create_library(libraryData.name, libraryData.metadata)
    library = vector_store.get_library(lib_id)
    return LibraryListItem.model_validate(library)