from fastapi import FastAPI
from app.api.library_router import router as library_router
from app.api.dto.Library import LibraryCreate, LibraryListItem, LibraryResponse, UpsertChunksDto
app = FastAPI()

@app.get('/health')
def health_check():
    return {"status": "ok"}

app.include_router(library_router, prefix="/library", tags=["libraries"])