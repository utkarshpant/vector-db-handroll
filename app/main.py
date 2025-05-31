from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from app.api.library_router import router as library_router
from app.api.dto.Library import LibraryCreate, LibraryListItem, LibraryResponse, UpsertChunksDto

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
def health_check():
    return {"status": "ok"}

app.include_router(library_router, prefix="/library", tags=["libraries"])

app.mount("/ui", StaticFiles(directory="./ui/dist/", html=True), name="static")
app.mount("/assets", StaticFiles(directory="./ui/dist/assets", html=False), name="chunks")