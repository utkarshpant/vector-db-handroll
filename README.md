# Stack Vector Database

A modular, Pythonic vector database for RAG and NLP tasks, built with FastAPI and inspired by Qdrant. This README details the architecture, design decisions, technical choices, and usage instructions.

## Table of Contents
- [Installation](#installation)
- [Feature Overview](#feature-overview)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Data Model](#data-model)
- [Indexing Algorithms](#indexing-algorithms)
- [Concurrency & Data Races](#concurrency--data-races)
- [API & Service Layer](#api--service-layer)
- [Testing](#testing)
- [Docker & Running Locally](#docker--running-locally)
- [Python Client](#python-client)
- [Web UI](#web-ui)
- [Evaluation Criteria Mapping](#evaluation-criteria-mapping)

## Installation

1. Ensure Docker is installed.
2. Clone the repository and `cd` into it.
3. Build and start the container:
   ```sh
   docker compose up --build -d
   ```
4. Access the API at [http://localhost:8000](http://localhost:8000).
5. Stop the container with:
   ```sh
   docker compose down
   ```

## Feature Overview
- Storage, indexing, and querying of dense vector embeddings
- Two index types: Brute-force KNN and Ball-Tree (no external libraries)
- Upserting, querying, and deleting embeddings with filter support
- Disk persistence with periodic snapshots
- Read/write concurrency control via a custom ReadWriteLock
- RESTful API and minimal Python client
- Basic React-based Web UI
- OpenAPI docs at `/docs`

## Architecture & Design Decisions

### Vector Store
- The `VectorStore` is a singleton, async-initialized class, ensuring only one instance exists across the app (API, business logic, etc.).
- All consumers (API endpoints, services) access the same, ready instance via `await VectorStore.get_instance()`.
- Disk I/O (loading and saving snapshots) is async wherever possible.

### Read/Write Locking
- All vector store operations are wrapped in a custom `ReadWriteLock` (see `utils/read_write_lock.py`).
- Multiple readers can access the store concurrently; only one writer can mutate the store at a time.
- Locking is handled in the service layer, __not__ the API layer, since that can be swapped out later for something else, but the underlying Service can still mitigate data races.

### Disk Persistence
- The vector store persists all data to disk using `pickle` files, and loads from this pickle file on the next startup.
- Snapshots are taken every 10 seconds.

### Modularity and Extensibility
- Indexes are pluggable: new index types can be added by sub-classing `BaseIndex`.
- Business logic is separated from the API layer (service pattern).
- The API layer is thin and only handles HTTP concerns.
- The system can be extended to support gRPC or other APIs without changing core logic.

## Data Model

### Library
- Collection of embeddings (chunks)
- Has a unique ID, name, metadata, and an index

### Chunk
- Embedding + metadata pair
- Arbitrary, JSON-serializable metadata supported

### Document
- After initially implementing the `Document` model, I chose to remove it on a subsequent iteration, since I felt like it didn't add much value in terms of usability or API. For example, `Document`s own `Chunks` as per the problem statement, but the API expected mentions that users should be able to add/remove chunks from a library and also search for __relevant Chunks__ within a library. In this case, there's little reason to expose `Document`s as a separate entity.
- I believe this was a good design choice seeing that my reference database, Qdrant, also implements only `Collections (Libraries)` and `Points (Chunks)`!

## Indexing Algorithms

### Brute-force KNN
- Simple linear search over all embeddings
- Time Complexity: O(n) to build, O(n * d) to query
- Space Complexity: O(n * d)
- Chosen for simplicity and as a baseline

### Ball-Tree Index
- Tree-based structure for faster nearest neighbor search
- Time Complexity: O(log n) to build, O(log n * d) to query
- Space Complexity: O(n * d)
- Chosen for relatively improved performance on larger datasets


## Concurrency & Data Races
- Custom `ReadWriteLock` ensures thread/process safety.
- Multiple readers, single writer model.
- All service functions acquire the appropriate lock before accessing the vector store.

## API & Service Layer
- All CRUD and query logic is implemented in the `services/LibraryService.py` service layer.
- API endpoints (in `api/library_router.py`) _only_ handle HTTP concerns.
- All endpoints use Pydantic schemas for request/response validation (see http://localhost:8000/docs for the schema).

## Testing
- Unit and integration tests with `pytest` (see `tests/`)
- Test coverage includes CRUD, indexing, querying, and locking
- Run tests with:
  ```sh
  pytest -q
  ```

## Docker & Running Locally
- See [Installation](#installation) for steps.

## Python Client
- Minimal client in `sdk/` for sync operations
- Example usage in `demo.py`

---

## Web UI
- Basic React UI for visualizing libraries and chunks
- Typescript types for API responses