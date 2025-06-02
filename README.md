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

#### Singleton design choice

It was important for the Vector Database to be decoupled from the REST API, since that is more of an implementation detail and can be swapped out for something else. Thus, it was a conscious choice to not use FastAPI's `Depends` dependency injection or similar API's to manage the data store. However, FastAPI's lifecycle methods like "startup" are used to kick off the snapshotting coroutine that runs async once the server is started. The singleton pattern makes it simple to ensure that only a single vector store instance is running at any time, and no additional ones are instantiated accidentally due to code errors.

### Read/Write Locking
- All vector store operations are wrapped in a custom `ReadWriteLock` (see `utils/read_write_lock.py`).
- Multiple readers can access the store concurrently; only one writer can mutate the store at a time.
- Locking is handled in the service layer, __not__ the API layer, since that can be swapped out later for something else, but the underlying Service can still mitigate data races.
- `asyncio` locks were used for this implementation since they are better suited to concurrent execution, as compared to locks provided by `threads`.

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
- Has a 1:many, compositional relationship with Chunks - one Library _has_ an array of multiple Chunks.
- Has a compositional relationship with Indexes. Each Library maintains one Index for its Chunks.

### Chunk
- Embedding + metadata pair
- Arbitrary, JSON-serializable metadata supported

### Document
- After initially implementing the `Document` model, I chose to remove it on a subsequent iteration, since I felt like it didn't add much value in terms of usability or API. For example, `Document`s own `Chunks` as per the problem statement, but the API expected mentions that users should be able to add/remove chunks from a library and also search for __relevant Chunks__ within a library. In this case, there's little reason to expose `Document`s as a separate entity.
- I believe this was a good design choice seeing that my reference database, Qdrant, also implements only `Collections (Libraries)` and `Points (Chunks)`!

## Indexing Algorithms

Indexes are implemented using inhertance. A `BaseIndex` abstract base class defined the `build` and `search` methods that different subclasses implement. For example, the BruteForceIndex implements traditional KNN indexing, while BallTreeIndex implements a tree data structure that is constructed and rebalanced using `build`.

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

### Other algorithms considered

## k-D Trees
- I considered k-D trees but decided not to implement them due to the fact that the high dimensionality of our embeddings would cause degraded search and building performance. Ball Trees seemed like a slightly better choice owing to their "radius-based" assignment of tree nodes.
- I also considered HNSW indexes, but due to the complexity and limited time for implementation, decided against them for the time being.

## Concurrency & Data Races
- Custom `ReadWriteLock` ensures safe data access/mutations.
- Multiple readers, single writer model.
- All service functions acquire the appropriate, __library-level lock__ before accessing/mutating data.
- A _global_ lock on the _entire_ store is used __only when saving/loading snapshots to/from disk.__

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

## Web UI
- Basic React UI for visualizing libraries and chunks
- Typescript types for API responses