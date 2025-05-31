# Stack Vector Database

This project aims to provide a simplistic vector database for RAG and NLP tasks, and this (slightly long!) README goes over some of the design decisions (system design, scalability, API surface, etc.). It also documents how to install and use the database in other projects.

## Installation

This project is Dockerized for ease of use and distribution, and installation is fairly straightforward.

1. Ensure you have Docker installed on your machine.
2. Clone the repository, and `cd` into it.
3. Build the Docker container using `docker compose --build -d`, and wait for it to finish.
4. Once the container is provisioned, you can access the API at `http://localhost:8000`.
5. To stop the container, run `docker compose down`.

## Feature overview

This vector database is heavily inspired by [Qdrant](https://qdrant.tech/) and aims to support a similar API. As such, it supports the following features:
1. Storage, indexing and querying of dense vector embeddings.
2. Two kinds of Indexes:
    - __Brute-force KNN__, and
    - __Ball-Tree Indexing__ for faster querying
3. __Upsering, querying and deleting embeddings based on filters.__ (More on this later.)
4. Persistence on disk, so that the database can be restarted without losing data, with snapshots taken every 10 seconds.
5. We try to mitigate read-write contention by using a ReadWriteLock, which allows multiple readers to read from the database at the same time, while only allowing one writer to write to the database at a time.
6. A RESTful API that allows you to interact with the database using HTTP requests, with support for JSON payloads.
7. A minimal Python Client that allows you to interact with the database using Python, currently supporting only `sync` operations.
8. A __very basic__ Web UI that allows you (for now) to see currently saved Libraries, and delete them. More views can be added later.

__P.S.__ Since the REST API is built using FastAPI, we have the free benefit of _awesome_ OpenAPI-style documentation, which you can access at `http://localhost:8000/docs` after starting the container.

## Architecture

The main considerations for the architecture of this vector database were:
1. Maintaining a simple data model, and consequently, a simple API.
2. Modularity, so that the database can be extended in the future.

To that end, the following data model was chosen:

### `Library`:

A Library is a collection of embeddings, and is the main entity in the database. It has the following properties:
- `id`: A unique identifier for the library.
- `name`: A human-readable name for the library.
- `embeddings`: A list of embeddings, where each embedding is a vector of floats.
- `index`: The index used for querying the embeddings, which can be either a brute-force KNN index or a ball-tree index.
- `metadata`: A dictionary of metadata associated with the library, which can be used to store additional information about the library.
    - `created_at`: A timestamp of when the library was created. __This field is added to metadata automatically.__

The Vector Database relies on `Libraries` to orchestrate the storage and retrieval of Chunks, building and maintaining indexes, and providing a simple API for interacting with the database.

### `Chunk`:

A `Chunk` is a single _embedding_ and _metadata_ pair. __It supports addition of arbitrary metadata fields, as long as they are JSON-serializable and can be filtered on using logical/arithmetic operators.__

It has the following properties:
- `id`: A unique identifier for the chunk.
- `embedding`: A vector of floats representing the __dense__ embedding.
- `metadata`: A dictionary of metadata associated with the chunk, which can be used to store additional information about the chunk, such as the corresponding text, source, etc.

### Omission of `Document`:

I made the decision to __remove the implementation of the `Document` entity__, which was present in the original design and assignment details. This was done because the `Document` entity did not seem to add much value to the data model, or to the API, since the primary expectation is to be able to manipulate Libraries and Chunks. 

Additionally, as noted earlier, this vector database is heavily inspired by Qdrant, which does not have a `Document` entity either. There, `Point`s are analogous to `Chunk`s, and `Collection`s are analogous to a `Library`.

### Libraries and Chunks have a one-to-many _compositional_ relationship!

### Indexing:

A `BaseIndex` abstract class is used to define the interface for all indexes, and two concrete implementations of this class are provided: `BruteForceKNNIndex` and `BallTreeIndex`.

A `BaseIndex` descendent implements:
1. `build_index(library: Library)`: Builds the index for the given library.
2. `search(query_embedding: List[float], k: int, library: Library)`: Searches for the `k` nearest neighbors of the given query embedding in the specified library.

### `Indexes` are implemented using inheritance to make the vector store _extensible_ but not _modifyable_. Each `Index` adheres to the same API.

In the interest of simplicity, the database currently supports two kinds of indexes:
1. __Brute-force KNN__: This is the simplest form of indexing, where all embeddings are stored in a list, and the distance between the query embedding and all other embeddings is computed to find the nearest neighbors. This is not very efficient for large datasets, but it is simple to implement and works well for small datasets.
    - Time Complexity: __O(n)__ to build and __O(n * d)__ to query, where $n$ is the number of embeddings and $d$ is the dimensionality of the embeddings.
    - Space Complexity: __O(n * d)__ to store the embeddings, where $n$ is the number of embeddings and $d$ is the dimensionality of the embeddings.

2. __Ball-Tree Indexing__: This is a more efficient form of indexing, where the embeddings are stored in a tree structure, and the distance between the query embedding and the embeddings in the tree is computed to find the nearest neighbors. This is more efficient for large datasets, but might become less efficient as the number of dimensions in the embeddings increases.
    - Time Complexity: __O(log n)__ to build and __O(log n * d)__ to query, where $n$ is the number of embeddings and $d$ is the dimensionality of the embeddings.
    - Space Complexity: __O(n * d)__ to store the embeddings, where $n$ is the number of embeddings and $d$ is the dimensionality of the embeddings.

I considered other indexing methods such as

1. __k-D trees__, but decided against them since their performance seems to degrade _very quickly_ as the number of dimensions in the embeddings increases;
2. __HNSW (Hierarchical Navigable Small World) graphs__, but decided against them in the interest of time since I am not very familiar with how HNSW works.

### Domain-driven design:

A common pattern in REST API design I have used is to separate the business-logic from the API layer, so that the API layer is only responsible for handling HTTP requests and responses, while the business-logic is handled by a separate, `Service` layer. In this project, I've implemented a `LibraryService` class that handles the business-logic for the vector database.

__This way, at a later date, a different API layer can be implemented, such as a gRPC API, without having to change the business-logic.__

#### Disk persistence and read-write locks:

The `LibraryService` layer incorporates the logic to acquire and release read-write locks before every relevant operation.

The `VectorStore` class is responsible for managing the persistence of the libraries and chunks on disk. It uses the `pickle` module to serialize and deserialize the libraries and chunks, and stores them in a directory on disk. The `VectorStore` class also implements a snapshot mechanism that takes a snapshot of the libraries and chunks every 10 seconds, so that the database can be restarted without losing data.

## Testing and Validation

To make sure I was testing the end result of different operations and not the implementation detail, I used `pytest` to write unit and integration tests for `Library`, `Chunk`, `VectorStore`, and the REST API. The test suite covers CRUD operations, indexing, and querying of embeddings, as well as tests for the ReadWrite Lock implementation that prevents read-write contention.

> To run the tests, `cd` into the repository and run `pytest -q` from the command line. Some tags also exist for integration tests, such as `--tags=integration`.

Additionally, having defined the Data Model using `pydantic`, I was able to use the `pydantic` validation features to ensure that the data being passed to the API, and the responses it sends are also valid and serializable.

## Python Client
A minimal Python client is provided in the `sdk` directory, which allows you to interact with the vector database using Python. The client currently supports only synchronous operations, and provides methods for all operations supported by the REST API.

`demo.py` in the project root directory provides a simple example of how to use the client to create a library, upsert chunks, and query the library.

## Web UI

Inspired by Qdrant's Web UI, I also tried to use the REST API to develop a Web UI to visualize the data store. This was built using React Router v7, and I tried to define rich TypeScript types wherever possible to compare Python's type system with TypeScript's, which I am more familiar with.

## Documentation

The API documentation is automatically generated by FastAPI, and can be accessed at `http://localhost:8000/docs` after starting the container. It provides a comprehensive overview of the API endpoints, request and response schemas, and example requests.

__Additionally, the repository root also contains a Postman collection that can be used to test the API endpoints. You can import the collection into Postman and use it to test the API.__