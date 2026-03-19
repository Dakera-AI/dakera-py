# Dakera Python SDK

[![CI](https://github.com/dakera-ai/dakera-py/actions/workflows/ci.yml/badge.svg)](https://github.com/dakera-ai/dakera-py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dakera)](https://pypi.org/project/dakera/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/dakera)](https://pypi.org/project/dakera/)

Official Python client for [Dakera](https://dakera.ai) — a high-performance vector database for AI agent memory.

## Installation

```bash
pip install dakera
```

For async support:
```bash
pip install dakera[async]
```

## Quick Start

### Synchronous

```python
from dakera import DakeraClient

client = DakeraClient("http://localhost:3000")

# Upsert vectors
client.upsert("my-namespace", vectors=[
    {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"label": "a"}},
    {"id": "vec2", "values": [0.4, 0.5, 0.6], "metadata": {"label": "b"}},
])

# Query similar vectors
results = client.query("my-namespace", vector=[0.1, 0.2, 0.3], top_k=10)
for result in results.results:
    print(f"{result.id}: {result.score}")
```

### Async

```python
import asyncio
from dakera import AsyncDakeraClient

async def main():
    async with AsyncDakeraClient("http://localhost:3000") as client:
        # Upsert vectors
        await client.upsert("my-namespace", vectors=[
            {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"label": "a"}},
            {"id": "vec2", "values": [0.4, 0.5, 0.6], "metadata": {"label": "b"}},
        ])

        # Query similar vectors
        results = await client.query("my-namespace", vector=[0.1, 0.2, 0.3], top_k=10)
        for result in results.results:
            print(f"{result.id}: {result.score}")

asyncio.run(main())
```

## Features

- **Vector Operations**: Upsert, query, delete, fetch vectors
- **Full-Text Search**: Index documents and perform BM25 search
- **Hybrid Search**: Combine vector and text search with configurable weights
- **Namespace Management**: Create, list, delete namespaces
- **Metadata Filtering**: Filter queries by metadata fields
- **Async Support**: `AsyncDakeraClient` built on `httpx` for non-blocking I/O
- **Agent Memory**: Store, recall, and manage memories for AI agents
- **Type Hints**: Full type annotation support
- **Context Manager**: Automatic connection cleanup (sync and async)

## Usage Examples

### Vector Operations

```python
from dakera import DakeraClient, Vector

client = DakeraClient("http://localhost:3000")

# Using dataclass
vectors = [
    Vector(id="vec1", values=[0.1, 0.2, 0.3], metadata={"category": "A"}),
    Vector(id="vec2", values=[0.4, 0.5, 0.6], metadata={"category": "B"}),
]
client.upsert("my-namespace", vectors)

# Using dictionaries
client.upsert("my-namespace", vectors=[
    {"id": "vec3", "values": [0.7, 0.8, 0.9]},
])

# Query with metadata filter
results = client.query(
    "my-namespace",
    vector=[0.1, 0.2, 0.3],
    top_k=5,
    filter={"category": {"$eq": "A"}},
    include_metadata=True,
)

# Batch query
batch_results = client.batch_query("my-namespace", queries=[
    {"vector": [0.1, 0.2, 0.3], "top_k": 5},
    {"vector": [0.4, 0.5, 0.6], "top_k": 3},
])

# Delete vectors
client.delete("my-namespace", ids=["vec1", "vec2"])
client.delete("my-namespace", filter={"category": {"$eq": "obsolete"}})
```

### Full-Text Search

```python
# Index documents
client.index_documents("my-namespace", documents=[
    {"id": "doc1", "content": "Machine learning is transforming industries"},
    {"id": "doc2", "content": "Vector databases enable semantic search"},
])

# Search
results = client.fulltext_search(
    "my-namespace",
    query="machine learning",
    top_k=10,
)

for result in results:
    print(f"{result.id}: {result.score}")
```

### Hybrid Search

```python
# Combine vector and text search
results = client.hybrid_search(
    "my-namespace",
    vector=[0.1, 0.2, 0.3],  # Embedding of query
    query="machine learning",  # Text query
    top_k=10,
    alpha=0.7,  # 0 = pure vector, 1 = pure text
)

for result in results:
    print(f"{result.id}: score={result.score}, vector={result.vector_score}, text={result.text_score}")
```

### Namespace Management

```python
# Create namespace with specific configuration
client.create_namespace(
    "embeddings",
    dimensions=384,
    index_type="hnsw",
)

# List all namespaces
namespaces = client.list_namespaces()
for ns in namespaces:
    print(f"{ns.name}: {ns.vector_count} vectors")

# Get namespace info
info = client.get_namespace("embeddings")
print(f"Dimensions: {info.dimensions}, Index: {info.index_type}")

# Delete namespace
client.delete_namespace("old-namespace")
```

### Metadata Filtering

Dakera supports rich metadata filtering:

```python
# Equality
filter = {"status": {"$eq": "active"}}

# Comparison
filter = {"price": {"$gt": 100, "$lt": 500}}

# In list
filter = {"category": {"$in": ["electronics", "books"]}}

# Logical operators
filter = {
    "$and": [
        {"status": {"$eq": "active"}},
        {"price": {"$lt": 1000}},
    ]
}

results = client.query(
    "products",
    vector=query_embedding,
    filter=filter,
    top_k=20,
)
```

### Async Operations

Use `AsyncDakeraClient` (requires `pip install dakera[async]`) for non-blocking I/O with `asyncio`:

```python
import asyncio
from dakera import AsyncDakeraClient

async def main():
    async with AsyncDakeraClient("http://localhost:3000") as client:
        # All methods are coroutines — await each call
        await client.upsert("my-namespace", vectors=[
            {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"category": "A"}},
        ])

        # Parallel queries with asyncio.gather
        results_a, results_b = await asyncio.gather(
            client.query("my-namespace", vector=[0.1, 0.2, 0.3], top_k=5,
                         filter={"category": {"$eq": "A"}}),
            client.hybrid_search("my-namespace",
                                 vector=[0.1, 0.2, 0.3],
                                 query="machine learning",
                                 top_k=5, alpha=0.7),
        )

asyncio.run(main())
```

#### Async Agent Memory

```python
import asyncio
from dakera import AsyncDakeraClient

async def main():
    async with AsyncDakeraClient("http://localhost:3000") as client:
        # Store a memory
        stored = await client.store_memory(
            "my-agent",
            "The user prefers concise responses",
            importance=0.8,
            metadata={"source": "feedback"},
        )

        # Recall relevant memories
        memories = await client.recall("my-agent", "user preferences", top_k=5)
        for mem in memories:
            print(f"{mem['id']}: {mem['content']} (score: {mem.get('score', 'N/A')})")

asyncio.run(main())
```

### Error Handling

```python
from dakera import (
    DakeraClient,
    NotFoundError,
    ValidationError,
    RateLimitError,
)

client = DakeraClient("http://localhost:3000")

try:
    results = client.query("nonexistent", vector=[0.1, 0.2])
except NotFoundError as e:
    print(f"Namespace not found: {e}")
except ValidationError as e:
    print(f"Invalid request: {e}")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after} seconds")
```

### Context Manager

```python
# Synchronous
with DakeraClient("http://localhost:3000") as client:
    client.upsert("my-namespace", vectors=[...])
    results = client.query("my-namespace", vector=[...])

# Async
import asyncio
from dakera import AsyncDakeraClient

async def main():
    async with AsyncDakeraClient("http://localhost:3000") as client:
        await client.upsert("my-namespace", vectors=[...])
        results = await client.query("my-namespace", vector=[...])

asyncio.run(main())
```

### Authentication

```python
# With API key
client = DakeraClient(
    "http://localhost:3000",
    api_key="your-api-key",
)

# With custom headers
client = DakeraClient(
    "http://localhost:3000",
    headers={"X-Custom-Header": "value"},
)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | required | Dakera server URL |
| `api_key` | str | None | API key for authentication |
| `timeout` | float | 30.0 | Request timeout in seconds |
| `max_retries` | int | 3 | Max retries for failed requests |
| `headers` | dict | None | Additional HTTP headers |

## API Reference

Both `DakeraClient` (sync) and `AsyncDakeraClient` (async) expose identical interfaces. Async methods are prefixed with `await`.

### Vector Operations
- `upsert(namespace, vectors)` - Insert or update vectors
- `query(namespace, vector, top_k, filter, ...)` - Query similar vectors
- `delete(namespace, ids, filter, delete_all)` - Delete vectors
- `fetch(namespace, ids)` - Fetch vectors by ID
- `batch_query(namespace, queries)` - Execute multiple queries

### Full-Text Operations
- `index_documents(namespace, documents)` - Index documents
- `fulltext_search(namespace, query, top_k, filter)` - BM25 text search
- `hybrid_search(namespace, vector, query, alpha, ...)` - Hybrid vector + text search

### Namespace Operations
- `list_namespaces()` - List all namespaces
- `get_namespace(namespace)` - Get namespace info
- `create_namespace(namespace, dimensions, index_type)` - Create namespace
- `delete_namespace(namespace)` - Delete namespace

### Agent Memory Operations
- `store_memory(agent_id, content, importance, metadata, ...)` - Store a memory
- `recall(agent_id, query, top_k, ...)` - Recall relevant memories
- `get_memory(agent_id, memory_id)` - Fetch a specific memory
- `update_memory(agent_id, memory_id, content, ...)` - Update a memory
- `forget(agent_id, memory_id)` - Delete a memory
- `search_memories(agent_id, query, top_k, ...)` - Search memories

### Admin Operations
- `health()` - Check server health
- `get_index_stats(namespace)` - Get index statistics
- `compact(namespace)` - Trigger compaction
- `flush(namespace)` - Flush pending writes

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/dakera

# Linting
ruff check src/
```

## Related Repositories

| Repository | Description |
|------------|-------------|
| [dakera](https://github.com/dakera-ai/dakera) | Core vector database engine (Rust) |
| [dakera-js](https://github.com/dakera-ai/dakera-js) | TypeScript/JavaScript SDK |
| [dakera-go](https://github.com/dakera-ai/dakera-go) | Go SDK |
| [dakera-rs](https://github.com/dakera-ai/dakera-rs) | Rust SDK |
| [dakera-cli](https://github.com/dakera-ai/dakera-cli) | Command-line interface |
| [dakera-mcp](https://github.com/dakera-ai/dakera-mcp) | MCP Server for AI agent memory |
| [dakera-dashboard](https://github.com/dakera-ai/dakera-dashboard) | Admin dashboard (Leptos/WASM) |
| [dakera-docs](https://github.com/dakera-ai/dakera-docs) | Documentation |
| [dakera-deploy](https://github.com/dakera-ai/dakera-deploy) | Deployment configs and Docker Compose |

## License

MIT License - see [LICENSE](LICENSE) for details.
