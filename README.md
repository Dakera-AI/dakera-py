[![Docs](https://img.shields.io/badge/docs-dakera.ai-D4A843)](https://dakera.ai/docs)
# dakera-py



[![CI](https://github.com/Dakera-AI/dakera-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Dakera-AI/dakera-py/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/dakera?logo=python&logoColor=white)](https://pypi.org/project/dakera/) [![Downloads](https://img.shields.io/pypi/dm/dakera)](https://pypi.org/project/dakera/) [![License: MIT](https://img.shields.io/github/license/Dakera-AI/dakera-py)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-dakera.ai-D4A843)](https://dakera.ai/docs)

Python SDK for Dakera AI — store, recall, and search agent memories against a Dakera instance.

Part of [Dakera AI](https://dakera.ai) — the memory engine for AI agents.

> The Dakera memory engine scores **87.6% on LoCoMo** (1,540 questions, standard eval) — [benchmark details](https://dakera.ai/benchmark)

---

## Run Dakera

You need a running Dakera server before using this SDK. The fastest way:

```bash
docker run -d \
  --name dakera \
  -p 3300:3300 \
  -e DAKERA_ROOT_API_KEY=dk-mykey \
  ghcr.io/dakera-ai/dakera:latest
```

For persistent storage (recommended for anything beyond a quick test):

```bash
curl -sSfL https://raw.githubusercontent.com/Dakera-AI/dakera-deploy/main/docker-compose.yml \
  -o docker-compose.yml
DAKERA_API_KEY=dk-mykey docker compose up -d

curl http://localhost:3300/health  # -> {"status":"ok"}
```

Full deployment guide (Docker Compose, Kubernetes, Helm): [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy)

---

## Install

```bash
pip install dakera
```

For async support:

```bash
pip install dakera[async]
```

## Quick Start

```python
from dakera import DakeraClient

client = DakeraClient(base_url="http://localhost:3300", api_key="dk-mykey")

# Store an agent memory
client.store_memory(
    agent_id="my-agent",
    content="User prefers concise responses with code examples",
    importance=0.9,
    tags=["preference"],
)

# Recall memories (semantic search)
response = client.recall(agent_id="my-agent", query="what does the user prefer?", top_k=5)
for m in response.memories:
    print(f"[{m.importance:.2f}] {m.content}")

# Upsert vectors
client.upsert("my-namespace", vectors=[
    {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"category": "docs"}},
])

# Full-text search
results = client.fulltext_search("my-namespace", query="completed task", top_k=5)
for r in results:
    print(r.id, r.score)
```

### Async

```python
from dakera import AsyncDakeraClient

async def main():
    client = AsyncDakeraClient(base_url="http://localhost:3300", api_key="dk-mykey")
    response = await client.recall(agent_id="my-agent", query="preferences", top_k=5)
    for m in response.memories:
        print(m.content)
```

## Features

- **Agent Memory** — store, recall, search, and forget memories with importance scoring
- **Sessions** — group memories by conversation with auto-consolidation on session end
- **Knowledge Graph** — traverse memory relationships, find paths, export graphs
- **Vector Search** — ANN queries with metadata filters and batch operations
- **Full-Text Search** — BM25 ranking with stemming and stop-word filtering
- **Hybrid Search** — combine vector similarity with keyword matching
- **Text Auto-Embedding** — server-side embedding generation (no local model needed)
- **Feedback Loop** — upvote/downvote/flag memories to improve recall quality
- **Entity Extraction** — GLiNER NER for automatic entity detection
- **Streaming** — SSE event subscriptions for real-time memory updates
- **Sync + Async** — full parity between `DakeraClient` and `AsyncDakeraClient`
- **Typed Models** — full type annotations with strict mypy, PEP 561 `py.typed` marker
- **Retry & Rate Limiting** — built-in exponential backoff and rate-limit header tracking
- **Filter DSL** — `F.eq()`, `F.gt()`, `F.contains()` typed filter builder

## Examples

See the [`examples/`](examples/) directory:

- [`basic_usage.py`](examples/basic_usage.py) — vectors, namespaces, queries, filters
- [`hybrid_search.py`](examples/hybrid_search.py) — full-text, vector, and hybrid search

## Connect to Dakera

```python
from dakera import DakeraClient

# Self-hosted
client = DakeraClient(base_url="http://your-server:3300", api_key="your-key")

# Cloud (early access)
client = DakeraClient(base_url="https://api.dakera.ai", api_key="your-key")

# With custom retry config
from dakera import RetryConfig
client = DakeraClient(
    base_url="http://localhost:3300",
    api_key="your-key",
    retry_config=RetryConfig(max_retries=5, base_delay=0.2),
)
```

## Documentation

-> [Full docs](https://dakera.ai/docs)  
-> [API reference](https://dakera.ai/docs/api)  
-> [Python SDK reference](https://dakera.ai/docs/sdk/python)

## Related

| Repo | What it is |
|---|---|
| [dakera-js](https://github.com/dakera-ai/dakera-js) | TypeScript SDK |
| [dakera-go](https://github.com/dakera-ai/dakera-go) | Go SDK |
| [dakera-rs](https://github.com/dakera-ai/dakera-rs) | Rust client |
| [dakera-cli](https://github.com/dakera-ai/dakera-cli) | CLI |
| [dakera-mcp](https://github.com/dakera-ai/dakera-mcp) | MCP server |
| [dakera-deploy](https://github.com/dakera-ai/dakera-deploy) | Self-host Dakera |

---

*Part of the Dakera AI open core. The engine is proprietary. The tools are yours.*
