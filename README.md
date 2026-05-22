<p align="center">
  <img src="https://github.com/dakera-ai.png" alt="Dakera AI" width="80" />
</p>

<h1 align="center">dakera-py</h1>

<p align="center">
  Python SDK for <a href="https://dakera.ai">Dakera AI</a> — the memory engine for AI agents
</p>

<p align="center">
  <a href="https://github.com/Dakera-AI/dakera-py/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Dakera-AI/dakera-py/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://pypi.org/project/dakera/"><img alt="PyPI" src="https://img.shields.io/pypi/v/dakera?logo=python&logoColor=white" /></a>
  <a href="https://pypi.org/project/dakera/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/dakera" /></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/Dakera-AI/dakera-py" /></a>
  <a href="https://dakera.ai/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-dakera.ai%2Fdocs-3b82f6?style=flat-square" /></a>
  <a href="https://dakera.ai/benchmark"><img alt="LoCoMo 87.8%" src="https://img.shields.io/badge/LoCoMo-87.8%25-22c55e?style=flat-square" /></a>
</p>

---

## Why Dakera?

| | Dakera | Others |
|---|---|---|
| **LoCoMo accuracy** | **87.8%** (1,540 Q standard eval) | 60–92% |
| **Deployment** | Single binary, Docker one-liner | External vector DB + embedding service required |
| **Embeddings** | Built-in — no OpenAI key needed | Requires external embedding API |
| **Search modes** | Vector · BM25 · Hybrid · Knowledge Graph | Usually one or two |
| **Transport** | HTTP + gRPC | HTTP only |

→ [Full benchmark results](https://dakera.ai/benchmark) · [dakera.ai](https://dakera.ai)

---

## Run Dakera

```bash
docker run -d \
  --name dakera \
  -p 3000:3000 \
  -e DAKERA_ROOT_API_KEY=dk-mykey \
  ghcr.io/dakera-ai/dakera:latest

curl http://localhost:3000/health  # → {"status":"ok"}
```

For persistent storage with Docker Compose:

```bash
curl -sSfL https://raw.githubusercontent.com/Dakera-AI/dakera-deploy/main/docker-compose.yml \
  -o docker-compose.yml
DAKERA_API_KEY=dk-mykey docker compose up -d
```

Full deployment guide (Docker Compose, Kubernetes, Helm): [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy)

---

## Install

```bash
pip install dakera
```

For async support (`AsyncDakeraClient`):

```bash
pip install dakera[async]
```

Works with **LangChain**, **LlamaIndex**, **CrewAI**, **AutoGen**, and any Python agent framework.

---

## Quick Start

```python
from dakera import DakeraClient

client = DakeraClient(base_url="http://localhost:3000", api_key="dk-mykey")

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

# Hybrid search (vector + BM25)
results = client.hybrid_search("my-namespace", query="completed task", top_k=5, vector_weight=0.7)
for r in results:
    print(r.id, r.score)
```

### Async

```python
import asyncio
from dakera import AsyncDakeraClient

async def main():
    client = AsyncDakeraClient(base_url="http://localhost:3000", api_key="dk-mykey")
    response = await client.recall(agent_id="my-agent", query="preferences", top_k=5)
    for m in response.memories:
        print(m.content)

asyncio.run(main())
```

---

## Features

- **Agent Memory** — store, recall, search, and forget memories with importance scoring
- **Sessions** — group memories by conversation with auto-consolidation on session end
- **Knowledge Graph** — traverse memory relationships, find paths, export graphs
- **Vector Search** — ANN queries with metadata filters and batch operations
- **Full-Text Search** — BM25 ranking with stemming and stop-word filtering
- **Hybrid Search** — combine vector similarity with keyword matching
- **Text Auto-Embedding** — server-side embedding generation (no local model needed)
- **Namespaces** — isolated vector stores per project, tenant, or use case
- **Feedback Loop** — upvote/downvote/flag memories to improve recall quality
- **Entity Extraction** — GLiNER NER for automatic entity detection
- **Streaming** — SSE event subscriptions for real-time memory updates
- **Sync + Async** — full parity between `DakeraClient` and `AsyncDakeraClient`
- **Typed Models** — full type annotations with strict mypy, PEP 561 `py.typed` marker
- **Retry & Rate Limiting** — built-in exponential backoff and rate-limit header tracking
- **Filter DSL** — `F.eq()`, `F.gt()`, `F.contains()` typed filter builder

---

## Connect to Dakera

```python
from dakera import DakeraClient, RetryConfig

# Self-hosted
client = DakeraClient(base_url="http://your-server:3000", api_key="your-key")

# Cloud (early access)
client = DakeraClient(base_url="http://localhost:3000", api_key="your-key")

# With custom retry config
client = DakeraClient(
    base_url="http://localhost:3000",
    api_key="your-key",
    retry_config=RetryConfig(max_retries=5, base_delay=0.2),
)
```

---

## Examples

See the [`examples/`](examples/) directory:

- [`basic_usage.py`](examples/basic_usage.py) — vectors, namespaces, queries, filters
- [`hybrid_search.py`](examples/hybrid_search.py) — full-text, vector, and hybrid search

---

## Resources

| | |
|---|---|
| [Documentation](https://dakera.ai/docs) | Full API reference and guides |
| [Python SDK docs](https://dakera.ai/docs/sdk/python) | Python-specific reference |
| [Benchmark](https://dakera.ai/benchmark) | LoCoMo evaluation results |
| [dakera.ai](https://dakera.ai) | Website and early access |
| [GitHub Org](https://github.com/dakera-ai) | All public repos |
| [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy) | Self-hosting guide |

### Other SDKs

| SDK | Package |
|---|---|
| [dakera-js](https://github.com/dakera-ai/dakera-js) | `@dakera-ai/dakera` (npm) |
| [dakera-rs](https://github.com/dakera-ai/dakera-rs) | `dakera-client` (crates.io) |
| [dakera-go](https://github.com/dakera-ai/dakera-go) | `github.com/dakera-ai/dakera-go` |
| [dakera-cli](https://github.com/dakera-ai/dakera-cli) | CLI tool |
| [dakera-mcp](https://github.com/dakera-ai/dakera-mcp) | MCP server for Claude/Cursor |

---

<p align="center">
  <a href="https://dakera.ai">dakera.ai</a> ·
  <a href="https://dakera.ai/docs">Docs</a> ·
  <a href="https://dakera.ai/benchmark">Benchmark</a> ·
  <a href="https://dakera.ai#cta">Request Early Access</a>
</p>

<p align="center"><sub>Built with Rust. Single binary. Zero external dependencies.</sub></p>
