# ⚡ dakera-py

[![CI](https://github.com/Dakera-AI/dakera-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Dakera-AI/dakera-py/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/dakera?logo=python&logoColor=white)](https://pypi.org/project/dakera/) [![Downloads](https://img.shields.io/pypi/dm/dakera)](https://pypi.org/project/dakera/) [![License: MIT](https://img.shields.io/github/license/Dakera-AI/dakera-py)](LICENSE)
[![dakera.ai](https://img.shields.io/badge/dakera.ai-website-22c55e?style=flat-square)](https://dakera.ai) [![Docs](https://img.shields.io/badge/docs-dakera.ai%2Fdocs-3b82f6?style=flat-square)](https://dakera.ai/docs)

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

curl http://localhost:3300/health  # → {"status":"ok"}
```

Full deployment guide (Docker Compose, Kubernetes, Helm): [dakera-deploy](https://github.com/Dakera-AI/dakera-deploy)

---

## Install

```bash
pip install dakera
```

## Quick Start

```python
from dakera import DakeraClient

client = DakeraClient(base_url="http://localhost:3300", api_key="your-key")

# Store a vector
client.vectors.upsert(
    id="vec-001",
    values=[0.1, 0.2, 0.3],
    metadata={"text": "agent completed task", "agent_id": "my-agent"}
)

# Full-text search
results = client.fulltext.search(query="completed task", top_k=5)
for r in results:
    print(r.id, r.score)

# Store an agent memory
client.memories.store(
    agent_id="my-agent",
    content="User prefers concise responses",
    importance=0.8,
    tags=["preference", "ux"]
)
```

## Connect to Dakera

```python
from dakera import DakeraClient

# Self-hosted
client = DakeraClient(base_url="http://your-server:3300", api_key="your-key")

# Cloud (early access)
client = DakeraClient(base_url="https://api.dakera.ai", api_key="your-key")
```

## Documentation

→ [Full docs](https://dakera.ai/docs)  
→ [API reference](https://dakera.ai/docs/api)  
→ [Python SDK reference](https://dakera.ai/docs/sdk/python)

## Related

| Repo | What it is |
|---|---|
| [dakera-js](https://github.com/dakera-ai/dakera-js) | TypeScript SDK |
| [dakera-go](https://github.com/dakera-ai/dakera-go) | Go SDK |
| [dakera-rs](https://github.com/dakera-ai/dakera-rs) | Rust client |
| [dakera-cli](https://github.com/dakera-ai/dakera-cli) | CLI |
| [dakera-mcp](https://github.com/dakera-ai/dakera-mcp) | MCP server · 83 tools |
| [dakera-deploy](https://github.com/dakera-ai/dakera-deploy) | Self-host Dakera |

---

**[dakera.ai](https://dakera.ai)** · [Documentation](https://dakera.ai/docs) · [Request Early Access](https://dakera.ai#cta)

<sub>Part of the Dakera AI open-source ecosystem. Built with Rust. Self-hosted. Zero dependencies.</sub>
