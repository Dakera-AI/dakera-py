# ⚡ dakera-py

Python SDK for Dakera AI — store, recall, and search agent memories against a Dakera instance.

Part of [Dakera AI](https://dakera.ai) — the memory engine for AI agents.

> The Dakera memory engine scores **87.8% on LoCoMo** (1,540 questions, standard eval) — [benchmark details](https://dakera.ai/benchmark)

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

*Part of the Dakera AI open core. The engine is proprietary. The tools are yours.*
