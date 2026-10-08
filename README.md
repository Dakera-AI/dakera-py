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
  <a href="https://dakera.ai/benchmark"><img alt="LoCoMo 88.2%" src="https://img.shields.io/badge/LoCoMo-88.2%25-22c55e?style=flat-square" /></a>
  <a href="https://dakera.ai/playground"><img alt="Playground" src="https://img.shields.io/badge/playground-try_it-ff6b35?style=flat-square" /></a>
</p>

---

## Why Dakera?

| | Dakera | Others |
|---|---|---|
| **LoCoMo Recall@20** | **88.2%** (1,536 Q, LLM-judged retrieval recall) | not directly comparable |
| **Deployment** | Single binary, Docker one-liner | External vector DB + embedding service required |
| **Embeddings** | Built-in — no OpenAI key needed | Requires external embedding API |
| **Search modes** | Vector · BM25 · Hybrid · Knowledge Graph | Usually one or two |
| **Transport** | HTTP + gRPC | HTTP only |

→ [Try the playground](https://dakera.ai/playground) · [Full benchmark results](https://dakera.ai/benchmark) · [dakera.ai](https://dakera.ai)

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
curl -sSfL https://raw.githubusercontent.com/Dakera-AI/dakera-deploy/main/docker/docker-compose.yml \
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
client.store_memory(agent_id="my-agent", content="User prefers brevity", importance=0.9)
```

Full example — store, recall, upsert, and hybrid search:

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
- **T-I-F Reliability** — `TifScore` and `evaluate_tif()` for Truth-Indeterminacy-Falsity scoring of memory reliability
- **Entity Extraction** — GLiNER NER for automatic entity detection
- **Streaming** — SSE event subscriptions for real-time memory updates
- **Sync + Async** — full parity between `DakeraClient` and `AsyncDakeraClient`
- **Typed Models** — full type annotations with strict mypy, PEP 561 `py.typed` marker
- **Retry & Rate Limiting** — built-in exponential backoff, `Retry-After` honoured on `503`/`429`, and rate-limit header tracking
- **Attachments & Records** — upload audio/images, transcribe or index them into memories, store multi-representation records (server v0.12+)
- **Agents, Keys & Session Lifecycle** — create agents, edit keys and grant prefix patterns, rotate with a grace period, `whoami`, session idle timeouts and `touch` (server v0.12.2+)
- **Filter DSL** — `F.eq()`, `F.gt()`, `F.contains()` typed filter builder

---

## What's new for Dakera server v0.12.2

Version 0.14.0 of this SDK adds support for Dakera server **v0.12.2** and stays
**compatible with v0.12.0 and v0.12.1 servers**: every new request field is sent
only when you set it, and every new response field is optional (`None` / `[]`
from an older server). The v0.12.2-only routes answer `404` / `405` on an older
server.

- **Agents** — `create_agent(agent_id)` (`POST /v1/agents`) creates an agent's
  memory namespace before its first memory (`created=False` for an existing one).
- **Keys** — `update_key()` / `update_namespace_key()` rename a key or replace its
  `namespaces` (`all_namespaces=True` grants every namespace); `rotate_key(key_id,
  grace_secs=N)` keeps the old key working up to 7 days (`old_key_id`,
  `old_key_expires_at` in the answer; `RotateKeyResponse.from_dict()` types it);
  `whoami()` (`GET /v1/auth/whoami`); `KeyInfo.grants_version` /
  `inert_namespaces`; `create_key()` sends `scope` (default `read`),
  `namespaces` (exact names or `p*` prefix patterns such as
  `["_dakera_agent_mlx-*"]`) and `expires_in_days`.
- **Sessions** — `start_session(..., idle_timeout_secs=N)`, `touch_session()`
  (`SessionTouchResponse`: `session_state`, `idle_deadline_at`), and the session
  fields `last_activity_at`, `ended_reason` (`client` | `idle`), `idle_since`,
  `idle_timeout_secs` (`Session.from_dict()` types them). `store_memory()`
  returns `session_state` when the memory went into a session;
  `BatchStoreMemoryResponse.ended_sessions` lists ended sessions a batch stored
  into. `update_config(session_idle_timeout_secs=N)` sets the server-wide timeout.
  `ChatMemorySession.create(..., idle_timeout_secs=N)` and `.touch()`.
- **Listings** — `agent_memories(..., include_derived=, content_preview_chars=,
  offset=)`, `session_memories(..., content_preview_chars=, limit=, offset=)`,
  `wake_up(..., include_derived=)`, and `content_preview_chars` on
  `full_knowledge_graph()` / `cross_agent_network()`. With a preview each memory
  or node carries `content_len` and `content_truncated`; read a truncated memory
  in full with `get_memory()` before showing or editing it.
- **Derived data** — `derivations_status()` and `drain_derivations(timeout_secs=)`
  (`GET /admin/derivations/status`, `POST /admin/derivations/drain`; a running
  drain is a `ConflictError`).
- **Capabilities v2** — `capabilities().auth`, `.naming`, `.sessions`;
  `NamespaceInfo.kind` (`agent` / `data` / `system`).
- **Additive fields** — `duplicates_skipped_changed` (deduplicate),
  `CompressResponse.summaries_skipped`, `unavailable` on node-wide endpoints
  (`NamespaceUnavailable` on `ttl_stats()`, `memory_type_stats()`,
  `storage_tier_overview()`; passed through on the dict-returning ones such as
  `ops_stats()`), `unavailable` on `list_agents()` entries, `MemoryEvent.reason`.

### Behaviour changes you may hit with a v0.12.2 server

These are server changes; the SDK does not hide them.

- **Sessions are authorized by their agent.** A key needs Read/Write on
  `_dakera_agent_<agent_id>`; a `_dakera_sessions` grant is no longer needed and
  is reported in `inert_namespaces`. A key without grants lists no sessions.
  `end_session()` with a Read key is a `403`.
- **Sessions end automatically after 4 h without activity** by default
  (`DAKERA_SESSION_IDLE_TIMEOUT_SECS`), with `ended_reason: "idle"`. Activity is a
  memory stored / updated with the session, a session-scoped recall or search, or
  `touch_session()`. Storing into an ended session still succeeds: check
  `session_state` / `ended_sessions`. On upgrade, sessions already idle longer
  than the timeout are closed on the first passes.
- **Stricter validation (`400`, the message names the field).** Invalid key
  `namespaces` entries; reserved markers (the `dakera-curated` tag, `_dakera_*`
  metadata keys other than `_dakera_content_date` / `_dakera_lang`, ids
  `mem_s` + 24 hex characters); metadata over 100 fields; `ttl_seconds` over 100
  years; agent ids over 241 bytes; `_dakera_embedding_models` is reserved.
- **The memory content limit is in UTF-8 bytes** (default 100000,
  `DAKERA_MAX_MEMORY_CONTENT_BYTES`), not characters. It now also applies to
  `update_memory()` (a memory stored above the limit may only be updated to
  content no larger than it is) and to the `end_session()` summary.
- **Listings exclude derived records by default.** `agent_memories()` and
  `wake_up()` no longer return the derived sentence sub-memories; pass
  `include_derived=True` for the previous listing.
- **Legacy `foo*` key entries stay inert** until the key's `namespaces` are saved
  again (`update_key(..., namespaces=[...])`); `grants_version` is `0` for such keys.

```python
from dakera import DakeraClient

client = DakeraClient("http://localhost:3000", api_key="your-key")

client.create_agent("mlx-dev")
session = client.start_session("mlx-dev", idle_timeout_secs=2 * 3600)
stored = client.store_memory("mlx-dev", "User prefers dark mode", session_id=session["id"])
if stored.get("session_state") == "ended":
    session = client.start_session("mlx-dev")

client.touch_session(session["id"])  # keep an idle session open
page = client.agent_memories("mlx-dev", limit=50, content_preview_chars=200)
print(client.whoami().scope)
```

---

## What's new for Dakera server v0.12.0

Version 0.13.0 of this SDK adds support for Dakera server **v0.12.0**
(operator upgrade guide: `docs/v0.12/UPGRADE.md` in the server release; release notes in the
[Dakera changelog](https://dakera.ai/docs/changelog)).

**Compatible with both v0.11.108 and v0.12.0 servers.** Everything new is additive:
calls that do not use a v0.12 feature send exactly what they sent before, and
the v0.12-only calls fail with a clear error (`NotFoundError` / `405`) on a v0.11
server.

- **Health and readiness** — a v0.12 server binds its port while models load and
  answers `503` + `Retry-After` on `/health`. `client.is_ready()` /
  `client.wait_until_ready()` use `/health/ready` (a `503` is never "healthy");
  `health_ready()` and `health_live()` map to `/health/ready` and `/health/live`.
- **Errors and retries** — every error body is JSON. `503` raises
  `ServiceUnavailableError` (a `ServerError`) and the retry logic waits for the
  server's `Retry-After`. `413` raises `PayloadTooLargeError` (`.is_quota` for a
  namespace quota, `.is_oversize` for an over-size request), `501` raises
  `FeatureNotAvailableError` (`details` names the `DAKERA_*` switch), `409`
  raises `ConflictError`; `NotFoundError.resource` says what was not found.
- **`GET /v1/capabilities`** — `client.capabilities()`: models (`bge-m3`,
  `colbert-small`), index kinds (`ivfpq`), search mode (`rabitq`), record kinds and
  dtypes, query languages, plus `scoring`, `attachments`, `vision`. Unknown
  strings parse as unknown enum members instead of raising.
- **Attachments** (opt-in on the server, `DAKERA_ATTACHMENTS`) —
  `upload_attachment`, `list_attachments`, `download_attachment`,
  `delete_attachment`, `store_memory(..., attachment_ref=...)`,
  `transcribe_attachment` / `get_transcription_job` / `wait_for_transcription`
  and, with `DAKERA_VISION`, `index_attachment` / `wait_for_index`.
- **Records** (opt-in, `DAKERA_RECORDS`) — `upsert_records` / `get_record`: one
  primary vector plus named `dense`, `token_multivector` or `patch_multivector`
  representations stored as `f32`, `f16` or `i8`.
- **Per-request `lang`** on `store_memory`, `store_memories_batch`,
  `update_memory`, `recall`, `search_memories` and `extract_entities`.
- **Namespace config** — `replace_namespace_ner_config()` (`PUT`) replaces the
  entity-extraction config; the v0.12 server's `PATCH` merges and refuses unknown
  fields.
- **Fix** — async `extract_entities()` sent `text` instead of `content`.

```python
from dakera import DakeraClient, Record, Representation, RepresentationKind, BlockDType

client = DakeraClient("http://localhost:3000", api_key="your-key")
client.wait_until_ready(timeout=120)

caps = client.capabilities()
if caps.supports_attachments:
    up = client.upload_attachment("_dakera_agent_a1", "note.wav")
    job = client.transcribe_attachment("_dakera_agent_a1", up.attachment_ref, "a1", lang="en")
    done = client.wait_for_transcription("_dakera_agent_a1", up.attachment_ref, job.job_id)

if caps.supports_records:
    client.upsert_records("docs", [Record(
        id="r1", values=[0.1, 0.2, 0.3, 0.4],
        representations=[Representation(
            "tokens", [[0.1, 0.2], [0.3, 0.4]],
            kind=RepresentationKind.TOKEN_MULTIVECTOR, store_as=BlockDType.F16)])])
```

Note: the v0.12 server's gRPC port requires an API key. This SDK speaks REST only.

---

## Connect to Dakera

```python
from dakera import DakeraClient, RetryConfig

# Self-hosted
client = DakeraClient(base_url="http://your-server:3000", api_key="your-key")

# Cloud (early access)
client = DakeraClient(base_url="http://<your-server-ip>:3000", api_key="your-key")

# With custom retry config
client = DakeraClient(
    base_url="http://localhost:3000",
    api_key="your-key",
    retry_config=RetryConfig(max_retries=5, base_delay=0.2),
)
```

---

## Integrations

### TealTiger Governance Middleware

[TealTiger](https://github.com/agentguard-ai/tealtiger) is a governance middleware for AI agents that enforces cost limits, decision policies, and delegation rules.  Use Dakera as the persistent backend for all TealTiger artefacts:

```bash
pip install dakera[tealtiger] tealtiger
```

```python
import asyncio
from dakera.async_client import AsyncDakeraClient
from dakera.integrations.tealtiger import (
    DakeraCostStorage,
    DakeraDecisionStore,
    DakeraDelegationHelper,
)

client = AsyncDakeraClient("http://localhost:3000", api_key="dk-mykey")

# Drop-in async CostStorage backend — passes directly to TealTiger client
cost_storage = DakeraCostStorage(client)

from tealtiger import TealOpenAI, TealOpenAIConfig
teal_client = TealOpenAI(config=TealOpenAIConfig(cost_storage=cost_storage))

# Governance decision audit trail with idempotency checks (all methods are async)
decision_store = DakeraDecisionStore(client)
# receipt_id = await decision_store.store_receipt("my-agent", decision)
# is_duplicate = await decision_store.is_terminal("my-agent", correlation_id)

# Multi-hop delegation chain traversal via memory knowledge graph
delegation = DakeraDelegationHelper(client)
await delegation.link_delegation(child_id=child_mem_id, parent_id=parent_mem_id, agent_id="my-agent")
chain = await delegation.get_delegation_chain("my-agent", root_id, max_depth=5)
```

All cost records, decision receipts, and delegation chains are stored in Dakera memory with importance-weighted retention (DENY receipts at 0.95 outlast ALLOW at 0.80) and full knowledge-graph traversal for audit purposes.

See [`examples/tealtiger_governance.py`](examples/tealtiger_governance.py) for a complete walkthrough.  Join the [integration discussion](https://github.com/Dakera-AI/dakera-deploy/discussions/169) or visit the [TealTiger repo](https://github.com/agentguard-ai/tealtiger).

---

## Examples

See the [`examples/`](examples/) directory:

- [`basic_usage.py`](examples/basic_usage.py) — vectors, namespaces, queries, filters
- [`hybrid_search.py`](examples/hybrid_search.py) — full-text, vector, and hybrid search
- [`ollama_memory_chat.py`](examples/ollama_memory_chat.py) — persistent memory for a local Ollama chat loop
- [`ollama_memory_proxy.py`](examples/ollama_memory_proxy.py) — transparent memory proxy in front of Ollama (`/api/chat`)
- [`tealtiger_governance.py`](examples/tealtiger_governance.py) — TealTiger governance middleware

---

## Resources

| | |
|---|---|
| [Documentation](https://dakera.ai/docs) | Full API reference and guides |
| [Python SDK docs](https://dakera.ai/docs/python-sdk) | Python-specific reference |
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
