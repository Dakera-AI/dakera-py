# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.2] - 2026-04-16

### Changed
- **v0.11.2:** Server default fusion strategy changed from `FusionStrategy.RRF` to
  `FusionStrategy.MINMAX` (CEO architecture decision, DAK-1948). A/B benchmark conclusive:
  MinMax +6.3pp overall Recall@10, +13.5pp temporal (Cat 3). Callers that rely on the server
  default will now use MinMax; pass `FusionStrategy.RRF` explicitly to keep RRF behaviour.
  Updated docstrings to reflect the new server default.

## [0.11.1] - 2026-04-16

### Fixed
- No code changes in this release. Version bump for parity with `dakera-rs` v0.11.1, which
  fixed a serialization bug where `FusionStrategy::MinMax` was sent as `"min_max"` instead of
  `"minmax"`. Python serialized `FusionStrategy.MINMAX` correctly as `"minmax"` in v0.11.0 —
  no action required if you are using this SDK.

## [0.11.0] - 2026-04-15

### Added
- **CE-14:** `FusionStrategy` enum (`FusionStrategy.RRF` / `FusionStrategy.MINMAX`) — controls how vector and BM25 scores are combined in hybrid recall.
- **CE-14:** `fusion` parameter on `DakeraClient.recall()` and `AsyncDakeraClient.recall()`. `None` uses server default (`FusionStrategy.RRF`). Use `FusionStrategy.MINMAX` for legacy weighted min-max normalization.
- **v0.11.0:** `neighborhood` parameter on `DakeraClient.recall()` and `AsyncDakeraClient.recall()`. Session-adjacent memory enrichment (±5 min). `None` uses server default (`True`). Pass `False` to disable for latency-sensitive paths.

## [0.10.3] - 2026-04-15

### Added
- **CE-13:** `rerank` parameter on `DakeraClient.recall()` and `DakeraClient.search_memories()` (and async equivalents). Enables cross-encoder reranking via `Xenova/bge-reranker-base`. `None` uses server default (`True` for recall, `False` for search). Pass `False` to disable on latency-sensitive paths.
- **CE-13:** `EmbeddingModel.BGE_LARGE` (`"bge-large"`, 1024 dimensions) — new server-default embedding model.

### Changed
- Updated `pytest` dev dependency 9.0.2 → 9.0.3.

### Notes
- CE-13 reranking and `BGE_LARGE` align with server v0.10.2; the SDK v0.10.2 tag predated these additions.

## [0.10.2] - 2026-04-13

### Notes
- Server v0.10.2 tracking release (bge-large embedding + cross-encoder reranking). SDK CE-13 additions ship in v0.10.3.

## [0.10.0] - 2026-04-12

### Added
- **CE-10:** `RoutingMode` enum (`auto` | `vector` | `bm25` | `hybrid`) — controls which retrieval index to use for recall and search.
- **CE-10:** `routing` parameter on `DakeraClient.recall()` and `DakeraClient.search_memories()` (and async equivalents). Defaults to `None` (server picks `auto`).
- **CE-12:** `compress_agent(agent_id)` method on `DakeraClient` and `AsyncDakeraClient` — calls `POST /v1/agents/{id}/compress` and returns a `CompressResponse` with before/after memory counts and timing.
- **CE-12:** `CompressResponse` dataclass with fields `agent_id`, `memories_before`, `memories_after`, `removed_count`, `duration_ms`.
- **CE-10:** `MemoryPolicy.dedup_on_store` (bool, default `False`) — enable similarity deduplication at store time.
- **CE-10:** `MemoryPolicy.dedup_threshold` (float, default `0.92`) — cosine-similarity threshold for store-time deduplication.

## [0.9.15] - 2026-04-08

### Notes
- Version bump to match server v0.9.15. No SDK API changes.
- Server changes (transparent to SDK callers):
  - **DAK-1691:** Session-end auto-consolidation — `end_session` now clusters near-duplicate session memories via DBSCAN and soft-expires them (30-day TTL). High-importance memories (>0.8) are protected from decay. No request/response signature change.
  - **DAK-1689:** HNSW post-filter ANN fix — filtered vector queries are now O(N·ANN) instead of O(N·linear). No SDK change.

## [0.9.14] - 2026-04-07

### Added
- **DAK-1690: Agent wake-up context endpoint:**
  - `DakeraClient.wake_up(agent_id, top_n=20, min_importance=0.0)` and `AsyncDakeraClient.wake_up(...)` — `GET /v1/agents/{agent_id}/wake-up` — returns a `WakeUpResponse` with top-N memories ranked by importance × recency decay. Sub-millisecond; no embedding inference. Requires Read scope.
  - `WakeUpResponse` dataclass exported from `dakera.models` and top-level `dakera` package: `agent_id`, `memories: list[Memory]`, `total_available: int`.

## [0.9.13] - 2026-04-07

### Fixed
- **Session response unwrapping (DAK-1548):** `start_session()` now correctly unwraps the `{"session": {...}}` server response, resolving a `KeyError` on `result["id"]` when using the returned session object directly. Affects both sync (`DakeraClient`) and async (`AsyncDakeraClient`) variants.

## [0.9.12] - 2026-04-06

### Added
- **OBS-2: Product KPI Snapshot endpoint:**
  - `DakeraClient.get_kpis()` and `AsyncDakeraClient.get_kpis()` — `GET /v1/kpis` — returns a
    `KpiSnapshot` with 8 real-time operational metrics. Sub-millisecond; served from in-memory counters.
    Requires Admin scope.
  - `KpiSnapshot` dataclass exported from `dakera.models` and the top-level `dakera` package:
    - `recall_latency_p50_ms` / `recall_latency_p99_ms` — median/p99 recall latency (ms)
    - `store_latency_p50_ms` — median store latency (ms)
    - `api_error_rate_5xx_pct` — 5xx error rate as a percentage of total requests
    - `active_agents_count` — distinct agents active in the last 24 hours
    - `session_count_week` — sessions created in the rolling 7-day window
    - `cross_agent_network_node_count` — nodes in the cross-agent knowledge graph
    - `memory_retention_7d_pct` — percentage of memories from 7 days ago still active

### Server-side only (no SDK changes required)
- **v0.9.12 performance fixes:** session-agent index lookup reduced to O(1); memory counters
  now updated via atomic increments; S3 flushes are async (non-blocking).

## [0.9.11] - 2026-04-01

### Added
- **KG-3: Deep Associative Recall bindings:**
  - `RecalledMemory` gains a new `depth: int | None` field — the KG hop at which an associated memory was found.
  - `DakeraClient.recall()` and `AsyncDakeraClient.recall()` accept two new optional parameters:
    - `associated_memories_depth: int | None` — KG traversal depth 1–3 (default: `1`); requires `include_associated=True`.
    - `associated_memories_min_weight: float | None` — minimum KG edge weight threshold (default: `0.0`).
  - Fully backward-compatible: omitting both new parameters retains depth-1 (COG-2) behaviour.
- **COG-3: Proactive Memory Consolidation bindings:**
  - `MemoryPolicy` gains four new fields:
    - `consolidation_enabled: bool` (default `False`) — opt-in background DBSCAN deduplication.
    - `consolidation_threshold: float` (default `0.92`) — cosine-similarity epsilon; higher = only merge very close neighbours.
    - `consolidation_interval_hours: int` (default `24`) — how often the background job runs.
    - `consolidated_count: int` (default `0`, **read-only**) — lifetime count of memories merged by the server.
  - `MemoryPolicy.to_dict()` serialises the three writable COG-3 fields; `consolidated_count` is excluded from PUT payloads.
  - `MemoryPolicy.from_dict()` deserialises all four fields from GET responses.
- **SEC-5: Per-namespace rate limiting bindings:**
  - `MemoryPolicy` gains three new fields:
    - `rate_limit_enabled: bool` (default `False`) — opt-in per-namespace store/recall rate limiting.
    - `rate_limit_stores_per_minute: int | None` (default `None` = unlimited) — max store operations per minute.
    - `rate_limit_recalls_per_minute: int | None` (default `None` = unlimited) — max recall operations per minute.
  - `MemoryPolicy.to_dict()` serialises `rate_limit_enabled` unconditionally; the two optional int fields are omitted when `None`.
  - `MemoryPolicy.from_dict()` deserialises all three fields from GET responses.
  - When a limit is exceeded the server returns HTTP 429; the existing `RateLimitError` is raised with `retry_after=60`.

## [0.9.9] - 2026-03-31

### Added
- **CE-7: Time-Window Recall bindings:**
  - `recall()` (sync and async) now accepts `since: str | None = None` and
    `until: str | None = None` ISO-8601 timestamp parameters.
  - Filters are applied server-side before semantic ranking — only memories
    created within the specified window are considered.
  - Invalid ISO-8601 values raise a `400` error from the server.

## [0.9.8] - 2026-03-31

### Added
- **COG-2: Associative Recall bindings:**
  - `recall()` now accepts `include_associated: bool = False` and
    `associated_memories_cap: int | None = None` parameters.
  - When `include_associated=True`, the server performs a KG depth-1
    traversal from the primary recalled memories and returns
    associatively linked memories in `associated_memories`.
  - Return type changed from `list[dict]` to `RecallResponse` — a
    dataclass with `memories: list[RecalledMemory]` and
    `associated_memories: list[RecalledMemory] | None`.
  - Available on both `DakeraClient` (sync) and `AsyncDakeraClient` (async).
  - New export: `RecallResponse`.
- **COG-1: Cognitive Memory Lifecycle bindings:**
  - `get_memory_policy(namespace)` — retrieve the memory lifecycle policy for a
    namespace (`GET /v1/namespaces/{namespace}/memory_policy`). Returns
    `MemoryPolicy` with the current settings (or COG-1 defaults if not set).
  - `set_memory_policy(namespace, policy)` — set the memory lifecycle policy
    (`PUT /v1/namespaces/{namespace}/memory_policy`). Policy persists across
    restarts and is applied immediately to the decay engine.
  - Both methods available on `DakeraClient` (sync) and `AsyncDakeraClient` (async).
  - New type: `MemoryPolicy` — controls per-type TTLs
    (`working_ttl_seconds`, `episodic_ttl_seconds`, `semantic_ttl_seconds`,
    `procedural_ttl_seconds`), per-type decay strategies (`working_decay`,
    `episodic_decay`, `semantic_decay`, `procedural_decay` — one of
    `"exponential"`, `"linear"`, `"step"`, `"power_law"`, `"logarithmic"`,
    `"flat"`), and spaced repetition (`spaced_repetition_factor`,
    `spaced_repetition_base_interval_seconds`).

## [0.9.7] - 2026-03-31

### Added
- **KG-2: Graph Query & Export bindings:**
  - `knowledge_query(agent_id, root_id?, edge_type?, min_weight?, max_depth?, limit?)` —
    filter-based DSL query over the memory knowledge graph
    (`GET /v1/knowledge/query`). Returns `KgQueryResponse` with matching edges
    and deduplicated node count.
  - `knowledge_path(agent_id, from_id, to_id)` — BFS shortest path between two
    memory IDs (`GET /v1/knowledge/path`). Returns `KgPathResponse` with the
    ordered hop list.
  - `knowledge_export(agent_id, format?)` — export the full graph as JSON or
    GraphML (`GET /v1/knowledge/export`). Returns `KgExportResponse` for
    `format="json"`.
  - All three methods available on both `DakeraClient` (sync) and
    `AsyncDakeraClient` (async).
  - New types: `KgQueryResponse`, `KgPathResponse`, `KgExportResponse`.

## [0.9.6] - 2026-03-30

### Added
- **GLiNER Entity Extraction via ODE sidecar (ODE-2):**
  - `ode_extract_entities(content, agent_id, memory_id?, entity_types?)` — extract
    named entities from text using the dakera-ode GLiNER sidecar
    (`POST /ode/extract`). Returns `ExtractEntitiesResponse` with per-entity
    character offsets, confidence scores, the model variant used, and processing
    time in ms.
  - New `ode_url` constructor parameter on `DakeraClient` and `AsyncDakeraClient`.
  - Async variant: `ode_extract_entities()` on `AsyncDakeraClient`.
  - New types: `OdeEntity` (fields: `text`, `label`, `start`, `end`, `score`),
    `ExtractEntitiesResponse` (fields: `entities`, `model`, `processing_time_ms`).

## [0.9.5] - 2026-03-30

### Added
- **AES-256-GCM Encryption Key Rotation (SEC-3):**
  - `rotate_encryption_key(new_key, namespace?)` — re-encrypt all memory content
    blobs with a new AES-256-GCM key (`POST /v1/admin/encryption/rotate-key`).
    Pass `namespace=None` to rotate all namespaces. Returns
    `RotateEncryptionKeyResponse`. Requires Admin scope.
  - Async variant: `async_rotate_encryption_key()` available on `AsyncDakeraClient`.
  - New types: `RotateEncryptionKeyRequest`, `RotateEncryptionKeyResponse`
    (fields: `rotated`, `skipped`, `namespaces`).

## [0.9.4] - 2026-03-30

### Added
- **Memory Import/Export (DX-1):**
  - `import_memories(data, format, agent_id?, namespace?)` — import memories from
    Mem0, Zep, JSONL, or CSV format (`POST /v1/import`). Returns
    `MemoryImportResponse` with counts and errors.
  - `export_memories(format, agent_id?, namespace?, limit?)` — export memories to
    a portable format (`GET /v1/export`). Returns `MemoryExportResponse`.
  - New types: `MemoryImportResponse`, `MemoryExportResponse`.
  - Async variants available on `AsyncDakeraClient`.
- **Business-Event Audit Log (OBS-1):**
  - `list_audit_events(agent_id?, event_type?, from_ts?, to_ts?, limit?, cursor?)`
    — paginated audit log query (`GET /v1/audit`). Returns `AuditListResponse`.
  - `stream_audit_events(agent_id?, event_type?, timeout?)` — live SSE stream of
    audit events (`GET /v1/audit/stream`). Yields `DakeraEvent` objects.
  - `export_audit(format?, agent_id?, event_type?, from_ts?, to_ts?)` — bulk
    export audit entries (`POST /v1/audit/export`). Returns `AuditExportResponse`.
  - New types: `AuditEvent`, `AuditListResponse`, `AuditExportResponse`.
  - Async variants available on `AsyncDakeraClient`.
- **DBSCAN Adaptive Consolidation (CE-6):** `consolidate()` now accepts an
  optional `config: ConsolidationConfig` parameter to select the clustering
  algorithm (`"dbscan"` or `"greedy"`) and tune `min_samples`/`eps`.
  Response may include a `log` list of `ConsolidationLogEntry` steps.
  New types: `ConsolidationConfig`, `ConsolidationLogEntry`.
- **External Extraction Providers (EXT-1):**
  - `extract_text(text, namespace?, provider?, model?)` — extract entities from
    text via a pluggable provider (`POST /v1/extract`). Providers: `gliner`
    (zero-config, bundled), `openai`, `anthropic`, `openrouter`, `ollama`.
    Returns `ExtractionResult`.
  - `list_extract_providers()` — list available providers and their models
    (`GET /v1/extract/providers`). Returns `list[ExtractionProviderInfo]`.
  - `configure_namespace_extractor(namespace, provider, model?)` — set the
    default extractor for a namespace (`PATCH /v1/namespaces/{ns}/extractor`).
  - New types: `ExtractionResult`, `ExtractionProviderInfo`.
  - Async variants available on `AsyncDakeraClient`.
- **Redis Health (OPS-3):** `cluster_status()` response now includes a
  `redis_healthy` boolean field indicating Redis connectivity.
- **Cluster Env Aliases (DIST-1):** Documented new server-side environment
  variable aliases: `DAKERA_CLUSTER_NODE_ID`, `SEED_NODES`, `BIND_ADDR`.
- **Memory Encryption (SEC-3):** Server supports AES-256-GCM at-rest encryption
  via `DAKERA_ENCRYPTION_KEY`. No SDK changes required — transparent to clients.

## [0.9.3] - 2026-03-29

### Added
- **Prometheus Metrics (INFRA-3):** `ops_metrics()` — returns the raw Prometheus
  text exposition format string from `GET /v1/ops/metrics` (Admin scope). Enables
  programmatic access to all 10+ core memory API metrics.

## [0.9.2] - 2026-03-27

### Added
- **Namespace-scoped API Keys (SEC-1):**
  - `create_namespace_key(namespace, name, expires_in_days)` /
    `async create_namespace_key()` — create a scoped API key for a namespace
    (`POST /v1/namespaces/{ns}/keys`). Returns `CreateNamespaceKeyResponse`.
    The raw key is shown **only once** — store it securely.
  - `list_namespace_keys(namespace)` / `async list_namespace_keys()` — list all
    API keys for a namespace (`GET /v1/namespaces/{ns}/keys`). Returns
    `ListNamespaceKeysResponse` (key secrets are never exposed in listings).
  - `delete_namespace_key(namespace, key_id)` / `async delete_namespace_key()` —
    revoke a namespace API key (`DELETE /v1/namespaces/{ns}/keys/{key_id}`).
    Returns a dict with `success` and `message` fields.
  - `get_namespace_key_usage(namespace, key_id)` /
    `async get_namespace_key_usage()` — retrieve usage stats for a key
    (`GET /v1/namespaces/{ns}/keys/{key_id}/usage`). Returns
    `NamespaceKeyUsageResponse`.
  - New types: `NamespaceKeyInfo`, `CreateNamespaceKeyResponse`,
    `ListNamespaceKeysResponse`, `NamespaceKeyUsageResponse` — all exported from
    the top-level `dakera` package.

### Fixed
- `__version__` in `dakera/__init__.py` was not updated during the v0.9.1 release;
  corrected to match `pyproject.toml`.

## [0.9.1] - 2026-03-26

### Added
- **Memory Feedback Loop (INT-1):**
  - `feedback_memory(memory_id, agent_id, signal, note)` / `async feedback_memory()` — submit
    feedback (upvote/downvote/flag) for a memory (`POST /v1/memories/{id}/feedback`). Returns
    `FeedbackResponse`.
  - `patch_memory_importance(memory_id, agent_id, importance)` / `async patch_memory_importance()`
    — directly set a memory's importance score (`PATCH /v1/memories/{id}/importance`). Returns
    `FeedbackResponse`.
  - `get_memory_feedback_history(memory_id)` / `async get_memory_feedback_history()` — retrieve
    all feedback events for a memory (`GET /v1/memories/{id}/feedback/history`). Returns
    `FeedbackHistoryResponse`.
  - `get_agent_feedback_summary(agent_id)` / `async get_agent_feedback_summary()` — aggregate
    feedback counts and health score for an agent (`GET /v1/agents/{id}/feedback/summary`).
    Returns `AgentFeedbackSummary`.
  - `get_feedback_health(agent_id)` / `async get_feedback_health()` — health score (mean
    importance of non-expired memories) for an agent (`GET /v1/feedback/health`). Returns
    `FeedbackHealthResponse`.
  - New types: `FeedbackSignal` (enum: `UPVOTE` / `DOWNVOTE` / `FLAG`), `FeedbackResponse`,
    `FeedbackHistoryEntry`, `FeedbackHistoryResponse`, `MemoryFeedbackBody`,
    `MemoryImportancePatch`, `AgentFeedbackSummary`, `FeedbackHealthResponse` — all exported
    from the top-level `dakera` package.

### Security
- **CVE-2026-25645** — bumped `requests` to `>=2.33.0` to fix an SSRF/redirect vulnerability.
- Dropped Python 3.9 support (EOL October 2025). Minimum supported version is now **Python 3.10**.

## [0.9.0] - 2026-03-26

### Added
- **Memory Knowledge Graph API (SDK-9 / CE-5 pre-impl):**
  - `memory_graph(memory_id, depth, types)` / `async memory_graph()` — returns the graph of
    memories connected to `memory_id` (`GET /v1/memories/{id}/graph`). Depth and edge-type
    filters are optional.
  - `memory_path(source_id, target_id)` / `async memory_path()` — shortest path between two
    memory nodes (`GET /v1/memories/{id}/path`).
  - `memory_link(source_id, target_id, edge_type)` / `async memory_link()` — create a directed
    edge between two memories (`POST /v1/memories/{id}/links`).
  - `agent_graph_export(agent_id, format)` / `async agent_graph_export()` — export the full
    memory graph for an agent as JSON or CSV (`GET /v1/agents/{id}/graph/export`).
  - New types: `EdgeType`, `GraphEdge`, `GraphNode`, `MemoryGraph`, `GraphPath`,
    `GraphLinkResponse`, `GraphExport` — all exported from the top-level `dakera` package.
  - **Note:** requires server CE-5 for end-to-end functionality; unit tests use mocked
    responses and pass fully against the current server (server CE-5 / DAK-1002).
- **Real-time memory event streaming (SDK-10):**
  - `agents_subscribe(agent_id, tag_filter, reconnect)` — async generator yielding
    `MemoryEvent` objects from `GET /v1/events/stream`. Supports tag-based filtering and
    optional auto-reconnect. Skips the `connected` handshake event automatically.

## [0.8.6] - 2026-03-25

### Changed
- `ops_stats()` / `async ops_stats()` — response now includes `state` field (`"healthy"` or
  `"degraded"`) reflecting storage health. Syncs with core DAK-918 (`/v1/ops/stats` fix).

## [0.8.5] - 2026-03-25

### Added
- `DakeraClient.ops_stats()` / `AsyncDakeraClient.ops_stats()` — new Read-scoped endpoint
  `GET /v1/ops/stats` returns `version`, `total_vectors`, `namespace_count`, `uptime_seconds`,
  `timestamp`. Works with read-only API keys; use instead of `cluster_status()` when you only
  need basic server stats (core DAK-852).

## [0.8.4] - 2026-03-24

### Security
- Bumped `urllib3` to `>=2.2.2` to resolve CVE-2024-37891 (proxy credential leak via
  `Proxy-Authorization` header). Python runtime dependency only; no API changes.

## [0.8.2] - 2026-03-23

### Added
- `StoreMemoryRequest.expires_at` — optional explicit expiry Unix timestamp (seconds). Takes
  precedence over `ttl_seconds` when both are set (core DECAY-3 / DAK-740).
- `AsyncDakeraClient.store_memory()` — added `ttl_seconds` and `expires_at` parameters, matching
  the sync `DakeraClient.store_memory()` interface.
- `MemoryEvent`: SSE `connected` handshake event is now surfaced. When the server emits
  `event: connected` on stream subscription, callers receive a `MemoryEvent` with
  `event_type="connected"` and `agent_id=""`. Use this to distinguish *connected-and-idle*
  from *not-yet-connected* (core DAK-720).

### Changed
- `MemoryEvent.agent_id` default changed from required to `""` to accommodate the `connected`
  handshake event, which carries no agent context.

## [0.8.1] - 2026-03-23

### Fixed
- `DakeraClient.hybrid_search()` / `AsyncDakeraClient.hybrid_search()` — corrected endpoint URL
  from `/v1/namespaces/{ns}/fulltext/hybrid` to `/v1/namespaces/{ns}/hybrid` (DAK-679).
  Hybrid search was returning HTTP 404 in production since v0.8.0.
  (Rust SDK dakera-client was unaffected.)

## [0.8.0] - 2026-03-23

### Changed
- `DakeraClient.hybrid_search()` — `vector` parameter is now optional (keyword-only, default `None`).
  When omitted the server performs BM25-only full-text search. Existing callers that pass `vector`
  continue to work unchanged. **Signature change**: `query` is now the second positional arg;
  `vector` moves to a keyword argument. (core v0.8.0 / dakera-mcp PR#20)

## [0.7.3] - 2026-03-23

### Added
- `DakeraClient.store_memory()` — new `ttl_seconds` and `expires_at` parameters (DECAY-3).
  `expires_at` accepts a Unix timestamp (seconds) and takes precedence over `ttl_seconds`
  when both are provided; the memory is hard-deleted by the decay engine on expiry.
- `DakeraClient.decay_config()` — `GET /v1/admin/decay/config` — current decay strategy,
  half-life, and minimum importance threshold (DECAY-1). Requires Admin scope.
- `DakeraClient.decay_update_config()` — `PUT /v1/admin/decay/config` — update decay
  settings at runtime with no restart required (DECAY-1). All parameters optional.
- `DakeraClient.decay_stats()` — `GET /v1/admin/decay/stats` — cumulative decay counters
  and last-cycle snapshot (DECAY-2). Requires Admin scope.

## [0.7.2] - 2026-03-23

### Added
- `DakeraClient.autopilot_status()` — `GET /v1/admin/autopilot/status` — current config + last-run stats (PILOT-1)
- `DakeraClient.autopilot_update_config()` — `PUT /v1/admin/autopilot/config` — live config update (PILOT-2);
  all parameters optional (`enabled`, `dedup_threshold`, `dedup_interval_hours`, `consolidation_interval_hours`)
- `DakeraClient.autopilot_trigger(action)` — `POST /v1/admin/autopilot/trigger` — manual dedup/consolidation
  cycle; `action` is one of `"dedup"`, `"consolidate"`, or `"all"` (PILOT-3)

## [0.7.1] - 2026-03-22

### Security
- Bumped `urllib3` from 2.2.3 → 2.6.3 (via `requests` transitive dependency) — resolves 5 CVEs:
  - **CVE-2024-37891** (HIGH): urllib3 forwarding `Authorization` headers to third-party redirects
  - Two additional HIGH-severity decompression vulnerability CVEs
  - Two MEDIUM-severity CVEs
- Dropped Python 3.8 support (EOL October 2024); minimum supported version is now Python 3.9

## [0.7.0] - 2026-03-22

### Added
- Batch recall endpoint: `POST /v1/memories/recall/batch` — `recall_batch()` / `async_recall_batch()` methods
- Batch forget endpoint: `DELETE /v1/memories/forget/batch` — `forget_batch()` / `async_forget_batch()` methods
- Rate-limit response headers exposed: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## [0.6.2] - 2026-03-21

### Added
- `CrossAgentNetworkResponse.node_count` field — reflects the `node_count` field added in
  dakera server v0.6.2 (PR #26). Previously the field was silently ignored.
- SSE endpoints now support `?api_key=<key>` query-parameter authentication in addition to
  the `Authorization: Bearer` header. Useful when constructing streaming URLs for clients that
  cannot send custom headers (e.g. browser-native `EventSource`).

## [0.2.0] - 2026-03-19

### Changed
- Version aligned with Go and Rust SDKs (v0.2.0 parity)

## [0.1.0] - 2025-03-15

### Added
- Initial release of Dakera Python SDK
- Synchronous client with full API coverage
- Vector operations: upsert, query, fetch, delete
- Namespace management
- Full-text search support
- Agent memory operations
- Session management
- Knowledge graph operations
- Inference (auto-embedding) support
- Typed models for all API responses
- Comprehensive test suite
- Example scripts
