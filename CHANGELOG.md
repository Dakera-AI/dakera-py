# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
