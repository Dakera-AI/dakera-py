# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.9] - 2026-07-02

### Changed

- **Server sync v0.11.96** — no SDK API surface changes; bundled with server v0.11.96
  engine and routes-handler improvements (DAK-7243).

## [0.12.8] - 2026-06-26

### Changed

- fix(changelog): correct reranker model org and version in CE-13 CHANGELOG entry (#175, #176)

## [0.12.7] - 2026-06-24

### Added

- **`EmbeddingModel.MODERNBERT_EMBED_BASE` / `EmbeddingModel.GTE_MODERNBERT_BASE`** — add two
  new enum members matching server DAK-7098/DAK-7102: `"modernbert-embed-base"` (768d, MRL,
  8192 tokens) and `"gte-modernbert-base"` (768d, MTEB retrieval 64.38). Pass via the `model`
  parameter of `upsert_text()` and `query_text()`.

## [0.12.6] - 2026-06-23

### Fixed

- **`ttl_cleanup()`** — add missing binding for `POST /v1/admin/ttl/cleanup`; previous
  callers received a 404 silently. (DAK-7069, [#170](https://github.com/Dakera-AI/dakera-py/pull/170))
- **Key management routes** — fix `create_key`, `list_keys`, `get_key`, `delete_key`,
  `deactivate_key`, `rotate_key`, `get_key_usage` to use `/admin/keys` instead of the
  incorrect `/v1/keys` path. (DAK-7066, [#171](https://github.com/Dakera-AI/dakera-py/pull/171))

### Internal

- Add 11 tests for 5 previously untested admin methods.
  ([#166](https://github.com/Dakera-AI/dakera-py/pull/166))

## [0.12.5] - 2026-06-17

### Added

- **`AsyncChatMemorySession`** — async counterpart to `ChatMemorySession` for use with
  `AsyncDakeraClient`. Supports `async with` context manager, async factory
  `await AsyncChatMemorySession.create(client, agent_id)`, and the same
  `store`/`recall`/`close` API as the sync class. Exported as
  `dakera.AsyncChatMemorySession`. (DAK-6895, [#160](https://github.com/Dakera-AI/dakera-py/pull/160))

### Internal

- **CI version-drift gate** — `__version__` in `src/dakera/__init__.py` is now
  asserted to match `pyproject.toml` on every CI run to prevent version skew.
  (DAK-6850, [#159](https://github.com/Dakera-AI/dakera-py/pull/159))

## [0.12.4] - 2026-06-16

### Added

- **`ChatMemorySession`** — new convenience helper class for the LLM chat
  comparison pattern used by the playground (DAK-6846,
  [#158](https://github.com/Dakera-AI/dakera-py/pull/158)). Wraps
  `start_session` / `store_memory` / `recall` / `end_session` into a
  clean context-manager API:

  ```python
  from dakera import DakeraClient
  from dakera.session import ChatMemorySession  # or: from dakera import ChatMemorySession

  client = DakeraClient(url, api_key=key)

  with ChatMemorySession.create(client, agent_id="my-agent") as session:
      session.store("user", "My name is Alice.")
      context = session.recall("who am I", top_k=5)
      # pass context to your LLM call
  # session auto-closed on exit
  ```

  Key methods: `ChatMemorySession.create()`, `.store()`, `.recall()`,
  `.close()`. Also available as a plain class (no `with` required) for
  async or long-lived use-cases.

- **Playground chat comparison example** —
  `examples/playground/chat_comparison.py` shows the full side-by-side
  pattern: seed turns, open a comparison session, recall context, call an
  LLM with and without memory, and display a diff.

## [0.12.3] - 2026-06-16

### Added

- **`admin_reembed_static_count()`** — new `DakeraClient` and `AsyncDakeraClient` method
  for `GET /v1/admin/reembed/static-count` (v0.11.91+, DAK-6781,
  [#152](https://github.com/Dakera-AI/dakera-py/pull/152)). Returns a
  `StaticCountResponse` with the count of `_embedding_kind=static` vectors pending
  ONNX upgrade. A `static_count` of 0 means steady state.

- **Playground quickstart** — `examples/playground/quickstart.py` demonstrates store,
  recall, search, and knowledge-graph link against the public sandbox. (DAK-6737,
  [#145](https://github.com/Dakera-AI/dakera-py/pull/145))

### Fixed

- **`memory_link()` `KeyError: 'edge'` on forbidden endpoint** (DAK-6785,
  [#154](https://github.com/Dakera-AI/dakera-py/pull/154)) — when the server
  returns an application-level error body (`{"error": "forbidden_endpoint"}`) in a
  200 response, `memory_link()` now raises `AuthorizationError` instead of crashing
  in `GraphLinkResponse.from_dict()`.

- **`search_memories()` content always empty** — flattens the server's nested
  `{"memory": {"content": "..."}, "score": N}` envelope so callers receive populated
  content, ID, and metadata. Previously content was always `""`. (DAK-6780,
  [#151](https://github.com/Dakera-AI/dakera-py/pull/151))

- **Admin URL corrections** — `admin_index_stats()`, `rebuild_indexes()`, and
  `restore_backup()` used incorrect URL patterns causing 404s. (DAK-6775,
  [#150](https://github.com/Dakera-AI/dakera-py/pull/150))

- **Playground HTTPS endpoint** — quickstart now uses
  `https://5-75-177-31.sslip.io` (valid TLS) instead of bare HTTP IP. (DAK-6743,
  [#146](https://github.com/Dakera-AI/dakera-py/pull/146))

- **Sandbox KG-link restriction** — playground examples now handle the sandbox 403
  on `/v1/memories/{id}/links` gracefully instead of raising. (DAK-6749,
  [#148](https://github.com/Dakera-AI/dakera-py/pull/148))

- **Quickstart search step nested envelope** — search results in the quickstart example
  now correctly access `result["memory"]["content"]`. (DAK-6784,
  [#153](https://github.com/Dakera-AI/dakera-py/pull/153))

### Testing

- **Remove `xfail` from admin endpoint tests** — `TestAdminEndpoints` xfail markers
  removed after server v0.11.91 confirmed all admin routes live under `/v1/admin/`.
  ([#147](https://github.com/Dakera-AI/dakera-py/pull/147))

- **Playground integration tests** — `tests/test_playground_integration.py` added;
  covers store, recall, search, and KG-link (graceful). Tests skip automatically
  when `DAKERA_TEST_URL` is absent.
  ([#149](https://github.com/Dakera-AI/dakera-py/pull/149))

## [0.12.2] - 2026-06-14

### CI

- **`actions/checkout` 4 → 6** — CI workflow dependency upgrade
  ([#142](https://github.com/Dakera-AI/dakera-py/pull/142)). No runtime changes.

> **Note on release ordering**: this tag was cut on 2026-06-14 from a Dependabot PR
> before the feature releases v0.12.0 (TealTiger integration) and v0.12.1 (version fix)
> were tagged on 2026-06-15. Version ordering reflects SemVer, not chronological order.

## [0.12.1] - 2026-06-15

### Fixed

- `dakera.__version__` now correctly returns `"0.12.1"` (was `"0.11.91"` in the
  initial `0.12.0` tag due to a version bump ordering issue).
- CHANGELOG: corrected `DakeraCostStorage` method names — `store()`, `get()`,
  `get_by_agent_id()`, `get_by_date_range()`, `get_by_request_id()`, `get_summary()`,
  `delete_older_than()`, `clear()` (previously listed as `store_cost()`, `get_cost()`, etc.).

## [0.12.0] - 2026-06-15

### Added

- **TealTiger governance integration** (`pip install dakera[tealtiger]`) — optional adapter
  layer for the [TealTiger](https://github.com/tealtigerlabs/tealtiger) governance framework.
  Three new classes in `dakera.integrations.tealtiger`:
  - **`DakeraCostStorage`** — cost record adapter (8 async methods). Persists cost
    records as Dakera memories under a dedicated agent namespace, enabling full cost
    lifecycle tracking across LLM calls with importance-based retention.
    Methods: `store()`, `get()`, `get_by_request_id()`, `get_by_agent_id()`,
    `get_by_date_range()`, `get_summary()`, `delete_older_than()`, `clear()`.
  - **`DakeraDecisionStore`** — receipt persistence for TealTiger governance decisions.
    Maps `Decision` objects to Dakera memories with importance derived from action type
    (DENY/TIMED_OUT=0.9, ALLOW=0.7, REQUIRE_APPROVAL=0.8). `is_terminal()` correctly
    distinguishes pending decisions (REQUIRE_APPROVAL) from resolved ones (ALLOW/DENY/
    TIMED_OUT). Methods: `store_receipt()`, `lookup_receipt()`, `list_receipts()`,
    `is_terminal()`.
  - **`DakeraDelegationHelper`** — knowledge-graph-backed delegation chain tracker.
    Stores delegation relationships as linked Dakera memories, enabling full graph
    traversal of agent delegation hierarchies. Methods: `link_delegation()`,
    `get_delegation_chain()`.
  - New example: `examples/tealtiger_governance.py` — end-to-end governance workflow
    demonstrating cost tracking, decision receipts, and delegation chains.
  - 34 new tests in `tests/test_tealtiger.py` covering all three adapters with and
    without TealTiger installed (graceful `ImportError` handling).
  - Install with: `pip install dakera[tealtiger]` (requires `tealtiger>=1.3.0`).
- **`TifScore`** — new dataclass in `dakera.models` for Truth-Indeterminacy-Falsity
  reliability scoring (T-I-F RFC Phase 3). Fields: `truth`, `indeterminacy`, `falsity`
  (all `float`, 0–1), `feedback_count` (`int`). Read-only property `classification`
  returns one of `"confident_reuse"`, `"ask_clarification"`, `"surface_contradiction"`,
  or `"verify_before_use"` based on the dominant signal.
  - `TifScore.from_feedback_history(history)` — compute T-I-F proportions from a
    `FeedbackHistoryResponse` (upvotes, downvotes, flags counted from `entries`).
    Zero-feedback history returns `TifScore(truth=0.0, indeterminacy=1.0, falsity=0.0)`.
  - `TifScore.from_metadata(data)` — parse a `metadata.reliability` dict previously
    stored via Phase 1/2 T-I-F validation scripts.
- **`DakeraClient.evaluate_tif(memory_id)`** and **`AsyncDakeraClient.evaluate_tif(memory_id)`**
  — convenience method that calls `get_memory_feedback_history()` and returns a
  `TifScore` in one step.

## [0.11.91] - 2026-06-14

### Added

- **`AsyncDakeraClient`**: 7 new async methods achieving full parity with `DakeraClient` —
  `hybrid_search()`, `batch_recall()`, `store_memories_batch()`, `autopilot_status()`,
  `autopilot_update_config()`, `autopilot_trigger()`, and `decay_stats()`. All accept the
  same arguments as their synchronous counterparts and return identical response types.
- **3 new example scripts** under `examples/`:
  - `advanced.py` — `upsert_text`, `query_text`, `fulltext_search`, `hybrid_search`,
    `memory_link` / `graph_path`, `feedback_memory`, `analytics_overview`.
  - `fulltext_search.py` — `index_documents`, `fulltext_search`, `FullTextIndexStats`,
    `fulltext_delete`.
  - `memory.py` — `store_memory`, `recall`, `search_memories`, `BatchRecallRequest`,
    `start_session`, `agent_stats`.
  - Examples CI now runs all three on every push. Note: `upsert_text` requires the
    namespace to be created with `dimensions=1024` to match the server ONNX model output.

## [0.11.90] - 2026-06-14

### Added

- **`batch_recall(request: BatchRecallRequest) -> BatchRecallResponse`** — filter-based
  memory listing by agent, tags, importance range, time window, or session id. Returns
  paginated results without a query string. Useful for listing all memories for an agent
  or inspecting session contents. Also available as `AsyncDakeraClient.batch_recall()`.
  (API: `POST /recall/batch`)
- **`hybrid_search(namespace, query, *, vector, alpha, limit) -> list[HybridSearchResult]`**
  — BM25 full-text + HNSW vector similarity search in a single call. Omit `vector` to
  let the server generate an ONNX embedding from `query`; supply it for manual re-ranking.
  `alpha` (0–1) controls BM25/vector weight blend. Also available as
  `AsyncDakeraClient.hybrid_search()`. (API: `POST /namespaces/{ns}/search/hybrid`)
- **`store_memories_batch(request: BatchStoreMemoryRequest) -> BatchStoreMemoryResponse`**
  — batch ingest of multiple memory records in one HTTP request. Response contains
  `stored`, `failed`, and per-item `errors`. Also available as
  `AsyncDakeraClient.store_memories_batch()`. (API: `POST /memories/batch`)
- **`autopilot_status() -> dict`**, **`autopilot_update_config(request) -> dict`**,
  **`autopilot_trigger(action: str) -> dict`** — read and control the server's Autopilot
  dedup/consolidation engine. `autopilot_trigger()` accepts `"dedup"` or `"consolidate"`.
  Also available as `AsyncDakeraClient` counterparts.
  (API: `GET /admin/autopilot/status`, `POST /admin/autopilot/config`,
  `POST /admin/autopilot/trigger`)
- **`decay_config() -> dict`**, **`decay_update_config(request) -> dict`**,
  **`decay_stats() -> dict`** — introspect and tune the decay engine at runtime.
  `decay_stats()` reports `memories_decayed`, `total_deleted`, `last_run_at`, and
  `cycles_completed`. Also available as `AsyncDakeraClient` counterparts.
  (API: `GET /admin/decay/config`, `POST /admin/decay/config`, `GET /admin/decay/stats`)

### Documentation

- **Quickstart README overhaul** — added a minimal 3-line quickstart at the top of the
  README so new users can reach their first `store_memory` / `recall` in under 60 seconds.

## [0.11.89] - 2026-06-11

### Changed

- **Server compatibility**: tracks Dakera server v0.11.86–v0.11.89.
  - v0.11.86: CE-OVERHAUL safe subset — RRF single-modality virtual ranking, temporal
    date-range inference, cross-session entity bridging. All engine-internal; no client
    API changes required.
  - v0.11.87: Honor cross-session `fetch_n` override in session-scoped recall path — inert
    for SDK consumers; server-side env knob only.
  - v0.11.88: Opt-in CE-31 sentence decomposition on batch ingest
    (`DAKERA_BATCH_SENTENCE_DECOMP` server env) — no client API changes.
  - v0.11.89: List-aware CE-31 decomposition + hardened supersession demotion, both
    inert-by-default server-side env flags — no client API changes.

## [0.11.85] - 2026-06-05

### Changed

- **Server compatibility**: tracks Dakera server v0.11.84–v0.11.85.
  - v0.11.84: Entity vector search for temporal queries (BM25 + entity-filtered HNSW,
    automatic routing — no client changes required); reranker queues under load instead
    of dropping to unranked results; background re-embed now upgrades all memories
    regardless of importance floor; `build_sha` added to `/health` response.
  - v0.11.85: `compute_fetch_n` extracted + `DAKERA_HYBRID_FETCH_MULT` /
    `DAKERA_CROSS_SESSION_FETCH_MULT` server-side env knobs — no client API changes.

## [0.11.83] - 2026-06-04

### Added

- **`drain_reembed()`** — new `DakeraClient` and `AsyncDakeraClient` method for
  `POST /admin/reembed/drain` (v0.11.82+). Synchronously runs the re-embedding
  upgrade loop until all `_embedding_kind=static` vectors are upgraded to full ONNX
  quality, or the optional `timeout_secs` cap is reached. Returns a
  `DrainReembedResponse` with `processed`, `remaining`, `elapsed_ms`, `cycles`, and
  `timed_out` fields. Requires Admin scope. Useful as a pre-benchmark steady-state gate
  when `DAKERA_TIERED=1`.

### Changed

- **Server compatibility**: tracks Dakera server v0.11.76–v0.11.83.
  - v0.11.76: `search_hybrid()` binary overselection formula corrected (Recall@10 restored).
  - v0.11.77: `SearchMode` default flipped to `Hybrid`; `embed_text_write()` now returns
    `is_static` flag for ReembedJob targeting.
  - v0.11.78–v0.11.79: TieredEngine pre-warm at startup; GPU inference semaphore fixes;
    batch store now routes through TieredEngine write path.
  - v0.11.80: SIMD HNSW distance calculations (3–8× throughput); ONNX memory fixes.
  - v0.11.81: `OnnxBackend` pool capped to 1 for GPU; BFCArena retry depth extended.
  - v0.11.82: Model2Vec static-write tier in production (`DAKERA_TIERED=1`);
    `/health/ready` now reports `tiered_engine: ok|degraded|disabled`.
  - v0.11.83: Deterministic HNSW build order (CE-127); raw-fs 9× write perf (DAK-6287);
    O(namespace) per-batch list removed from storage upsert (DAK-6299).
  No breaking changes to existing SDK method signatures.

## [0.11.75] - 2026-05-31

### Changed

- **Server compatibility**: tracks Dakera server v0.11.75 (TieredEngine registered in
  AppState, binary HNSW dispatch wired in search paths, ReembedJob spawned at startup).
  No client API surface changes required — all existing calls work unchanged. Binary HNSW
  is opt-in server-side via `DAKERA_SEARCH_MODE=hybrid`; the SDK sends requests identically
  regardless of server search mode.

## [0.11.57] - 2026-05-22

### Added

- **`store_memories_batch()`** — new `DakeraClient` method for `POST /v1/memories/store/batch`,
  enabling high-throughput batch memory ingestion (DAK-5508)
  - `BatchStoreMemoryItem` — per-item fields matching the server batch schema
  - `BatchStoreMemoryRequest` — `agent_id` + `List[BatchStoreMemoryItem]`
  - `BatchStoredMemory` / `BatchStoreMemoryResponse` — response types
  - Full async support via `AsyncDakeraClient.store_memories_batch()`

## [0.11.56] - 2026-05-17

### Changed

- **BREAKING: `hybrid_search` parameter renamed `alpha` → `vector_weight`** — the blending
  parameter that controls the vector-vs-BM25 weight in hybrid search has been renamed from
  `alpha` to `vector_weight` across all SDKs for consistency with `recall()`'s existing
  `vector_weight` parameter. Update all call sites:
  ```python
  # Before
  client.hybrid_search("ns", "query", vector=[...], alpha=0.7)
  # After
  client.hybrid_search("ns", "query", vector=[...], vector_weight=0.7)
  ```

### Added

- **40+ new client methods** for full engine parity (sync + async):
  - **Health probes**: `health_ready()`, `health_live()`
  - **Vector bulk ops**: `bulk_update_vectors()`, `bulk_delete_vectors()`, `count_vectors()`
  - **Agent consolidation**: `consolidate_agent()`, `get_consolidation_log()`, `patch_consolidation_config()`
  - **Namespace config**: `get_namespace_entity_config()`, `get_namespace_extractor()`
  - **Admin cluster**: `admin_cluster_replication()`, `admin_list_shards()`, `admin_rebalance_shards()`
  - **Admin maintenance**: `admin_maintenance_status()`, `admin_enable_maintenance()`, `admin_disable_maintenance()`
  - **Admin quotas**: `admin_list_quotas()`, `admin_get_default_quota()`, `admin_set_default_quota()`, `admin_get_quota()`, `admin_set_quota()`, `admin_delete_quota()`, `admin_check_quota()`
  - **Admin slow queries**: `admin_list_slow_queries()`, `admin_slow_query_summary()`, `admin_clear_slow_queries()`, `admin_update_slow_query_config()`
  - **Admin backups**: `admin_list_backups()`, `admin_create_backup()`, `admin_get_backup()`, `admin_delete_backup()`, `admin_get_backup_schedule()`, `admin_update_backup_schedule()`, `admin_restore_backup()`, `admin_get_restore_status()`
  - **Ops**: `ops_diagnostics()`, `ops_list_jobs()`, `ops_get_job()`, `ops_compact()`, `ops_shutdown()`
  - **Fulltext**: `fulltext_stats()`, `fulltext_delete()`
  - **TTL**: `ttl_stats()`
  - **Query routing**: `route_query()`
  - **Import jobs**: `import_job_status()`
  - **Backup I/O**: `download_backup()`, `upload_backup()`
  - **Storage tiers**: `storage_tier_overview()`
  - **Background activity**: `background_activity()`
  - **Memory type stats**: `memory_type_stats()`
  - **Namespace migration**: `migrate_namespace_dimensions()`
- **16 new dataclass types** in `models.py` for structured responses
- **188 unit tests** covering all SDK methods
- **6 new examples**: admin operations, analytics, fulltext search, knowledge graph, ops diagnostics, vector operations
- **Docker integration tests in CI** — full end-to-end integration tests against a live
  Dakera server container on every PR and push.

## [0.11.54] - 2026-05-13

### Notes
- Version bump to match server v0.11.54 (CE-115: INFERENCE_TEMPORAL_MULT_BETA 0.5→0.65, Cat3 +2.2pp to 73.9%). Scoring-only change — no API changes.

## [0.11.53] - 2026-05-08

### Notes
- Version bump to match server v0.11.53. Server improvements v0.11.52–v0.11.53:
  - **v0.11.53** — CE-106 entity+year co-occurrence BM25 boost for Cat2 multi-hop queries; CE-94 temporal-inference centroid tightening (12 patterns, -14.7pp Cat2 false-positive rate); distribution week1 (crate metadata, MCP registry, Docker Hub workflows).
  - **v0.11.52** — CE-86 multiplicative post-reranker temporal scaling (+2.2pp Cat3); complete recall/search metrics coverage (4 PRs).

## [0.11.51] - 2026-05-06

### Added
- **`admin_fulltext_reindex(namespace=None)`** (sync + async): backfill the BM25 fulltext
  index for memories stored before CE-12 auto-indexing (CE-54). Optional `namespace` arg;
  omit to reindex all agent namespaces. Returns `FulltextReindexResponse` with per-namespace
  breakdown (`newly_indexed`, `already_indexed`, `parse_failures`).
- **`FulltextReindexResponse`** and **`FulltextReindexNamespaceResult`** dataclasses exported
  from `dakera`.

### Notes
- Version bump to match server v0.11.51. Server improvements v0.11.47–v0.11.51:
  - **v0.11.51** — Fix flaky SEC-5 rate-limit tests (configurable window).
  - **v0.11.50** — DAK-3430 S3 retry cap (OpenDAL retry 10→3, MinIO limit 1500→6000).
  - **v0.11.49** — Dependency bumps (governor, opendal, redis, criterion).
  - **v0.11.48** — Security: openssl 0.10.78→0.10.79.
  - **v0.11.47** — ArrayContains HNSW pre-filter (SDK already exposed in v0.11.46).

## [0.11.46] - 2026-04-30

### Added
- **`F` filter builder class** (`from dakera import F`): typed helper methods that produce
  the server's filter DSL as plain dicts — IDE-autocompletable and discoverable:
  - Comparison: `F.eq`, `F.ne`, `F.gt`, `F.gte`, `F.lt`, `F.lte`, `F.in_`, `F.nin`, `F.exists`
  - String: `F.contains`, `F.icontains`, `F.starts_with`, `F.ends_with`, `F.glob`, `F.regex`
  - Array (CE-79): `F.array_contains(v)`, `F.array_contains_all([...])`, `F.array_contains_any([...])`
    — match memories whose metadata array field includes the given value(s); enables
    entity-scoped vector search (e.g. `{"tags": F.array_contains("entity:PERSON:alice")}`).
  - Logical: `F.and_(*conditions)`, `F.or_(*conditions)`

### Notes
- Version bump to match server v0.11.46. Server improvements v0.11.37–v0.11.46:
  - **CE-79 — ArrayContains filter operators**: New `$arrayContains`, `$arrayContainsAll`,
    `$arrayContainsAny` for HNSW pre-filtering on array metadata fields.
  - **CE-73 — Auto-PRF for hybrid inference queries**: Cat3 +4.2pp.
  - **CE-71 — ML query classifier**: Temporal inference detection on by default.
  - **CE-68/69/70 — Temporal boost + recency bias + S3 retry backoff**.
  - **CE-58 — Configurable RRF k-parameter** (`DAKERA_RRF_K` env var).

## [0.11.36] - 2026-04-26

### Notes
- Version bump to match server v0.11.36. No SDK API changes.
- Server improvements v0.11.32–v0.11.36 (all transparent to SDK callers):
  - **CE-53 — BM25 session pre-filter**: BM25 full-text candidates constrained to the
    active `session_id` before cross-encoder ranking, closing the symmetry gap with HNSW
    session pre-filter (CE-52). Session-scoped queries no longer bleed cross-session results.
  - **CE-53 — fetch_n 20×→5×**: Cross-encoder candidate workload cut by 4×, eliminating
    408 timeouts on high-memory conversations (1200+ memories). Full 1540Q bench: **82.4%
    overall** (Cat1 80.1%, Cat2 85.7%, Cat3 55.2%, Cat4 85.0%).
  - **CE-52 — Session HNSW pre-filter**: HNSW ANN search pre-filtered by `session_id`
    for multi-session namespaces, eliminating cross-session bleed at scale.
  - **CE-51 — Entity-prioritized PRF term extraction**: Hybrid PRF now prioritises
    entity tokens during pseudo-relevance feedback expansion.
  - **CE-49 — Hybrid PRF honors `iterations`**: `iterations` param now correctly applied
    in Hybrid routing mode (was silently ignored in some PRF paths).
  - **CE-33 — HNSW cache invalidation**: All write endpoints (store, update, delete,
    consolidate, feedback) now invalidate the cached HNSW index, preventing stale search
    results during high-throughput ingestion.
  - **Parallel S3/Minio reads**: `ObjectStorage::get_all()` uses `buffer_unordered(32)` —
    ~32× throughput improvement for bulk reads, fixing recall timeouts at 1000+ memories.

## [0.11.31] - 2026-04-25

### Notes
- Version bump to match server v0.11.31. No SDK API changes.
- Server improvements (all transparent to SDK callers):
  - **CE-48 — BM25 English stemming for new fulltext indices**: All new fulltext indices
    now use Snowball English stemmer at both index and query time. Morphological variants
    (e.g. "running"→"run", "memories"→"memori") are normalized, increasing BM25 term
    overlap. Only affects NEW indices — persisted indices retain their original config.
    Expect +3–5pp on Cat1 (factual) and Cat4 (multi-hop) queries.

## [0.11.30] - 2026-04-25

### Notes
- Version bump to match server v0.11.30. No SDK API changes.
- Server improvements since v0.11.4 (all transparent to SDK callers):
  - **CE-48 — Hybrid PRF for inference queries (Cat3 +24pp)**: Pseudo-relevance
    feedback now applied to `routing="auto"` Hybrid queries classified as temporal/inference.
    Pass-1 Hybrid results seed a BM25 expansion pass; RRF-merged (k=60). Gated behind
    `QueryClassifier::Temporal` to prevent Cat1 regression.
  - **CE-47a — Cross-encoder reranking for BM25 temporal queries**: Cross-encoder reranker
    now fires on temporal BM25 queries (was previously skipped for BM25 paths), correcting
    BM25 rank-order errors caused by date-prefixed memories.
  - **CE-43/39/35 — Temporal PRF hardening**: Auto-PRF (iterations=2) applied server-side
    for all temporal BM25 queries. Pass-1 pool widened to 40 candidates. Date-window
    narrowing (±90 days from anchor date) applied to pass-2 BM25.
  - **CE-34 v2 — Tighter MultiHop classifier**: Structural-context guards on pronoun-after-
    sequential-marker patterns protect Cat2 multi-hop queries from misrouting.
  - **CE-31 — Sentence decomposition at store**: Content ≥80 chars is split into up to 5
    atomic sentences, each embedded and indexed independently as sibling memories. Individual
    facts become independently retrievable without scoring the full parent blob.
  - **SEC-3 hardening (v0.11.30)**: Empty or short encryption passphrases are now rejected
    at the API boundary (NIST 800-63B). Affects `rotate_encryption_key()` callers — supply
    a passphrase ≥ 8 chars or a full 64-hex raw key.
  - **Security (v0.11.29)**: Server dep bumps: rustls-webpki 0.103.13 (RUSTSEC-2026-0104),
    rand 0.9.1 (RUSTSEC-2026-0097). No SDK impact.

## [0.11.4] - 2026-04-18

### Added
- **CE-23 — PRF iterative BM25 `iterations` param**: `recall()` and `async_client.recall()`
  now accept an optional `iterations: int | None` parameter (1–3, default: 1). Pass `2` or `3`
  for multi-hop or temporal queries to enable server-side pseudo-relevance feedback (PRF):
  a second BM25 pass over entities extracted from the first pass improves recall on
  evidence-chain queries. Only effective when `routing=RoutingMode.BM25`; omitting the
  parameter preserves single-pass behaviour — zero breaking changes.
  (server: [#175](https://github.com/Dakera-AI/dakera/pull/175))

## [0.11.3] - 2026-04-18

### Added
- **CE-17 — Explicit `vector_weight` for Hybrid recall**: `recall()` and `async_client.recall()`
  now accept an optional `vector_weight: float | None` parameter (0.0–1.0). When set, overrides
  the server's adaptive vector/BM25 heuristic for `routing=RoutingMode.HYBRID` calls, giving
  callers per-query control over retrieval balance. Omitting the parameter preserves existing
  adaptive behaviour — zero breaking changes.
  (server: [#173](https://github.com/Dakera-AI/dakera/pull/173))

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
- **CE-13:** `rerank` parameter on `DakeraClient.recall()` and `DakeraClient.search_memories()` (and async equivalents). Enables cross-encoder reranking via `bge-reranker-v2-m3` (upgraded from `bge-reranker-base` in server v0.11.95 CE-C). `None` uses server default (`True` for recall, `False` for search). Pass `False` to disable on latency-sensitive paths.
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
