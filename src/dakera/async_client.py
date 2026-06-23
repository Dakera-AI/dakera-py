"""
Dakera Async Client

Async client for interacting with Dakera server using httpx.

Requires the optional `async` extra:
    pip install dakera[async]

Example:
    >>> import asyncio
    >>> from dakera import AsyncDakeraClient
    >>>
    >>> async def main():
    ...     async with AsyncDakeraClient("http://localhost:3000") as client:
    ...         await client.upsert("my-ns", vectors=[
    ...             {"id": "v1", "values": [0.1, 0.2, 0.3]},
    ...         ])
    ...         results = await client.query("my-ns", vector=[0.1, 0.2, 0.3])
    ...         for r in results.results:
    ...             print(f"{r.id}: {r.score}")
    >>>
    >>> asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncGenerator
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "httpx is required for AsyncDakeraClient. Install it with: pip install dakera[async]"
    ) from exc

from dakera.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    DakeraError,
    ErrorCode,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from dakera.models import (
    AccessPatternHint,
    AgentFeedbackSummary,
    # OBS-1
    AuditExportResponse,
    AuditListResponse,
    BatchForgetRequest,
    BatchForgetResponse,
    BatchRecallRequest,
    BatchRecallResponse,
    BatchStoreMemoryRequest,
    BatchStoreMemoryResponse,
    BatchTextQueryResponse,
    # CE-12
    CompressResponse,
    ConfigureNamespaceRequest,
    ConfigureNamespaceResponse,
    # CE-6
    ConsolidationConfig,
    CreateNamespaceKeyResponse,
    CrossAgentNetworkResponse,
    DakeraEvent,
    DistanceMetric,
    Document,
    DocumentInput,
    DrainReembedResponse,
    EdgeType,
    EmbeddingModel,
    EntityExtractionResponse,
    # ODE-2
    ExtractEntitiesResponse,
    ExtractionProviderInfo,
    # EXT-1
    ExtractionResult,
    FeedbackHealthResponse,
    FeedbackHistoryResponse,
    FeedbackResponse,
    FeedbackSignal,
    FilterDict,
    FullTextIndexStats,
    # CE-54
    FulltextReindexResponse,
    FullTextSearchResult,
    # CE-14
    FusionStrategy,
    GraphExport,
    GraphLinkResponse,
    GraphPath,
    HybridSearchResult,
    ImportJobStatus,
    IndexStats,
    # KG-2
    KgExportResponse,
    KgPathResponse,
    KgQueryResponse,
    # OBS-2
    KpiSnapshot,
    ListNamespaceKeysResponse,
    MemoryEntitiesResponse,
    MemoryEvent,
    MemoryExportResponse,
    MemoryGraph,
    # DX-1
    MemoryImportResponse,
    # COG-1
    MemoryPolicy,
    MemoryTypeStatsResponse,
    MigrateDimensionsResponse,
    NamespaceInfo,
    NamespaceKeyUsageResponse,
    NamespaceNerConfig,
    RateLimitHeaders,
    ReadConsistency,
    # COG-2
    RecallResponse,
    RetryConfig,
    # SEC-3
    RotateEncryptionKeyResponse,
    RouteResponse,
    # CE-10
    RoutingMode,
    SearchResult,
    StalenessConfig,
    StaticCountResponse,
    StorageTierOverview,
    TextDocument,
    TextDocumentInput,
    TextQueryResponse,
    TextUpsertResponse,
    TifScore,
    TtlCleanupResponse,
    TtlStatsResponse,
    Vector,
    VectorInput,
    WakeUpResponse,
    WarmCacheRequest,
    WarmCacheResponse,
    WarmingPriority,
    WarmingTargetTier,
)


class AsyncDakeraClient:
    """
    Async client for Dakera AI memory platform.

    Uses httpx for non-blocking HTTP requests. All methods are coroutines.
    Supports async context manager protocol.

    Example:
        >>> async with AsyncDakeraClient("http://localhost:3000") as client:
        ...     results = await client.query("ns", vector=[0.1, 0.2, 0.3])
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        connect_timeout: float | None = None,
        max_retries: int = 3,
        retry_config: RetryConfig | None = None,
        headers: dict[str, str] | None = None,
        ode_url: str | None = None,
    ) -> None:
        """
        Initialize async Dakera client.

        Args:
            base_url: Base URL of the Dakera server (e.g., "http://localhost:3000")
            api_key: Optional API key for authentication
            timeout: Per-request timeout in seconds (default: 30.0)
            connect_timeout: Connection establishment timeout in seconds.
                Defaults to ``timeout`` when not set.
            max_retries: Maximum number of retries for transient errors (default: 3).
                Ignored when ``retry_config`` is provided.
            retry_config: Fine-grained retry configuration.  When provided,
                ``max_retries`` is ignored in favour of
                ``retry_config.max_retries``.
            headers: Additional headers to include in all requests
            ode_url: Base URL of the dakera-ode sidecar
                (e.g., ``"http://localhost:8080"``).  Required to call
                :meth:`extract_entities`.
        """
        self.base_url = base_url.rstrip("/")
        self.ode_url = ode_url.rstrip("/") if ode_url else None
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = connect_timeout if connect_timeout is not None else timeout

        if retry_config is not None:
            self._retry_config = retry_config
        else:
            self._retry_config = RetryConfig(max_retries=max_retries)

        default_headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            default_headers.update(headers)

        self._client = httpx.AsyncClient(
            headers=default_headers,
            timeout=httpx.Timeout(timeout, connect=self.connect_timeout),
        )

        # OPS-1: last seen rate-limit headers (updated after every response)
        self._last_rate_limit_headers: RateLimitHeaders | None = None

    @property
    def last_rate_limit_headers(self) -> RateLimitHeaders | None:
        """Rate-limit headers from the most recent API response (OPS-1).

        Returns ``None`` until the first successful request has been made.
        """
        return self._last_rate_limit_headers

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        return f"{self.base_url}/{path.lstrip('/')}"

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle API response and raise appropriate exceptions."""
        # OPS-1: capture rate-limit headers before consuming the body
        self._last_rate_limit_headers = RateLimitHeaders.from_headers(dict(response.headers))

        try:
            body = response.json() if response.content else None
        except json.JSONDecodeError:
            body = response.text

        if response.status_code in (200, 201):
            return body
        if response.status_code == 204:
            return None

        raw_code = body.get("code") if isinstance(body, dict) else None
        try:
            error_code = ErrorCode(raw_code) if raw_code is not None else ErrorCode.UNKNOWN
        except ValueError:
            error_code = ErrorCode.UNKNOWN

        if response.status_code == 400:
            raise ValidationError(
                message=(
                    body.get("error", "Validation error") if isinstance(body, dict) else str(body)
                ),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        if response.status_code == 401:
            raise AuthenticationError(
                message=(
                    body.get("error", "Authentication failed")
                    if isinstance(body, dict)
                    else "Authentication failed"
                ),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        if response.status_code == 403:
            raise AuthorizationError(
                message=(body.get("error", "Forbidden") if isinstance(body, dict) else "Forbidden"),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        if response.status_code == 404:
            raise NotFoundError(
                message=(
                    body.get("error", "Resource not found") if isinstance(body, dict) else str(body)
                ),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message="Rate limit exceeded",
                status_code=response.status_code,
                response_body=body,
                retry_after=int(retry_after) if retry_after else None,
            )
        if response.status_code >= 500:
            raise ServerError(
                message=body.get("error", "Server error") if isinstance(body, dict) else str(body),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        raise DakeraError(
            message=f"Unexpected status code: {response.status_code}",
            status_code=response.status_code,
            response_body=body,
            code=error_code,
        )

    @staticmethod
    def _compute_backoff(rc: RetryConfig, attempt: int) -> float:
        """Compute exponential backoff delay for the given attempt index."""
        delay = min(rc.max_delay, rc.base_delay * (2**attempt))
        if rc.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay

    async def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make async HTTP request with retry logic and exponential backoff."""
        url = self._url(path)
        rc = self._retry_config

        for attempt in range(rc.max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                )
                return self._handle_response(response)
            except httpx.ConnectError as e:
                if attempt == rc.max_retries - 1:
                    raise ConnectionError(f"Failed to connect to {url}: {e}") from e
            except httpx.TimeoutException as e:
                if attempt == rc.max_retries - 1:
                    raise TimeoutError(f"Request timed out: {e}") from e
            except RateLimitError as e:
                if attempt == rc.max_retries - 1:
                    raise
                wait = (
                    float(e.retry_after)
                    if e.retry_after is not None
                    else self._compute_backoff(rc, attempt)
                )
                await asyncio.sleep(wait)
                continue
            except ServerError:
                if attempt == rc.max_retries - 1:
                    raise
            except DakeraError:
                raise

            await asyncio.sleep(self._compute_backoff(rc, attempt))

        raise DakeraError("Request failed after retries")

    # =========================================================================
    # Vector Operations
    # =========================================================================

    async def upsert(
        self,
        namespace: str,
        vectors: list[VectorInput],
    ) -> dict[str, Any]:
        """Upsert vectors into a namespace."""
        vector_dicts = []
        for v in vectors:
            if isinstance(v, Vector):
                vector_dicts.append(v.to_dict())
            else:
                vector_dicts.append(v)
        return await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/vectors",
            data={"vectors": vector_dicts},
        )

    async def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int = 10,
        filter: FilterDict | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
        distance_metric: DistanceMetric | None = None,
        consistency: ReadConsistency | None = None,
        staleness_config: StalenessConfig | None = None,
    ) -> SearchResult:
        """Query vectors by similarity."""
        data: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_values": include_values,
            "include_metadata": include_metadata,
        }
        if filter:
            data["filter"] = filter
        if distance_metric:
            data["distance_metric"] = distance_metric.value
        if consistency:
            data["consistency"] = consistency.value
        if staleness_config:
            data["staleness_config"] = staleness_config.to_dict()
        response = await self._request("POST", f"/v1/namespaces/{namespace}/query", data=data)
        return SearchResult.from_dict(response)

    async def delete(
        self,
        namespace: str,
        ids: list[str] | None = None,
        filter: FilterDict | None = None,
        delete_all: bool = False,
    ) -> dict[str, Any]:
        """Delete vectors from a namespace."""
        data: dict[str, Any] = {}
        if ids:
            data["ids"] = ids
        if filter:
            data["filter"] = filter
        if delete_all:
            data["delete_all"] = True
        return await self._request("POST", f"/v1/namespaces/{namespace}/delete", data=data)

    async def bulk_update_vectors(
        self,
        namespace: str,
        filter: FilterDict,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Bulk update vector metadata matching a filter."""
        return await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/vectors/bulk-update",
            data={"filter": filter, "update": update},
        )

    async def bulk_delete_vectors(
        self,
        namespace: str,
        filter: FilterDict,
    ) -> dict[str, Any]:
        """Bulk delete vectors matching a filter."""
        return await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/vectors/bulk-delete",
            data={"filter": filter},
        )

    async def count_vectors(
        self,
        namespace: str,
        filter: FilterDict | None = None,
    ) -> dict[str, Any]:
        """Count vectors in a namespace, optionally filtered."""
        data: dict[str, Any] = {}
        if filter is not None:
            data["filter"] = filter
        return await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/vectors/count",
            data=data,
        )

    async def fetch(
        self,
        namespace: str,
        ids: list[str],
        include_values: bool = True,
        include_metadata: bool = True,
    ) -> list[Vector]:
        """Fetch vectors by ID."""
        response = await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/fetch",
            data={
                "ids": ids,
                "include_values": include_values,
                "include_metadata": include_metadata,
            },
        )
        return [Vector.from_dict(v) for v in response.get("vectors", [])]

    async def batch_query(
        self,
        namespace: str,
        queries: list[dict[str, Any]],
    ) -> list[SearchResult]:
        """Execute multiple queries in a single request."""
        response = await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/batch-query",
            data={"queries": queries},
        )
        return [SearchResult.from_dict(r) for r in response.get("results", [])]

    # =========================================================================
    # Text-Based Inference Operations (Auto-Embedding)
    # =========================================================================

    async def upsert_text(
        self,
        namespace: str,
        documents: list[TextDocumentInput],
        model: EmbeddingModel | None = None,
    ) -> TextUpsertResponse:
        """Upsert text documents with automatic embedding generation."""
        doc_dicts = [d.to_dict() if isinstance(d, TextDocument) else d for d in documents]
        data: dict[str, Any] = {"documents": doc_dicts}
        if model:
            data["model"] = model.value
        response = await self._request("POST", f"/v1/namespaces/{namespace}/upsert-text", data=data)
        return TextUpsertResponse.from_dict(response)

    async def query_text(
        self,
        namespace: str,
        text: str,
        top_k: int = 10,
        filter: FilterDict | None = None,
        include_text: bool = True,
        include_vectors: bool = False,
        model: EmbeddingModel | None = None,
    ) -> TextQueryResponse:
        """Query using natural language text with automatic embedding."""
        data: dict[str, Any] = {
            "text": text,
            "top_k": top_k,
            "include_text": include_text,
            "include_vectors": include_vectors,
        }
        if filter:
            data["filter"] = filter
        if model:
            data["model"] = model.value
        response = await self._request("POST", f"/v1/namespaces/{namespace}/query-text", data=data)
        return TextQueryResponse.from_dict(response)

    async def batch_query_text(
        self,
        namespace: str,
        queries: list[str],
        top_k: int = 10,
        filter: FilterDict | None = None,
        include_vectors: bool = False,
        model: EmbeddingModel | None = None,
    ) -> BatchTextQueryResponse:
        """Batch query using multiple text queries with automatic embedding."""
        data: dict[str, Any] = {
            "queries": queries,
            "top_k": top_k,
            "include_vectors": include_vectors,
        }
        if filter:
            data["filter"] = filter
        if model:
            data["model"] = model.value
        response = await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/batch-query-text",
            data=data,
        )
        return BatchTextQueryResponse.from_dict(response)

    # =========================================================================
    # Full-Text Search Operations
    # =========================================================================

    async def index_documents(
        self,
        namespace: str,
        documents: list[DocumentInput],
    ) -> dict[str, Any]:
        """Index documents for full-text search."""
        doc_dicts = [d.to_dict() if isinstance(d, Document) else d for d in documents]
        return await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/fulltext/index",
            data={"documents": doc_dicts},
        )

    async def fulltext_search(
        self,
        namespace: str,
        query: str,
        top_k: int = 10,
        filter: FilterDict | None = None,
    ) -> list[FullTextSearchResult]:
        """Perform full-text search."""
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if filter:
            data["filter"] = filter
        response = await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/fulltext/search",
            data=data,
        )
        return [FullTextSearchResult.from_dict(r) for r in response.get("results", [])]

    async def hybrid_search(
        self,
        namespace: str,
        vector: list[float],
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        filter: FilterDict | None = None,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search combining vector and full-text."""
        data: dict[str, Any] = {
            "vector": vector,
            "text": query,
            "top_k": top_k,
            "vector_weight": vector_weight,
        }
        if filter:
            data["filter"] = filter
        response = await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/hybrid",
            data=data,
        )
        return [HybridSearchResult.from_dict(r) for r in response.get("results", [])]

    # =========================================================================
    # Namespace Operations
    # =========================================================================

    async def list_namespaces(self) -> list[NamespaceInfo]:
        """List all namespaces."""
        response = await self._request("GET", "/v1/namespaces")
        return [NamespaceInfo.from_dict(ns) for ns in response.get("namespaces", [])]

    async def get_namespace(self, namespace: str) -> NamespaceInfo:
        """Get namespace information."""
        response = await self._request("GET", f"/v1/namespaces/{namespace}")
        return NamespaceInfo.from_dict(response)

    async def create_namespace(
        self,
        namespace: str,
        dimensions: int | None = None,
        index_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NamespaceInfo:
        """Create a new namespace."""
        data: dict[str, Any] = {"name": namespace}
        if dimensions:
            data["dimensions"] = dimensions
        if index_type:
            data["index_type"] = index_type
        if metadata:
            data["metadata"] = metadata
        response = await self._request("POST", "/v1/namespaces", data=data)
        return NamespaceInfo.from_dict(response)

    async def configure_namespace(
        self,
        namespace: str,
        dimension: int,
        distance: DistanceMetric | None = None,
    ) -> ConfigureNamespaceResponse:
        """Create or update a namespace configuration (upsert semantics).

        Creates the namespace if it does not exist, or updates its distance
        metric configuration if it already exists.  Requires ``Scope::Write``.

        Args:
            namespace: Namespace name
            dimension: Vector dimension. Must match existing dimension on updates.
            distance: Distance metric (default: cosine).

        Returns:
            ConfigureNamespaceResponse with ``created=True`` if newly created.
        """
        req = ConfigureNamespaceRequest(dimension=dimension, distance=distance)
        response = await self._request("PUT", f"/v1/namespaces/{namespace}", data=req.to_dict())
        return ConfigureNamespaceResponse.from_dict(response)

    async def delete_namespace(self, namespace: str) -> None:
        """Delete a namespace and all its data."""
        await self._request("DELETE", f"/v1/namespaces/{namespace}")

    # =========================================================================
    # Admin / Stats Operations
    # =========================================================================

    async def health(self) -> dict[str, Any]:
        """Check server health status."""
        return await self._request("GET", "/health")

    async def health_ready(self) -> dict[str, Any]:
        """K8s readiness probe — checks storage and dependencies."""
        return await self._request("GET", "/health/ready")

    async def health_live(self) -> dict[str, Any]:
        """K8s liveness probe — checks process is alive."""
        return await self._request("GET", "/health/live")

    async def get_index_stats(self, namespace: str) -> IndexStats:
        """Get index statistics for a namespace."""
        response = await self._request("GET", f"/v1/namespaces/{namespace}/stats")
        return IndexStats.from_dict(response)

    async def compact(self, namespace: str) -> dict[str, Any]:
        """Trigger compaction for a namespace."""
        return await self._request("POST", f"/v1/namespaces/{namespace}/compact")

    async def flush(self, namespace: str) -> dict[str, Any]:
        """Flush pending writes for a namespace."""
        return await self._request("POST", f"/v1/namespaces/{namespace}/flush")

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def store_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "episodic",
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        ttl_seconds: int | None = None,
        expires_at: int | None = None,
    ) -> dict[str, Any]:
        """Store a memory for an agent.

        Args:
            agent_id: Agent identifier.
            content: Memory content text.
            memory_type: One of ``"episodic"``, ``"semantic"``, ``"procedural"``,
                or ``"working"``.
            importance: Importance score 0.0–1.0.
            metadata: Arbitrary metadata dictionary.
            session_id: Optional session ID to associate with.
            tags: Optional list of tags to associate with the memory.
            ttl_seconds: Optional TTL in seconds. The memory is hard-deleted
                after this many seconds from creation.
            expires_at: Optional explicit expiry as a Unix timestamp (seconds).
                Takes precedence over ``ttl_seconds`` when both are provided.
        """
        data: dict[str, Any] = {"content": content, "memory_type": memory_type}
        if importance is not None:
            data["importance"] = importance
        if metadata is not None:
            data["metadata"] = metadata
        if session_id is not None:
            data["session_id"] = session_id
        if tags is not None:
            data["tags"] = tags
        if ttl_seconds is not None:
            data["ttl_seconds"] = ttl_seconds
        if expires_at is not None:
            data["expires_at"] = expires_at
        data["agent_id"] = agent_id
        response = await self._request("POST", "/v1/memory/store", data=data)
        # Server wraps the memory in {"memory": {...}, "embedding_time_ms": ...}
        if isinstance(response, dict) and "memory" in response:
            return response["memory"]
        return response

    async def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None,
        min_importance: float | None = None,
        include_associated: bool = False,
        associated_memories_cap: int | None = None,
        associated_memories_depth: int | None = None,
        associated_memories_min_weight: float | None = None,
        since: str | None = None,
        until: str | None = None,
        routing: RoutingMode | str | None = None,
        rerank: bool | None = None,
        fusion: FusionStrategy | str | None = None,
        vector_weight: float | None = None,
        iterations: int | None = None,
        neighborhood: bool | None = None,
    ) -> RecallResponse:
        """Recall memories for an agent.

        Args:
            agent_id: The agent whose memories to recall.
            query: Semantic query text.
            top_k: Number of primary results to return (default: 5).
            memory_type: Filter by memory type.
            min_importance: Minimum importance threshold.
            include_associated: COG-2 — traverse KG from recalled memories
                and include associatively linked memories in
                ``associated_memories`` (default: False).
            associated_memories_cap: COG-2 — max associated memories to
                return (default: 10, max: 10).
            associated_memories_depth: KG-3 — traversal depth 1–3
                (default: 1).  Requires ``include_associated=True``.
            associated_memories_min_weight: KG-3 — minimum edge weight for
                KG traversal (default: 0.0).
            since: CE-7 — only recall memories created at or after this
                ISO-8601 timestamp (e.g. ``"2026-03-01T00:00:00Z"``).
            until: CE-7 — only recall memories created at or before this
                ISO-8601 timestamp (e.g. ``"2026-03-31T23:59:59Z"``).
            rerank: CE-13 — run cross-encoder reranking on ANN candidates
                (default: None = server default of ``True``). Pass
                ``False`` to disable for latency-sensitive paths.
            fusion: CE-14 — fusion strategy when routing=hybrid.
                ``FusionStrategy.MINMAX`` (server default since v0.11.2) uses
                min-max score normalization; ``FusionStrategy.RRF`` uses
                Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).
            vector_weight: CE-17 — explicit vector/BM25 weight for Hybrid
                routing (0.0–1.0). When set, overrides the adaptive heuristic
                from ``QueryClassifier``; omit for adaptive defaults
                (recommended for most callers). Only effective when
                ``routing=RoutingMode.HYBRID``.
            iterations: CE-23 — pseudo-relevance feedback (PRF) passes for
                BM25 routing (1–3, default: 1). Pass ``2`` or ``3`` for
                multi-hop or temporal queries where a second BM25 pass over
                extracted entities improves recall. Only effective when
                ``routing=RoutingMode.BM25``.
            neighborhood: v0.11.0 — fetch session-adjacent memories within
                ±5 min of each top result as context enrichment (default:
                None = server default of ``True``). Pass ``False`` to
                disable for latency-sensitive paths.

        Returns:
            :class:`RecallResponse` with ``memories`` and optionally
            ``associated_memories`` when ``include_associated`` is True.
            Each associated memory includes a ``depth`` field (KG-3).
        """
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        if include_associated:
            data["include_associated"] = True
        if associated_memories_cap is not None:
            data["associated_memories_cap"] = associated_memories_cap
        if associated_memories_depth is not None:
            data["associated_memories_depth"] = associated_memories_depth
        if associated_memories_min_weight is not None:
            data["associated_memories_min_weight"] = associated_memories_min_weight
        if since is not None:
            data["since"] = since
        if until is not None:
            data["until"] = until
        if routing is not None:
            data["routing"] = routing.value if hasattr(routing, "value") else routing
        if rerank is not None:
            data["rerank"] = rerank
        if fusion is not None:
            data["fusion"] = fusion.value if hasattr(fusion, "value") else fusion
        if vector_weight is not None:
            data["vector_weight"] = vector_weight
        if iterations is not None:
            data["iterations"] = iterations
        if neighborhood is not None:
            data["neighborhood"] = neighborhood
        data["agent_id"] = agent_id
        result = await self._request("POST", "/v1/memory/recall", data=data)
        if isinstance(result, dict):
            return RecallResponse.from_dict(result)
        return RecallResponse(memories=result)

    async def get_memory(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Get a specific memory."""
        return await self._request("GET", f"/v1/memory/get/{memory_id}")

    async def update_memory(
        self,
        agent_id: str,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing memory."""
        data: dict[str, Any] = {}
        if content is not None:
            data["content"] = content
        if metadata is not None:
            data["metadata"] = metadata
        if memory_type is not None:
            data["memory_type"] = memory_type
        return await self._request("PUT", f"/v1/memory/update/{memory_id}", data=data)

    async def forget(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Delete a memory."""
        data = {"agent_id": agent_id, "memory_ids": [memory_id]}
        return await self._request("POST", "/v1/memory/forget", data=data)

    async def batch_recall(self, request: BatchRecallRequest) -> BatchRecallResponse:
        """Bulk-recall memories using filter predicates (CE-2).

        Uses ``POST /v1/memories/recall/batch`` — no embedding required.

        Example:
            >>> filt = BatchMemoryFilter(tags=["sdk-lead"], min_importance=0.7)
            >>> resp = await client.batch_recall(BatchRecallRequest("agent-1", filter=filt))
            >>> print(f"Found {resp.filtered} memories")
        """
        result = await self._request("POST", "/v1/memories/recall/batch", data=request.to_dict())
        return BatchRecallResponse.from_dict(result)

    async def batch_forget(self, request: BatchForgetRequest) -> BatchForgetResponse:
        """Bulk-delete memories using filter predicates (CE-2).

        Uses ``DELETE /v1/memories/forget/batch``.  At least one filter
        predicate must be set (server safety guard).

        Example:
            >>> filt = BatchMemoryFilter(created_before=1700000000)
            >>> resp = await client.batch_forget(BatchForgetRequest("agent-1", filter=filt))
            >>> print(f"Deleted {resp.deleted_count} memories")
        """
        result = await self._request("DELETE", "/v1/memories/forget/batch", data=request.to_dict())
        return BatchForgetResponse.from_dict(result)

    async def store_memories_batch(
        self, request: BatchStoreMemoryRequest
    ) -> BatchStoreMemoryResponse:
        """Store multiple memories in a single request (DAK-5508).

        Uses ``POST /v1/memories/store/batch``. The server embeds all contents
        in a single ONNX inference pass, yielding ≥100× throughput vs. N
        sequential single-store calls. Accepts up to 1 000 memories per call.

        Args:
            request: Batch store request containing ``agent_id`` and list of
                :class:`BatchStoreMemoryItem` (1–1000 items).

        Returns:
            :class:`BatchStoreMemoryResponse` with stored memories and timing.

        Example:
            >>> items = [
            ...     BatchStoreMemoryItem("The user prefers dark mode", importance=0.8),
            ...     BatchStoreMemoryItem("The user is based in Berlin", importance=0.7),
            ... ]
            >>> resp = await client.store_memories_batch(BatchStoreMemoryRequest("agent-1", items))
            >>> print(f"Stored {resp.stored_count} memories")
        """
        result = await self._request("POST", "/v1/memories/store/batch", data=request.to_dict())
        return BatchStoreMemoryResponse.from_dict(result)

    async def search_memories(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
        min_importance: float | None = None,
        routing: RoutingMode | str | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories for an agent."""
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        if routing is not None:
            data["routing"] = routing.value if hasattr(routing, "value") else routing
        if rerank is not None:
            data["rerank"] = rerank
        data["agent_id"] = agent_id
        result = await self._request("POST", "/v1/memory/search", data=data)
        items = result.get("memories", result) if isinstance(result, dict) else result
        if isinstance(items, list):
            return [
                {**item["memory"], "score": item.get("score")}
                if isinstance(item, dict) and isinstance(item.get("memory"), dict)
                else item
                for item in items
            ]
        return items

    async def compress_agent(self, agent_id: str) -> CompressResponse:
        """Compress the memory namespace for an agent (CE-12)."""
        result = await self._request("POST", f"/v1/agents/{agent_id}/compress")
        return CompressResponse.from_dict(result)

    async def update_importance(
        self,
        agent_id: str,
        memory_ids: list[str],
        importance: float,
    ) -> dict[str, Any]:
        """Update importance of memories."""
        return await self._request(
            "POST",
            "/v1/memory/importance",
            data={"agent_id": agent_id, "memory_ids": memory_ids, "importance": importance},
        )

    async def consolidate(
        self,
        agent_id: str,
        memory_type: str | None = None,
        threshold: float | None = None,
        dry_run: bool = False,
        config: ConsolidationConfig | None = None,
    ) -> dict[str, Any]:
        """Consolidate memories for an agent (CE-6).

        Args:
            agent_id: Agent whose memories to consolidate.
            memory_type: Optional filter — only consolidate this memory type.
            threshold: Similarity threshold for grouping (0–1).
            dry_run: Preview changes without applying them.
            config: Optional :class:`~dakera.ConsolidationConfig` to select the
                clustering algorithm and tune its parameters.

        Returns:
            Dict with ``consolidated_count``, ``removed_count``, ``new_memories``
            and optionally a ``log`` list of consolidation steps.
        """
        data: dict[str, Any] = {"dry_run": dry_run}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if threshold is not None:
            data["threshold"] = threshold
        if config is not None:
            data["config"] = config.to_dict()
        data["agent_id"] = agent_id
        return await self._request("POST", "/v1/memory/consolidate", data=data)

    async def consolidate_agent(self, agent_id: str) -> dict[str, Any]:
        """Consolidate memories directly for an agent (DBSCAN clustering)."""
        return await self._request("POST", f"/v1/agents/{agent_id}/consolidate")

    async def get_consolidation_log(self, agent_id: str) -> list[dict[str, Any]]:
        """Get the consolidation execution log for an agent."""
        return await self._request("GET", f"/v1/agents/{agent_id}/consolidation/log")

    async def patch_consolidation_config(
        self,
        agent_id: str,
        enabled: bool | None = None,
        epsilon: float | None = None,
        min_samples: int | None = None,
        soft_deprecation_days: int | None = None,
    ) -> dict[str, Any]:
        """Update the consolidation configuration for an agent."""
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        if epsilon is not None:
            data["epsilon"] = epsilon
        if min_samples is not None:
            data["min_samples"] = min_samples
        if soft_deprecation_days is not None:
            data["soft_deprecation_days"] = soft_deprecation_days
        return await self._request(
            "PATCH", f"/v1/agents/{agent_id}/consolidation/config", data=data
        )

    async def memory_feedback(
        self,
        agent_id: str,
        memory_id: str,
        feedback: str,
        relevance_score: float | None = None,
    ) -> dict[str, Any]:
        """Submit feedback on a memory recall."""
        data: dict[str, Any] = {"memory_id": memory_id, "feedback": feedback}
        if relevance_score is not None:
            data["relevance_score"] = relevance_score
        return await self._request("POST", f"/v1/agents/{agent_id}/memories/feedback", data=data)

    # =========================================================================
    # Memory Feedback Loop — INT-1
    # =========================================================================

    async def feedback_memory(
        self,
        memory_id: str,
        agent_id: str,
        signal: FeedbackSignal | str,
    ) -> FeedbackResponse:
        """Submit upvote/downvote/flag feedback on a memory (INT-1).

        Args:
            memory_id: The memory to give feedback on.
            agent_id: The agent that owns the memory.
            signal: :class:`FeedbackSignal` value — ``upvote``, ``downvote``, or ``flag``.

        Returns:
            :class:`FeedbackResponse` with the updated importance and applied signal.
        """
        data: dict[str, Any] = {
            "agent_id": agent_id,
            "signal": signal.value if isinstance(signal, FeedbackSignal) else signal,
        }
        result = await self._request("POST", f"/v1/memories/{memory_id}/feedback", data=data)
        return FeedbackResponse.from_dict(result)

    async def get_memory_feedback_history(self, memory_id: str) -> FeedbackHistoryResponse:
        """Get the full feedback history for a memory (INT-1).

        Args:
            memory_id: The memory whose feedback history to retrieve.

        Returns:
            :class:`FeedbackHistoryResponse` with ordered list of feedback events.
        """
        result = await self._request("GET", f"/v1/memories/{memory_id}/feedback")
        return FeedbackHistoryResponse.from_dict(result)

    async def evaluate_tif(self, memory_id: str) -> TifScore:
        """Compute a T-I-F reliability score for a memory (T-I-F RFC Phase 3).

        Fetches the memory's full feedback history and reduces it to a
        :class:`TifScore` with truth/indeterminacy/falsity proportions and a
        human-readable :attr:`~TifScore.classification`.

        Args:
            memory_id: The memory to score.

        Returns:
            :class:`TifScore` derived from the memory's feedback history.
        """
        history = await self.get_memory_feedback_history(memory_id)
        return TifScore.from_feedback_history(history)

    async def get_agent_feedback_summary(self, agent_id: str) -> AgentFeedbackSummary:
        """Get aggregate feedback counts and health score for an agent (INT-1).

        Args:
            agent_id: The agent to summarise feedback for.

        Returns:
            :class:`AgentFeedbackSummary` with upvote/downvote/flag counts and health score.
        """
        result = await self._request("GET", f"/v1/agents/{agent_id}/feedback/summary")
        return AgentFeedbackSummary.from_dict(result)

    async def patch_memory_importance(
        self,
        memory_id: str,
        agent_id: str,
        importance: float,
    ) -> FeedbackResponse:
        """Directly override a memory's importance score (INT-1).

        Args:
            memory_id: The memory to update.
            agent_id: The agent that owns the memory.
            importance: New importance value (0.0–1.0).

        Returns:
            :class:`FeedbackResponse` with the new importance value.
        """
        data: dict[str, Any] = {"agent_id": agent_id, "importance": importance}
        result = await self._request("PATCH", f"/v1/memories/{memory_id}/importance", data=data)
        return FeedbackResponse.from_dict(result)

    async def get_feedback_health(self, agent_id: str) -> FeedbackHealthResponse:
        """Get overall feedback health score for an agent (INT-1).

        The health score is the mean importance of all non-expired memories (0.0–1.0).
        A higher score indicates a healthier, more relevant memory store.

        Args:
            agent_id: The agent to get health score for.

        Returns:
            :class:`FeedbackHealthResponse` with health score, memory count, and avg importance.
        """
        result = await self._request("GET", "/v1/feedback/health", params={"agent_id": agent_id})
        return FeedbackHealthResponse.from_dict(result)

    # =========================================================================
    # Memory Knowledge Graph Operations (CE-5 / SDK-9)
    # =========================================================================

    async def memory_graph(
        self,
        memory_id: str,
        depth: int = 1,
        types: list[str] | None = None,
    ) -> MemoryGraph:
        """Traverse the knowledge graph from a memory node.

        Requires CE-5 (Memory Knowledge Graph) on the server.

        Args:
            memory_id: Root memory ID to start traversal from.
            depth: Maximum traversal depth (default: 1, max: 3).
            types: Filter by edge types — any of ``"related_to"``,
                ``"shares_entity"``, ``"precedes"``, ``"linked_by"``.
                ``None`` returns all edge types.

        Returns:
            :class:`MemoryGraph` containing all nodes and edges reachable
            within *depth* hops.

        Example:
            >>> graph = await client.memory_graph("mem-abc", depth=2)
            >>> print(f"{len(graph.nodes)} nodes, {len(graph.edges)} edges")
        """
        params: dict[str, Any] = {"depth": depth}
        if types:
            params["types"] = ",".join(types)
        result = await self._request("GET", f"/v1/memories/{memory_id}/graph", params=params)
        return MemoryGraph.from_dict(result)

    async def memory_path(
        self,
        source_id: str,
        target_id: str,
    ) -> GraphPath:
        """Find the shortest path between two memories in the knowledge graph.

        Requires CE-5 (Memory Knowledge Graph) on the server.

        Args:
            source_id: Starting memory ID.
            target_id: Destination memory ID.

        Returns:
            :class:`GraphPath` with the ordered list of memory IDs and edges.
            If no path exists, ``path`` will be an empty list and ``hops`` will be -1.

        Example:
            >>> path = await client.memory_path("mem-abc", "mem-xyz")
            >>> print(" → ".join(path.path))
        """
        params: dict[str, Any] = {"target": target_id}
        result = await self._request("GET", f"/v1/memories/{source_id}/path", params=params)
        return GraphPath.from_dict(result)

    async def memory_link(
        self,
        source_id: str,
        target_id: str,
        edge_type: str | EdgeType = EdgeType.LINKED_BY,
    ) -> GraphLinkResponse:
        """Create an explicit edge between two memories.

        Requires CE-5 (Memory Knowledge Graph) on the server.

        Args:
            source_id: Source memory ID.
            target_id: Target memory ID.
            edge_type: Edge type — must be ``"linked_by"`` for user-created
                edges (automatic edges use other types).

        Returns:
            :class:`GraphLinkResponse` containing the newly created edge.

        Example:
            >>> resp = await client.memory_link("mem-abc", "mem-xyz")
            >>> print(resp.edge.id)
        """
        edge_type_str = edge_type.value if isinstance(edge_type, EdgeType) else edge_type
        data: dict[str, Any] = {"target_id": target_id, "edge_type": edge_type_str}
        result = await self._request("POST", f"/v1/memories/{source_id}/links", data=data)
        if isinstance(result, dict) and "error" in result:
            raise AuthorizationError(
                message=result.get("message") or result.get("error", "Forbidden"),
                status_code=403,
                response_body=result,
                code=ErrorCode.UNKNOWN,
            )
        return GraphLinkResponse.from_dict(result)

    async def agent_graph_export(
        self,
        agent_id: str,
        format: str = "json",
    ) -> GraphExport:
        """Export the full knowledge graph for an agent.

        Requires CE-5 (Memory Knowledge Graph) on the server.

        Args:
            agent_id: Agent whose graph to export.
            format: Export format — ``"json"`` (default), ``"graphml"``, or ``"csv"``.

        Returns:
            :class:`GraphExport` with serialised graph data and statistics.

        Example:
            >>> export = await client.agent_graph_export("my-agent", format="graphml")
            >>> with open("graph.graphml", "w") as f:
            ...     f.write(export.data)
        """
        params: dict[str, Any] = {"format": format}
        result = await self._request("GET", f"/v1/agents/{agent_id}/graph/export", params=params)
        return GraphExport.from_dict(result)

    # =========================================================================
    # Entity Extraction Operations (CE-4)
    # =========================================================================

    async def get_namespace_entity_config(self, namespace: str) -> dict[str, Any]:
        """Get entity extraction configuration for a namespace."""
        return await self._request("GET", f"/v1/namespaces/{namespace}/config")

    async def get_namespace_extractor(self, namespace: str) -> dict[str, Any]:
        """Get the extractor provider configuration for a namespace."""
        return await self._request("GET", f"/v1/namespaces/{namespace}/extractor")

    async def configure_namespace_ner(
        self,
        namespace: str,
        extract_entities: bool,
        entity_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Configure entity extraction for a namespace.

        Enables or disables GLiNER zero-shot NER + rule-based entity tagging
        on memories stored in this namespace.  Requires ``Scope::Write``.

        Args:
            namespace: Target namespace.
            extract_entities: Enable automatic entity extraction on store.
            entity_types: Entity types to extract, e.g. ``["person", "org",
                "location", "date"]``.  ``None`` keeps existing types.

        Returns:
            Updated namespace config dict.

        Note:
            Requires CE-4 (GLiNER) on the server.  The server falls back to
            rule-based extraction only when the GLiNER model has not yet been
            downloaded.
        """
        config = NamespaceNerConfig(
            extract_entities=extract_entities,
            entity_types=entity_types,
        )
        return await self._request(
            "PATCH", f"/v1/namespaces/{namespace}/config", data=config.to_dict()
        )

    async def extract_entities(
        self,
        text: str,
        entity_types: list[str] | None = None,
    ) -> EntityExtractionResponse:
        """Extract entities from arbitrary text without storing a memory.

        Uses the same GLiNER + rule-based pipeline as automatic extraction.
        Requires ``Scope::Read``.

        Args:
            text: Text to extract entities from.
            entity_types: Entity types to extract.  ``None`` uses server
                defaults (person, org, location, date, url, email).

        Returns:
            :class:`EntityExtractionResponse` with extracted entities.

        Note:
            Requires CE-4 (GLiNER) on the server.
        """
        data: dict[str, Any] = {"text": text}
        if entity_types is not None:
            data["entity_types"] = entity_types
        result = await self._request("POST", "/v1/memories/extract", data=data)
        return EntityExtractionResponse.from_dict(result)

    async def memory_entities(self, memory_id: str) -> MemoryEntitiesResponse:
        """Get entity tags attached to a stored memory.

        Returns entities that were extracted automatically when the memory
        was stored (requires ``extract_entities=True`` on the namespace) or
        via a manual extraction.  Requires ``Scope::Read``.

        Args:
            memory_id: Memory ID to fetch entities for.

        Returns:
            :class:`MemoryEntitiesResponse` with entity list.

        Note:
            Requires CE-4 (GLiNER) on the server.
        """
        result = await self._request("GET", f"/v1/memory/entities/{memory_id}")
        return MemoryEntitiesResponse.from_dict(result)

    # =========================================================================
    # Session Operations
    # =========================================================================

    async def start_session(
        self,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new session. Returns the session dict (unwrapped from the server response)."""
        data: dict[str, Any] = {"agent_id": agent_id}
        if metadata is not None:
            data["metadata"] = metadata
        result = await self._request("POST", "/v1/sessions/start", data=data)
        return result["session"]

    async def end_session(self, session_id: str, summary: str | None = None) -> dict[str, Any]:
        """End a session."""
        data: dict[str, Any] = {}
        if summary is not None:
            data["summary"] = summary
        return await self._request("POST", f"/v1/sessions/{session_id}/end", data=data)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get session details."""
        return await self._request("GET", f"/v1/sessions/{session_id}")

    async def list_sessions(
        self,
        agent_id: str | None = None,
        active_only: bool | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """List sessions."""
        params: dict[str, Any] = {}
        if agent_id is not None:
            params["agent_id"] = agent_id
        if active_only is not None:
            params["active_only"] = str(active_only).lower()
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._request("GET", "/v1/sessions", params=params)

    async def session_memories(self, session_id: str) -> list[dict[str, Any]]:
        """Get memories for a session."""
        return await self._request("GET", f"/v1/sessions/{session_id}/memories")

    # =========================================================================
    # Agent Operations
    # =========================================================================

    async def list_agents(self) -> list[dict[str, Any]]:
        """List all agents."""
        return await self._request("GET", "/v1/agents")

    async def agent_memories(
        self,
        agent_id: str,
        memory_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get memories for an agent."""
        params: dict[str, Any] = {}
        if memory_type is not None:
            params["memory_type"] = memory_type
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", f"/v1/agents/{agent_id}/memories", params=params)

    async def agent_stats(self, agent_id: str) -> dict[str, Any]:
        """Get statistics for an agent."""
        return await self._request("GET", f"/v1/agents/{agent_id}/stats")

    async def agent_sessions(
        self,
        agent_id: str,
        active_only: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get sessions for an agent."""
        params: dict[str, Any] = {}
        if active_only is not None:
            params["active_only"] = str(active_only).lower()
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", f"/v1/agents/{agent_id}/sessions", params=params)

    async def wake_up(
        self,
        agent_id: str,
        top_n: int = 20,
        min_importance: float = 0.0,
    ) -> WakeUpResponse:
        """Return top-N wake-up context memories for an agent (DAK-1690).

        Calls ``GET /v1/agents/{agent_id}/wake-up``. Returns memories ranked by
        ``importance × exp(-ln2 × age / 14d)`` — no embedding inference, served
        from the metadata index for sub-millisecond latency.

        Args:
            agent_id: Agent identifier.
            top_n: Maximum number of memories to return (default 20, max 100).
            min_importance: Only return memories with importance ≥ this value
                (default 0.0).

        Returns:
            :class:`~dakera.models.WakeUpResponse` with ranked memories and
            ``total_available`` count.
        """
        params: dict[str, Any] = {"top_n": top_n, "min_importance": min_importance}
        result = await self._request("GET", f"/v1/agents/{agent_id}/wake-up", params=params)
        return WakeUpResponse.from_dict(result)

    # =========================================================================
    # Cache Warming
    # =========================================================================

    async def warm_cache(
        self,
        namespace: str,
        vector_ids: list[str] | None = None,
        priority: WarmingPriority = WarmingPriority.NORMAL,
        target_tier: WarmingTargetTier = WarmingTargetTier.L2,
        background: bool = False,
        ttl_hint_seconds: int | None = None,
        access_pattern: AccessPatternHint = AccessPatternHint.RANDOM,
        max_vectors: int | None = None,
    ) -> WarmCacheResponse:
        """Warm cache for vectors in a namespace."""
        request = WarmCacheRequest(
            namespace=namespace,
            vector_ids=vector_ids,
            priority=priority,
            target_tier=target_tier,
            background=background,
            ttl_hint_seconds=ttl_hint_seconds,
            access_pattern=access_pattern,
            max_vectors=max_vectors,
        )
        response = await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/cache/warm",
            data=request.to_dict(),
        )
        return WarmCacheResponse.from_dict(response)

    # =========================================================================
    # Advanced Search Operations
    # =========================================================================

    async def multi_vector_search(
        self,
        namespace: str,
        positive: list[list[float]],
        negative: list[list[float]] | None = None,
        top_k: int = 10,
        filter: FilterDict | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        mmr_lambda: float | None = None,
        mmr_prefetch_k: int | None = None,
    ) -> dict[str, Any]:
        """Multi-vector search with positive/negative examples."""
        data: dict[str, Any] = {
            "positive": positive,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_vectors": include_vectors,
        }
        if negative is not None:
            data["negative"] = negative
        if filter:
            data["filter"] = filter
        if mmr_lambda is not None:
            data["mmr_lambda"] = mmr_lambda
        if mmr_prefetch_k is not None:
            data["mmr_prefetch_k"] = mmr_prefetch_k
        return await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/search/multi-vector",
            data=data,
        )

    async def unified_query(
        self,
        namespace: str,
        vector: list[float] | None = None,
        text: str | None = None,
        top_k: int = 10,
        filter: FilterDict | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        vector_weight: float | None = None,
        text_weight: float | None = None,
        fusion_method: str | None = None,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """Unified query combining vector and text search."""
        data: dict[str, Any] = {
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_vectors": include_vectors,
            "rerank": rerank,
        }
        if vector is not None:
            data["vector"] = vector
        if text is not None:
            data["text"] = text
        if filter:
            data["filter"] = filter
        if vector_weight is not None:
            data["vector_weight"] = vector_weight
        if text_weight is not None:
            data["text_weight"] = text_weight
        if fusion_method is not None:
            data["fusion_method"] = fusion_method
        return await self._request("POST", f"/v1/namespaces/{namespace}/search/unified", data=data)

    async def aggregate(
        self,
        namespace: str,
        vector: list[float] | None = None,
        group_by: str | None = None,
        metrics: list[str] | None = None,
        top_k: int | None = None,
        filter: FilterDict | None = None,
        top_groups: int | None = None,
    ) -> dict[str, Any]:
        """Aggregation query with grouping."""
        data: dict[str, Any] = {}
        if vector is not None:
            data["vector"] = vector
        if group_by is not None:
            data["group_by"] = group_by
        if metrics is not None:
            data["metrics"] = metrics
        if top_k is not None:
            data["top_k"] = top_k
        if filter:
            data["filter"] = filter
        if top_groups is not None:
            data["top_groups"] = top_groups
        return await self._request("POST", f"/v1/namespaces/{namespace}/aggregate", data=data)

    async def export_vectors(
        self,
        namespace: str,
        cursor: str | None = None,
        limit: int | None = None,
        filter: FilterDict | None = None,
        include_vectors: bool = False,
    ) -> dict[str, Any]:
        """Export vectors with optional cursor-based pagination."""
        data: dict[str, Any] = {"include_vectors": include_vectors}
        if cursor is not None:
            data["cursor"] = cursor
        if limit is not None:
            data["limit"] = limit
        if filter:
            data["filter"] = filter
        return await self._request("POST", f"/v1/namespaces/{namespace}/export", data=data)

    async def explain_query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int = 10,
        filter: FilterDict | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Explain query execution plan."""
        data: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }
        if filter:
            data["filter"] = filter
        return await self._request("POST", f"/v1/namespaces/{namespace}/query/explain", data=data)

    async def upsert_columns(
        self,
        namespace: str,
        ids: list[str],
        vectors: list[list[float]],
        attributes: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
        dimension: int | None = None,
    ) -> dict[str, Any]:
        """Column-format upsert for efficient bulk operations."""
        data: dict[str, Any] = {"ids": ids, "vectors": vectors}
        if attributes is not None:
            data["attributes"] = attributes
        if ttl_seconds is not None:
            data["ttl_seconds"] = ttl_seconds
        if dimension is not None:
            data["dimension"] = dimension
        return await self._request("POST", f"/v1/namespaces/{namespace}/upsert-columns", data=data)

    # =========================================================================
    # Knowledge Graph Operations
    # =========================================================================

    async def knowledge_graph(
        self,
        agent_id: str,
        memory_id: str | None = None,
        depth: int | None = None,
        min_similarity: float | None = None,
    ) -> dict[str, Any]:
        """Build a knowledge graph for an agent."""
        data: dict[str, Any] = {"agent_id": agent_id}
        if memory_id is not None:
            data["memory_id"] = memory_id
        if depth is not None:
            data["depth"] = depth
        if min_similarity is not None:
            data["min_similarity"] = min_similarity
        return await self._request("POST", "/v1/knowledge/graph", data=data)

    async def full_knowledge_graph(
        self,
        agent_id: str,
        max_nodes: int | None = None,
        min_similarity: float | None = None,
        cluster_threshold: float | None = None,
        max_edges_per_node: int | None = None,
    ) -> dict[str, Any]:
        """Build a full knowledge graph for an agent."""
        data: dict[str, Any] = {"agent_id": agent_id}
        if max_nodes is not None:
            data["max_nodes"] = max_nodes
        if min_similarity is not None:
            data["min_similarity"] = min_similarity
        if cluster_threshold is not None:
            data["cluster_threshold"] = cluster_threshold
        if max_edges_per_node is not None:
            data["max_edges_per_node"] = max_edges_per_node
        return await self._request("POST", "/v1/knowledge/graph/full", data=data)

    async def summarize(
        self,
        agent_id: str,
        memory_ids: list[str] | None = None,
        target_type: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Summarize memories for an agent."""
        data: dict[str, Any] = {"agent_id": agent_id, "dry_run": dry_run}
        if memory_ids is not None:
            data["memory_ids"] = memory_ids
        if target_type is not None:
            data["target_type"] = target_type
        return await self._request("POST", "/v1/knowledge/summarize", data=data)

    async def deduplicate(
        self,
        agent_id: str,
        threshold: float | None = None,
        memory_type: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Deduplicate memories for an agent."""
        data: dict[str, Any] = {"agent_id": agent_id, "dry_run": dry_run}
        if threshold is not None:
            data["threshold"] = threshold
        if memory_type is not None:
            data["memory_type"] = memory_type
        return await self._request("POST", "/v1/knowledge/deduplicate", data=data)

    # =========================================================================
    # KG-2: Graph Query & Export Operations
    # =========================================================================

    async def knowledge_query(
        self,
        agent_id: str,
        root_id: str | None = None,
        edge_type: str | None = None,
        min_weight: float | None = None,
        max_depth: int = 3,
        limit: int = 100,
    ) -> KgQueryResponse:
        """Query the memory knowledge graph using a filter DSL (KG-2).

        Calls ``GET /v1/knowledge/query``.

        Args:
            agent_id: Agent whose graph to query.
            root_id: Optional root memory ID — if set, performs BFS traversal
                from this node first (up to *max_depth* hops).
            edge_type: Filter edges by type (comma-separated, e.g.
                ``"related_to,shares_entity"``).
            min_weight: Minimum edge weight (0.0–1.0).
            max_depth: BFS depth when *root_id* is set (1–5, default 3).
            limit: Maximum number of edges to return (default 100, max 1000).
        """
        params: dict[str, Any] = {
            "agent_id": agent_id,
            "max_depth": max_depth,
            "limit": limit,
        }
        if root_id is not None:
            params["root_id"] = root_id
        if edge_type is not None:
            params["edge_type"] = edge_type
        if min_weight is not None:
            params["min_weight"] = min_weight
        result = await self._request("GET", "/v1/knowledge/query", params=params)
        return KgQueryResponse.from_dict(result)

    async def knowledge_path(
        self,
        agent_id: str,
        from_id: str,
        to_id: str,
    ) -> KgPathResponse:
        """Find the BFS shortest path between two memory IDs (KG-2).

        Calls ``GET /v1/knowledge/path``.

        Args:
            agent_id: Agent whose graph to traverse.
            from_id: Source memory ID.
            to_id: Target memory ID.

        Raises:
            :exc:`NotFoundError`: If no path exists between the two memories.
        """
        params: dict[str, Any] = {
            "agent_id": agent_id,
            "from": from_id,
            "to": to_id,
        }
        result = await self._request("GET", "/v1/knowledge/path", params=params)
        return KgPathResponse.from_dict(result)

    async def knowledge_export(
        self,
        agent_id: str,
        format: str = "json",
    ) -> KgExportResponse:
        """Export the memory knowledge graph as JSON or GraphML (KG-2).

        Calls ``GET /v1/knowledge/export``.

        Args:
            agent_id: Agent whose graph to export.
            format: Export format — ``"json"`` (default) or ``"graphml"``.

        Returns:
            :class:`KgExportResponse` for ``format="json"``.

        Note:
            For ``format="graphml"`` the server returns ``application/xml``.
            This method will raise a parse error in that case — use the raw
            HTTP client directly if you need the GraphML XML string.
        """
        params: dict[str, Any] = {"agent_id": agent_id, "format": format}
        result = await self._request("GET", "/v1/knowledge/export", params=params)
        return KgExportResponse.from_dict(result)

    # =========================================================================
    # Analytics Operations
    # =========================================================================

    async def analytics_overview(
        self,
        period: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Get analytics overview."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        if namespace:
            params["namespace"] = namespace
        return await self._request("GET", "/v1/analytics/overview", params=params)

    async def analytics_latency(
        self,
        period: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Get latency analytics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        if namespace:
            params["namespace"] = namespace
        return await self._request("GET", "/v1/analytics/latency", params=params)

    async def analytics_throughput(
        self,
        period: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Get throughput analytics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        if namespace:
            params["namespace"] = namespace
        return await self._request("GET", "/v1/analytics/throughput", params=params)

    async def analytics_storage(self, namespace: str | None = None) -> dict[str, Any]:
        """Get storage analytics."""
        params: dict[str, Any] = {}
        if namespace:
            params["namespace"] = namespace
        return await self._request("GET", "/v1/analytics/storage", params=params)

    # =========================================================================
    # Admin Operations
    # =========================================================================

    async def ops_stats(self) -> dict[str, Any]:
        """Get server stats (version, total_vectors, namespace_count,
        uptime_seconds, timestamp, state).

        Requires Read scope — works with read-only API keys, unlike cluster_status.
        The ``state`` field is ``"healthy"`` when storage is accessible, ``"degraded"`` otherwise.
        """
        return await self._request("GET", "/v1/ops/stats")

    async def ops_metrics(self) -> str:
        """Get Prometheus metrics in text exposition format (INFRA-3).

        Requires Admin scope. Returns the raw Prometheus text exposition
        format string suitable for scraping by a Prometheus server.
        """
        return await self._request("GET", "/v1/ops/metrics")

    async def cluster_status(self) -> dict[str, Any]:
        """Get cluster status."""
        return await self._request("GET", "/v1/admin/cluster/status")

    async def cluster_nodes(self) -> list[dict[str, Any]]:
        """Get cluster nodes."""
        return await self._request("GET", "/v1/admin/cluster/nodes")

    async def optimize_namespace(self, namespace: str) -> dict[str, Any]:
        """Optimize a namespace."""
        return await self._request("POST", f"/v1/admin/namespaces/{namespace}/optimize")

    async def index_stats(self, namespace: str) -> dict[str, Any]:
        """Get admin index stats for a namespace."""
        return await self._request("GET", f"/v1/admin/namespaces/{namespace}/index/stats")

    async def rebuild_indexes(self, namespace: str) -> dict[str, Any]:
        """Rebuild indexes for a namespace."""
        return await self._request("POST", f"/v1/admin/namespaces/{namespace}/index/rebuild")

    async def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self._request("GET", "/v1/admin/cache/stats")

    async def cache_clear(self, namespace: str | None = None) -> dict[str, Any]:
        """Clear cache."""
        path = f"/v1/admin/cache/clear/{namespace}" if namespace else "/v1/admin/cache/clear"
        return await self._request("POST", path)

    async def get_config(self) -> dict[str, Any]:
        """Get server configuration."""
        return await self._request("GET", "/v1/admin/config")

    async def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Update server configuration."""
        return await self._request("PUT", "/v1/admin/config", data=config)

    async def get_quotas(self) -> dict[str, Any]:
        """Get server quotas."""
        return await self._request("GET", "/v1/admin/quotas")

    async def update_quotas(self, quotas: dict[str, Any]) -> dict[str, Any]:
        """Update server quotas."""
        return await self._request("PUT", "/v1/admin/quotas", data=quotas)

    async def slow_queries(
        self,
        limit: int = 10,
        min_duration_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get slow query log."""
        params: dict[str, Any] = {"limit": limit}
        if min_duration_ms is not None:
            params["min_duration_ms"] = min_duration_ms
        return await self._request("GET", "/v1/admin/slow-queries", params=params)

    async def create_backup(self, include_data: bool = True) -> dict[str, Any]:
        """Create a backup."""
        return await self._request("POST", "/v1/admin/backups", data={"include_data": include_data})

    async def list_backups(self) -> list[dict[str, Any]]:
        """List backups."""
        return await self._request("GET", "/v1/admin/backups")

    async def restore_backup(self, backup_id: str) -> dict[str, Any]:
        """Restore from a backup."""
        return await self._request("POST", f"/v1/admin/backups/{backup_id}/restore")

    async def delete_backup(self, backup_id: str) -> dict[str, Any]:
        """Delete a backup."""
        return await self._request("DELETE", f"/v1/admin/backups/{backup_id}")

    async def configure_ttl(
        self,
        namespace: str,
        ttl_seconds: int,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Configure TTL for a namespace."""
        data: dict[str, Any] = {"namespace": namespace, "ttl_seconds": ttl_seconds}
        if strategy is not None:
            data["strategy"] = strategy
        return await self._request("PUT", f"/v1/admin/namespaces/{namespace}/ttl", data=data)

    async def autopilot_status(self) -> dict[str, Any]:
        """Get AutoPilot status: current config and last-run statistics (PILOT-1)."""
        return await self._request("GET", "/v1/admin/autopilot/status")

    async def autopilot_update_config(
        self,
        enabled: bool | None = None,
        dedup_threshold: float | None = None,
        dedup_interval_hours: int | None = None,
        consolidation_interval_hours: int | None = None,
    ) -> dict[str, Any]:
        """Update AutoPilot configuration at runtime (PILOT-2).

        All parameters are optional — omit any to keep its current value.
        """
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        if dedup_threshold is not None:
            data["dedup_threshold"] = dedup_threshold
        if dedup_interval_hours is not None:
            data["dedup_interval_hours"] = dedup_interval_hours
        if consolidation_interval_hours is not None:
            data["consolidation_interval_hours"] = consolidation_interval_hours
        return await self._request("PUT", "/v1/admin/autopilot/config", data=data)

    async def autopilot_trigger(self, action: str) -> dict[str, Any]:
        """Manually trigger an AutoPilot cycle (PILOT-3).

        Args:
            action: One of ``"dedup"``, ``"consolidate"``, or ``"all"``.
        """
        return await self._request("POST", "/v1/admin/autopilot/trigger", data={"action": action})

    async def decay_config(self) -> dict[str, Any]:
        """Get current decay engine configuration (DECAY-1).

        Returns the active decay strategy, half-life, and minimum importance
        threshold. Requires Admin scope.
        """
        return await self._request("GET", "/v1/admin/decay/config")

    async def decay_update_config(
        self,
        strategy: str | None = None,
        half_life_hours: float | None = None,
        min_importance: float | None = None,
    ) -> dict[str, Any]:
        """Update decay engine configuration at runtime (DECAY-1).

        Changes take effect on the next decay cycle — no restart required.
        All parameters are optional; omit any to keep its current value.

        Args:
            strategy: Decay strategy: ``"exponential"``, ``"linear"``, or
                ``"step"``.
            half_life_hours: Half-life in hours (must be > 0).
            min_importance: Minimum importance threshold 0.0–1.0; memories
                below this value are hard-deleted on the next cycle.
        """
        data: dict[str, Any] = {}
        if strategy is not None:
            data["strategy"] = strategy
        if half_life_hours is not None:
            data["half_life_hours"] = half_life_hours
        if min_importance is not None:
            data["min_importance"] = min_importance
        return await self._request("PUT", "/v1/admin/decay/config", data=data)

    async def decay_stats(self) -> dict[str, Any]:
        """Get decay engine activity counters and last-cycle snapshot (DECAY-2).

        Returns cumulative totals (memories decayed/deleted, cycles run) and
        per-cycle statistics from the most recent run. Requires Admin scope.
        """
        return await self._request("GET", "/v1/admin/decay/stats")

    async def get_kpis(self) -> KpiSnapshot:
        """Return a point-in-time product KPI snapshot (OBS-2).

        Calls ``GET /v1/kpis``. Returns 8 operational metrics covering
        latency, error rate, and retention. Sub-millisecond — served from
        in-memory counters. Requires Admin scope.

        Returns:
            :class:`~dakera.models.KpiSnapshot` with all 8 KPI fields.
        """
        result = await self._request("GET", "/v1/kpis")
        return KpiSnapshot.from_dict(result)

    async def rotate_encryption_key(
        self,
        new_key: str,
        namespace: str | None = None,
    ) -> RotateEncryptionKeyResponse:
        """Re-encrypt all memory content blobs with a new AES-256-GCM key (SEC-3).

        Requires Admin scope.

        Args:
            new_key: New passphrase or 64-char hex key.
            namespace: Rotate only this namespace. Omit to rotate all.

        Returns:
            :class:`RotateEncryptionKeyResponse` with counts of rotated,
            skipped, and affected namespace names.
        """
        data: dict[str, Any] = {"new_key": new_key}
        if namespace is not None:
            data["namespace"] = namespace
        result = await self._request("POST", "/v1/admin/encryption/rotate-key", data=data)
        return RotateEncryptionKeyResponse.from_dict(result)

    # =========================================================================
    # API Key Operations
    # =========================================================================

    async def create_key(
        self,
        name: str,
        permissions: list[str] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Create an API key."""
        data: dict[str, Any] = {"name": name}
        if permissions is not None:
            data["permissions"] = permissions
        if expires_at is not None:
            data["expires_at"] = expires_at
        return await self._request("POST", "/v1/keys", data=data)

    async def list_keys(self) -> list[dict[str, Any]]:
        """List API keys."""
        return await self._request("GET", "/v1/keys")

    async def get_key(self, key_id: str) -> dict[str, Any]:
        """Get an API key."""
        return await self._request("GET", f"/v1/keys/{key_id}")

    async def delete_key(self, key_id: str) -> dict[str, Any]:
        """Delete an API key."""
        return await self._request("DELETE", f"/v1/keys/{key_id}")

    async def deactivate_key(self, key_id: str) -> dict[str, Any]:
        """Deactivate an API key."""
        return await self._request("POST", f"/v1/keys/{key_id}/deactivate")

    async def rotate_key(self, key_id: str) -> dict[str, Any]:
        """Rotate an API key."""
        return await self._request("POST", f"/v1/keys/{key_id}/rotate")

    async def key_usage(self, key_id: str) -> dict[str, Any]:
        """Get usage statistics for an API key."""
        return await self._request("GET", f"/v1/keys/{key_id}/usage")

    # =========================================================================
    # SSE Streaming (CE-1)
    # =========================================================================

    def _parse_sse_block(self, block: str) -> DakeraEvent | None:
        """Parse a single SSE event block into a :class:`~dakera.models.DakeraEvent`."""
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith(":"):
                continue  # SSE comment / heartbeat
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        if not data_lines:
            return None
        try:
            return DakeraEvent.from_dict(json.loads("\n".join(data_lines)))
        except Exception:
            return None

    async def stream_namespace_events(
        self,
        namespace: str,
    ) -> AsyncGenerator[DakeraEvent, None]:
        """Stream SSE events scoped to *namespace*.

        Opens a long-lived HTTP connection to ``GET /v1/namespaces/{namespace}/events``
        and yields :class:`~dakera.models.DakeraEvent` objects as they arrive.

        Requires a Read-scoped API key.

        Args:
            namespace: The namespace to subscribe to.

        Yields:
            :class:`~dakera.models.DakeraEvent` — one per SSE event.

        Example::

            async for event in client.stream_namespace_events("my-ns"):
                print(event.type, event)
        """
        url = self._url(f"/v1/namespaces/{namespace}/events")
        async for event in self._stream_sse(url):
            yield event

    async def stream_global_events(self) -> AsyncGenerator[DakeraEvent, None]:
        """Stream all system events from the global event bus.

        Opens a long-lived HTTP connection to ``GET /ops/events`` and yields
        :class:`~dakera.models.DakeraEvent` objects as they arrive.

        Requires an Admin-scoped API key.

        Yields:
            :class:`~dakera.models.DakeraEvent` — one per SSE event.
        """
        url = self._url("/ops/events")
        async for event in self._stream_sse(url):
            yield event

    async def _stream_sse(self, url: str) -> AsyncGenerator[DakeraEvent, None]:
        """Low-level async SSE streaming helper."""
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        async with self._client.stream("GET", url, headers=headers, timeout=None) as response:
            if not response.is_success:
                # Read the error body before raising so we have context.
                await response.aread()
                self._handle_response(response)  # always raises
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                # SSE events are separated by a blank line (\n\n)
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    event = self._parse_sse_block(block)
                    if event is not None:
                        yield event

    # =========================================================================
    # DASH-B: Memory Lifecycle Event Stream
    # =========================================================================

    async def stream_memory_events(self) -> AsyncGenerator[MemoryEvent, None]:
        """Stream memory lifecycle events from the DASH-B SSE endpoint.

        Opens a long-lived HTTP connection to ``GET /v1/events/stream`` and
        yields :class:`~dakera.models.MemoryEvent` objects as they arrive.

        Requires a Read-scoped API key.

        Event types: ``stored``, ``recalled``, ``forgotten``, ``consolidated``,
        ``importance_updated``, ``session_started``, ``session_ended``.

        Yields:
            :class:`~dakera.models.MemoryEvent` — one per SSE event.

        Example::

            async for event in client.stream_memory_events():
                if event.event_type == "stored":
                    print(f"[{event.agent_id}] stored {event.memory_id}")
        """
        import json as _json

        url = self._url("/v1/events/stream")
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        async with self._client.stream("GET", url, headers=headers, timeout=None) as response:
            if not response.is_success:
                await response.aread()
                self._handle_response(response)
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    data_lines = []
                    for line in block.splitlines():
                        if line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    if data_lines:
                        try:
                            payload = _json.loads("\n".join(data_lines))
                            yield MemoryEvent.from_dict(payload)
                        except (ValueError, TypeError, KeyError):
                            pass

    # =========================================================================
    # DASH-A: Cross-Agent Network
    # =========================================================================

    async def cross_agent_network(
        self,
        agent_ids: list[str] | None = None,
        min_similarity: float = 0.3,
        max_nodes_per_agent: int = 50,
        min_importance: float = 0.0,
        max_cross_edges: int = 200,
    ) -> CrossAgentNetworkResponse:
        """Build the cross-agent memory similarity network.

        Calls ``POST /v1/knowledge/network/cross-agent`` (Admin scope) and
        returns a graph of agents, memory nodes, and cross-agent similarity
        edges suitable for rendering as a network diagram.

        Args:
            agent_ids: Limit the graph to these agent IDs.  ``None`` (default)
                includes all agents.
            min_similarity: Minimum cosine similarity for a cross-agent edge
                (0–1, default 0.3).
            max_nodes_per_agent: Maximum number of memories to include per
                agent, selected by descending importance (default 50).
            min_importance: Minimum importance score for a memory to be
                included (0–1, default 0.0).
            max_cross_edges: Maximum number of cross-agent edges to return
                (default 200).

        Returns:
            :class:`~dakera.models.CrossAgentNetworkResponse` with ``agents``,
            ``nodes``, ``edges``, and ``stats`` fields.

        Example::

            graph = await client.cross_agent_network(min_similarity=0.5)
            print(f"{graph.stats.total_agents} agents, "
                  f"{graph.stats.total_cross_edges} cross-agent edges")
        """
        payload: dict[str, Any] = {
            "min_similarity": min_similarity,
            "max_nodes_per_agent": max_nodes_per_agent,
            "min_importance": min_importance,
            "max_cross_edges": max_cross_edges,
        }
        if agent_ids is not None:
            payload["agent_ids"] = agent_ids

        data = await self._request("POST", "/v1/knowledge/network/cross-agent", data=payload)
        return CrossAgentNetworkResponse.from_dict(data)

    # =========================================================================
    # SDK-10: Agent Memory Subscription
    # =========================================================================

    async def agents_subscribe(
        self,
        agent_id: str,
        *,
        tags: list[str] | None = None,
        reconnect: bool = True,
        reconnect_delay: float = 1.0,
    ) -> AsyncGenerator[MemoryEvent, None]:
        """Subscribe to real-time memory lifecycle events for an agent.

        Opens a long-lived connection to ``GET /v1/events/stream`` and yields
        :class:`~dakera.models.MemoryEvent` objects filtered to the given
        ``agent_id``.  Reconnects automatically on connection drop when
        ``reconnect=True``.

        Requires a Read-scoped API key.

        Args:
            agent_id: Agent whose events to receive.
            tags: Optional tag filter — only events whose tags overlap this list
                are yielded.  Pass ``None`` (default) to receive all events for
                the agent.
            reconnect: Automatically reconnect on connection drop or error.
                Defaults to ``True``.
            reconnect_delay: Seconds to wait between reconnection attempts.
                Defaults to ``1.0``.

        Yields:
            :class:`~dakera.models.MemoryEvent` — one per matching SSE event.
            The ``connected`` handshake event is skipped.

        Example::

            async for event in client.agents_subscribe("my-bot", tags=["important"]):
                print(f"{event.event_type}: {event.memory_id}")
        """
        while True:
            try:
                async for event in self.stream_memory_events():
                    if event.event_type == "connected":
                        continue
                    if event.agent_id != agent_id:
                        continue
                    if tags and not any(t in (event.tags or []) for t in tags):
                        continue
                    yield event
                # Stream closed cleanly — reconnect if requested.
                if not reconnect:
                    return
            except Exception:
                if not reconnect:
                    raise
            await asyncio.sleep(reconnect_delay)

    # =========================================================================
    # Namespace API Keys — SEC-1
    # =========================================================================

    async def create_namespace_key(
        self,
        namespace: str,
        name: str,
        expires_in_days: int | None = None,
    ) -> CreateNamespaceKeyResponse:
        """Create a namespace-scoped API key (SEC-1).

        The returned ``key`` value is shown **only once**. Store it securely.

        Args:
            namespace: The namespace to scope this key to.
            name: Human-readable label for the key.
            expires_in_days: Optional expiry in days from now.

        Returns:
            :class:`CreateNamespaceKeyResponse` containing the raw API key.
        """
        data: dict[str, Any] = {"name": name}
        if expires_in_days is not None:
            data["expires_in_days"] = expires_in_days
        result = await self._request("POST", f"/v1/namespaces/{namespace}/keys", data=data)
        return CreateNamespaceKeyResponse.from_dict(result)

    async def list_namespace_keys(self, namespace: str) -> ListNamespaceKeysResponse:
        """List all API keys scoped to a namespace (SEC-1).

        Args:
            namespace: The namespace whose keys to list.

        Returns:
            :class:`ListNamespaceKeysResponse` with key metadata (no secrets).
        """
        result = await self._request("GET", f"/v1/namespaces/{namespace}/keys")
        return ListNamespaceKeysResponse.from_dict(result)

    async def delete_namespace_key(self, namespace: str, key_id: str) -> dict[str, Any]:
        """Revoke a namespace-scoped API key (SEC-1).

        Args:
            namespace: The namespace the key belongs to.
            key_id: The key to revoke.

        Returns:
            Dict with ``success`` and ``message`` fields.
        """
        return await self._request("DELETE", f"/v1/namespaces/{namespace}/keys/{key_id}")

    async def get_namespace_key_usage(
        self, namespace: str, key_id: str
    ) -> NamespaceKeyUsageResponse:
        """Get usage statistics for a namespace-scoped API key (SEC-1).

        Args:
            namespace: The namespace the key belongs to.
            key_id: The key whose usage to retrieve.

        Returns:
            :class:`NamespaceKeyUsageResponse` with request counts and latency.
        """
        result = await self._request("GET", f"/v1/namespaces/{namespace}/keys/{key_id}/usage")
        return NamespaceKeyUsageResponse.from_dict(result)

    # =========================================================================
    # DX-1: Memory Import / Export
    # =========================================================================

    async def import_memories(
        self,
        data: Any,
        format: str = "jsonl",
        agent_id: str | None = None,
        namespace: str | None = None,
    ) -> MemoryImportResponse:
        """Import memories from an external format (DX-1)."""
        body: dict[str, Any] = {"data": data, "format": format}
        if agent_id is not None:
            body["agent_id"] = agent_id
        if namespace is not None:
            body["namespace"] = namespace
        result = await self._request("POST", "/v1/import", data=body)
        return MemoryImportResponse.from_dict(result)

    async def export_memories(
        self,
        format: str = "jsonl",
        agent_id: str | None = None,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> MemoryExportResponse:
        """Export memories in a portable format (DX-1)."""
        params: dict[str, Any] = {"format": format}
        if agent_id is not None:
            params["agent_id"] = agent_id
        if namespace is not None:
            params["namespace"] = namespace
        if limit is not None:
            params["limit"] = limit
        result = await self._request("GET", "/v1/export", params=params)
        return MemoryExportResponse.from_dict(result)

    # =========================================================================
    # OBS-1: Business-Event Audit Log
    # =========================================================================

    async def list_audit_events(
        self,
        agent_id: str | None = None,
        event_type: str | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> AuditListResponse:
        """List business-event audit log entries (OBS-1)."""
        params: dict[str, Any] = {}
        if agent_id is not None:
            params["agent_id"] = agent_id
        if event_type is not None:
            params["event_type"] = event_type
        if from_ts is not None:
            params["from"] = from_ts
        if to_ts is not None:
            params["to"] = to_ts
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        result = await self._request("GET", "/v1/audit", params=params)
        return AuditListResponse.from_dict(result)

    async def stream_audit_events(
        self,
        agent_id: str | None = None,
        event_type: str | None = None,
    ) -> AsyncGenerator[DakeraEvent, None]:
        """Stream live audit events via SSE (OBS-1).

        Yields:
            :class:`~dakera.models.DakeraEvent` — one per audit event.
        """
        from urllib.parse import urlencode

        params: dict[str, str] = {}
        if agent_id is not None:
            params["agent_id"] = agent_id
        if event_type is not None:
            params["event_type"] = event_type
        url = self._url("/v1/audit/stream")
        if params:
            url = f"{url}?{urlencode(params)}"
        async for event in self._stream_sse(url):
            yield event

    async def export_audit(
        self,
        format: str = "jsonl",
        agent_id: str | None = None,
        event_type: str | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
    ) -> AuditExportResponse:
        """Bulk-export audit log entries (OBS-1)."""
        body: dict[str, Any] = {"format": format}
        if agent_id is not None:
            body["agent_id"] = agent_id
        if event_type is not None:
            body["event_type"] = event_type
        if from_ts is not None:
            body["from"] = from_ts
        if to_ts is not None:
            body["to"] = to_ts
        result = await self._request("POST", "/v1/audit/export", data=body)
        return AuditExportResponse.from_dict(result)

    # =========================================================================
    # EXT-1: External Extraction Providers
    # =========================================================================

    async def extract_text(
        self,
        text: str,
        namespace: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> ExtractionResult:
        """Extract entities from text using a pluggable provider (EXT-1)."""
        body: dict[str, Any] = {"text": text}
        if namespace is not None:
            body["namespace"] = namespace
        if provider is not None:
            body["provider"] = provider
        if model is not None:
            body["model"] = model
        result = await self._request("POST", "/v1/extract", data=body)
        return ExtractionResult.from_dict(result)

    async def list_extract_providers(self) -> list[ExtractionProviderInfo]:
        """List available extraction providers and their models (EXT-1)."""
        result = await self._request("GET", "/v1/extract/providers")
        items = result if isinstance(result, list) else result.get("providers", [])
        return [ExtractionProviderInfo.from_dict(p) for p in items]

    async def configure_namespace_extractor(
        self,
        namespace: str,
        provider: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Set the default extraction provider for a namespace (EXT-1)."""
        body: dict[str, Any] = {"provider": provider}
        if model is not None:
            body["model"] = model
        return await self._request("PATCH", f"/v1/namespaces/{namespace}/extractor", data=body)

    # =========================================================================
    # ODE-2: GLiNER Entity Extraction (dakera-ode sidecar)
    # =========================================================================

    async def ode_extract_entities(
        self,
        content: str,
        agent_id: str,
        memory_id: str | None = None,
        entity_types: list[str] | None = None,
    ) -> ExtractEntitiesResponse:
        """Extract named entities from text using the GLiNER sidecar (ODE-2).

        Calls ``POST /ode/extract`` on the dakera-ode sidecar service.
        Requires :attr:`ode_url` to be configured on the client.

        Args:
            content: The text to extract entities from.
            agent_id: Agent context for the extraction.
            memory_id: Optional memory ID to associate with the extraction.
            entity_types: Optional list of entity type labels to extract.
                When omitted, the ODE sidecar uses its default set of types.

        Returns:
            :class:`ExtractEntitiesResponse` containing extracted entities,
            the GLiNER model variant used, and processing time in ms.

        Raises:
            ValueError: If :attr:`ode_url` is not configured.
        """
        if not self.ode_url:
            raise ValueError(
                "ode_url must be configured to use ode_extract_entities(). "
                "Pass ode_url='http://localhost:8080' to AsyncDakeraClient."
            )
        body: dict[str, Any] = {"content": content, "agent_id": agent_id}
        if memory_id is not None:
            body["memory_id"] = memory_id
        if entity_types is not None:
            body["entity_types"] = entity_types
        response = await self._client.post(
            f"{self.ode_url}/ode/extract",
            json=body,
        )
        data = self._handle_response(response)
        return ExtractEntitiesResponse.from_dict(data)

    # =========================================================================
    # COG-1: Per-namespace Memory Lifecycle Policy
    # =========================================================================

    async def get_memory_policy(self, namespace: str) -> MemoryPolicy:
        """Return the memory lifecycle policy for a namespace (COG-1).

        Calls ``GET /v1/namespaces/{namespace}/memory_policy``.

        When no explicit policy has been configured the server returns the
        COG-1 defaults (working=4 h, episodic=30 d, semantic=365 d,
        procedural=730 d; exponential/power_law/logarithmic/flat decay;
        spaced-repetition factor 1.0).

        Args:
            namespace: Namespace to inspect.

        Returns:
            :class:`MemoryPolicy` describing the current lifecycle settings.
        """
        result = await self._request("GET", f"/v1/namespaces/{namespace}/memory_policy")
        return MemoryPolicy.from_dict(result)

    async def set_memory_policy(self, namespace: str, policy: MemoryPolicy) -> MemoryPolicy:
        """Set the memory lifecycle policy for a namespace (COG-1).

        Calls ``PUT /v1/namespaces/{namespace}/memory_policy``.

        The policy is persisted in namespace config and applied immediately to
        the decay engine background task.  Only set the fields you want to
        override — all fields have safe defaults.

        Args:
            namespace: Namespace to configure.
            policy: :class:`MemoryPolicy` with the desired settings.

        Returns:
            The updated :class:`MemoryPolicy` as confirmed by the server.
        """
        result = await self._request(
            "PUT",
            f"/v1/namespaces/{namespace}/memory_policy",
            data=policy.to_dict(),
        )
        return MemoryPolicy.from_dict(result)

    # =========================================================================
    # CE-54: Fulltext Reindex (Admin)
    # =========================================================================

    async def admin_fulltext_reindex(
        self, namespace: str | None = None
    ) -> FulltextReindexResponse:
        """Backfill the BM25 fulltext index for memories that were stored before
        CE-12 auto-indexing was added (CE-54).

        Calls ``POST /admin/fulltext/reindex``. Requires Admin scope.

        Scans all memories in *namespace* (or every agent namespace when
        *namespace* is omitted) and adds any that are missing from the BM25
        index. Safe to call multiple times — already-indexed memories are
        counted in ``total_skipped`` and not re-processed.

        Args:
            namespace: Target namespace. Omit to reindex all agent namespaces.

        Returns:
            :class:`FulltextReindexResponse` with per-namespace breakdown.
        """
        data: dict[str, Any] = {}
        if namespace is not None:
            data["namespace"] = namespace
        result = await self._request("POST", "/v1/admin/fulltext/reindex", data=data)
        return FulltextReindexResponse.from_dict(result)

    # =========================================================================
    # Admin — Cluster & Maintenance
    # =========================================================================

    async def admin_cluster_replication(self) -> dict[str, Any]:
        """GET /admin/cluster/replication — cluster replication status."""
        return await self._request("GET", "/v1/admin/cluster/replication")

    async def admin_list_shards(self) -> dict[str, Any]:
        """GET /admin/cluster/shards — list shards."""
        return await self._request("GET", "/v1/admin/cluster/shards")

    async def admin_rebalance_shards(
        self,
        shard_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """POST /admin/cluster/shards/rebalance — rebalance shards."""
        data: dict[str, Any] = {"dry_run": dry_run}
        if shard_ids is not None:
            data["shard_ids"] = shard_ids
        return await self._request("POST", "/v1/admin/cluster/shards/rebalance", data=data)

    async def admin_maintenance_status(self) -> dict[str, Any]:
        """GET /admin/cluster/maintenance — maintenance mode status."""
        return await self._request("GET", "/v1/admin/cluster/maintenance")

    async def admin_enable_maintenance(
        self,
        reason: str,
        node_ids: list[str] | None = None,
        reject_requests: bool = False,
        duration_minutes: int | None = None,
    ) -> dict[str, Any]:
        """POST /admin/cluster/maintenance/enable — enable maintenance mode."""
        data: dict[str, Any] = {"reason": reason, "reject_requests": reject_requests}
        if node_ids is not None:
            data["node_ids"] = node_ids
        if duration_minutes is not None:
            data["duration_minutes"] = duration_minutes
        return await self._request("POST", "/v1/admin/cluster/maintenance/enable", data=data)

    async def admin_disable_maintenance(self, force: bool | None = None) -> dict[str, Any]:
        """POST /admin/cluster/maintenance/disable — disable maintenance mode."""
        data: dict[str, Any] = {}
        if force is not None:
            data["force"] = force
        return await self._request("POST", "/v1/admin/cluster/maintenance/disable", data=data)

    # =========================================================================
    # Admin — Quotas
    # =========================================================================

    async def admin_list_quotas(self) -> dict[str, Any]:
        """GET /admin/quotas — list all namespace quotas."""
        return await self._request("GET", "/v1/admin/quotas")

    async def admin_get_default_quota(self) -> dict[str, Any]:
        """GET /admin/quotas/default — get default quota configuration."""
        return await self._request("GET", "/v1/admin/quotas/default")

    async def admin_set_default_quota(self, config: dict[str, Any] | None) -> dict[str, Any]:
        """PUT /admin/quotas/default — set default quota configuration."""
        return await self._request("PUT", "/v1/admin/quotas/default", data={"config": config})

    async def admin_get_quota(self, namespace: str) -> dict[str, Any]:
        """GET /admin/quotas/{namespace} — get namespace quota."""
        return await self._request("GET", f"/v1/admin/quotas/{namespace}")

    async def admin_set_quota(self, namespace: str, config: dict[str, Any]) -> dict[str, Any]:
        """PUT /admin/quotas/{namespace} — set namespace quota."""
        return await self._request("PUT", f"/v1/admin/quotas/{namespace}", data={"config": config})

    async def admin_delete_quota(self, namespace: str) -> dict[str, Any]:
        """DELETE /admin/quotas/{namespace} — remove namespace quota."""
        return await self._request("DELETE", f"/v1/admin/quotas/{namespace}")

    async def admin_check_quota(
        self,
        namespace: str,
        vector_ids: list[str],
        dimensions: int | None = None,
        metadata_bytes: int | None = None,
    ) -> dict[str, Any]:
        """POST /admin/quotas/{namespace}/check — check if operation would exceed quota."""
        data: dict[str, Any] = {"vector_ids": vector_ids}
        if dimensions is not None:
            data["dimensions"] = dimensions
        if metadata_bytes is not None:
            data["metadata_bytes"] = metadata_bytes
        return await self._request("POST", f"/v1/admin/quotas/{namespace}/check", data=data)

    # =========================================================================
    # Admin — Slow Queries
    # =========================================================================

    async def admin_list_slow_queries(
        self,
        namespace: str | None = None,
        query_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """GET /admin/slow-queries — list recent slow queries."""
        params: dict[str, Any] = {}
        if namespace is not None:
            params["namespace"] = namespace
        if query_type is not None:
            params["query_type"] = query_type
        if limit is not None:
            params["limit"] = limit
        return await self._request("GET", "/v1/admin/slow-queries", params=params)

    async def admin_slow_query_summary(self) -> dict[str, Any]:
        """GET /admin/slow-queries/summary — slow query summary."""
        return await self._request("GET", "/v1/admin/slow-queries/summary")

    async def admin_clear_slow_queries(self, namespace: str | None = None) -> dict[str, Any]:
        """DELETE /admin/slow-queries — clear slow query log."""
        params: dict[str, Any] = {}
        if namespace is not None:
            params["namespace"] = namespace
        return await self._request("DELETE", "/v1/admin/slow-queries", params=params)

    async def admin_update_slow_query_config(self, **kwargs: Any) -> dict[str, Any]:
        """PATCH /admin/slow-queries/config — update slow query configuration."""
        return await self._request("PATCH", "/v1/admin/slow-queries/config", data=kwargs)

    # =========================================================================
    # Admin — Backups
    # =========================================================================

    async def admin_list_backups(self) -> dict[str, Any]:
        """GET /admin/backups — list all backups."""
        return await self._request("GET", "/v1/admin/backups")

    async def admin_create_backup(
        self,
        name: str,
        backup_type: str | None = None,
        namespaces: list[str] | None = None,
        encrypt: bool | None = None,
        compression: str | None = None,
    ) -> dict[str, Any]:
        """POST /admin/backups — create a new backup."""
        data: dict[str, Any] = {"name": name}
        if backup_type is not None:
            data["backup_type"] = backup_type
        if namespaces is not None:
            data["namespaces"] = namespaces
        if encrypt is not None:
            data["encrypt"] = encrypt
        if compression is not None:
            data["compression"] = compression
        return await self._request("POST", "/v1/admin/backups", data=data)

    async def admin_get_backup(self, backup_id: str) -> dict[str, Any]:
        """GET /admin/backups/{id} — get backup details."""
        return await self._request("GET", f"/v1/admin/backups/{backup_id}")

    async def admin_delete_backup(self, backup_id: str) -> dict[str, Any]:
        """DELETE /admin/backups/{id} — delete a backup."""
        return await self._request("DELETE", f"/v1/admin/backups/{backup_id}")

    async def admin_get_backup_schedule(self) -> dict[str, Any]:
        """GET /admin/backups/schedule — get backup schedule."""
        return await self._request("GET", "/v1/admin/backups/schedule")

    async def admin_update_backup_schedule(self, **kwargs: Any) -> dict[str, Any]:
        """POST /admin/backups/schedule — update backup schedule."""
        return await self._request("POST", "/v1/admin/backups/schedule", data=kwargs)

    async def admin_restore_backup(
        self,
        backup_id: str,
        target_namespaces: list[str] | None = None,
        overwrite: bool | None = None,
        point_in_time: int | None = None,
    ) -> dict[str, Any]:
        """POST /admin/backups/restore — restore from backup."""
        data: dict[str, Any] = {"backup_id": backup_id}
        if target_namespaces is not None:
            data["target_namespaces"] = target_namespaces
        if overwrite is not None:
            data["overwrite"] = overwrite
        if point_in_time is not None:
            data["point_in_time"] = point_in_time
        return await self._request("POST", "/v1/admin/backups/restore", data=data)

    async def admin_get_restore_status(self, restore_id: str) -> dict[str, Any]:
        """GET /admin/backups/restore/{id} — restore operation status."""
        return await self._request("GET", f"/v1/admin/backups/restore/{restore_id}")

    # =========================================================================
    # Ops — Diagnostics & Jobs
    # =========================================================================

    async def ops_diagnostics(self) -> dict[str, Any]:
        """GET /ops/diagnostics — system diagnostics."""
        return await self._request("GET", "/ops/diagnostics")

    async def ops_list_jobs(self) -> list[dict[str, Any]]:
        """GET /ops/jobs — list background jobs."""
        return await self._request("GET", "/ops/jobs")

    async def ops_get_job(self, job_id: str) -> dict[str, Any]:
        """GET /ops/jobs/{id} — get job status."""
        return await self._request("GET", f"/ops/jobs/{job_id}")

    async def ops_compact(
        self,
        namespace: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """POST /ops/compact — trigger compaction."""
        data: dict[str, Any] = {"force": force}
        if namespace is not None:
            data["namespace"] = namespace
        return await self._request("POST", "/ops/compact", data=data)

    async def ops_shutdown(self) -> dict[str, Any]:
        """POST /ops/shutdown — request graceful shutdown."""
        return await self._request("POST", "/ops/shutdown")

    # =========================================================================
    # Phase 3 — Engine Parity
    # =========================================================================

    async def fulltext_stats(self, namespace: str) -> FullTextIndexStats:
        """GET /v1/namespaces/{namespace}/fulltext/stats — full-text index statistics."""
        data = await self._request("GET", f"/v1/namespaces/{namespace}/fulltext/stats")
        return FullTextIndexStats.from_dict(data)

    async def fulltext_delete(self, namespace: str, ids: list[str]) -> dict[str, Any]:
        """Delete documents from the full-text index.

        Calls ``POST /v1/namespaces/{namespace}/fulltext/delete``.
        """
        return await self._request(
            "POST", f"/v1/namespaces/{namespace}/fulltext/delete", data={"ids": ids}
        )

    async def ttl_stats(self) -> TtlStatsResponse:
        """GET /admin/ttl/stats — TTL expiration statistics across namespaces."""
        data = await self._request("GET", "/v1/admin/ttl/stats")
        return TtlStatsResponse.from_dict(data)

    async def ttl_cleanup(self, namespace: str | None = None) -> TtlCleanupResponse:
        """POST /admin/ttl/cleanup — remove expired vectors, optionally scoped to a namespace."""
        body: dict[str, Any] = {}
        if namespace is not None:
            body["namespace"] = namespace
        data = await self._request("POST", "/v1/admin/ttl/cleanup", data=body if body else None)
        return TtlCleanupResponse.from_dict(data)

    async def route_query(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.3,
        model: str | None = None,
    ) -> RouteResponse:
        """POST /v1/route — semantic query routing."""
        data: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "min_similarity": min_similarity,
        }
        if model is not None:
            data["model"] = model
        resp = await self._request("POST", "/v1/route", data=data)
        return RouteResponse.from_dict(resp)

    async def import_job_status(self, job_id: str) -> ImportJobStatus:
        """GET /v1/import/{job_id}/status — check import job progress."""
        data = await self._request("GET", f"/v1/import/{job_id}/status")
        return ImportJobStatus.from_dict(data)

    async def download_backup(self, backup_id: str) -> bytes:
        """GET /admin/backups/{id}/download — download a backup as gzip bytes."""
        url = self._url(f"/v1/admin/backups/{backup_id}/download")
        response = await self._client.get(url)
        response.raise_for_status()
        return response.content

    async def upload_backup(self, data: bytes) -> dict[str, Any]:
        """POST /admin/backups/upload — upload a gzip backup."""
        url = self._url("/v1/admin/backups/upload")
        response = await self._client.post(
            url,
            content=data,
            headers={"Content-Type": "application/gzip"},
        )
        response.raise_for_status()
        return response.json()

    async def storage_tier_overview(self) -> StorageTierOverview:
        """GET /admin/storage/tiers — storage tier architecture overview."""
        data = await self._request("GET", "/v1/admin/storage/tiers")
        return StorageTierOverview.from_dict(data)

    async def background_activity(self) -> dict[str, Any]:
        """GET /admin/background-activity — current background tasks and jobs."""
        return await self._request("GET", "/v1/admin/background-activity")

    async def memory_type_stats(self) -> MemoryTypeStatsResponse:
        """GET /admin/memory-type-stats — memory type distribution statistics."""
        data = await self._request("GET", "/v1/admin/memory-type-stats")
        return MemoryTypeStatsResponse.from_dict(data)

    async def migrate_namespace_dimensions(
        self,
        target_dimension: int = 1024,
        namespaces: list[str] | None = None,
    ) -> MigrateDimensionsResponse:
        """POST /admin/namespaces/migrate-dimensions — migrate namespace vector dimensions."""
        data: dict[str, Any] = {"target_dimension": target_dimension}
        if namespaces is not None:
            data["namespaces"] = namespaces
        resp = await self._request("POST", "/v1/admin/namespaces/migrate-dimensions", data=data)
        return MigrateDimensionsResponse.from_dict(resp)

    async def drain_reembed(
        self,
        timeout_secs: int | None = None,
        batch_size: int | None = None,
        min_importance: float | None = None,
    ) -> DrainReembedResponse:
        """``POST /admin/reembed/drain`` — drain static vectors to full ONNX quality (v0.11.82+).

        Async variant of :meth:`DakeraClient.drain_reembed`.
        Requires Admin scope.

        Args:
            timeout_secs: Hard wall-clock cap in seconds (default 600).
            batch_size: Candidates upgraded per cycle (default 10000).
            min_importance: Minimum importance threshold (default 0.0).

        Returns:
            :class:`DrainReembedResponse` with ``remaining=0`` on a full drain.
        """
        body: dict[str, Any] = {}
        if timeout_secs is not None:
            body["timeout_secs"] = timeout_secs
        if batch_size is not None:
            body["batch_size"] = batch_size
        if min_importance is not None:
            body["min_importance"] = min_importance
        resp = await self._request("POST", "/v1/admin/reembed/drain", data=body if body else None)
        return DrainReembedResponse.from_dict(resp)

    async def admin_reembed_static_count(self) -> StaticCountResponse:
        """``GET /admin/reembed/static-count`` — static vectors pending re-embedding.

        Async variant of :meth:`DakeraClient.admin_reembed_static_count`.
        Requires Admin scope.

        Returns:
            :class:`StaticCountResponse` with ``static_count`` field.
        """
        resp = await self._request("GET", "/v1/admin/reembed/static-count")
        return StaticCountResponse.from_dict(resp)

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self) -> AsyncDakeraClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and close client."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
