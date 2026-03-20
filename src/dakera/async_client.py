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

import json
from typing import Any, AsyncGenerator

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "httpx is required for AsyncDakeraClient. "
        "Install it with: pip install dakera[async]"
    ) from exc

from dakera.exceptions import (
    AuthenticationError,
    ConnectionError,
    DakeraError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from dakera.models import (
    AccessPatternHint,
    BatchTextQueryResponse,
    CrossAgentNetworkResponse,
    DakeraEvent,
    DistanceMetric,
    Document,
    DocumentInput,
    EmbeddingModel,
    FilterDict,
    FullTextSearchResult,
    HybridSearchResult,
    IndexStats,
    MemoryEvent,
    NamespaceInfo,
    ReadConsistency,
    SearchResult,
    StalenessConfig,
    TextDocument,
    TextDocumentInput,
    TextQueryResponse,
    TextUpsertResponse,
    Vector,
    VectorInput,
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
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize async Dakera client.

        Args:
            base_url: Base URL of the Dakera server (e.g., "http://localhost:3000")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retries for failed requests (default: 3)
            headers: Additional headers to include in all requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        default_headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            default_headers.update(headers)

        self._client = httpx.AsyncClient(
            headers=default_headers,
            timeout=httpx.Timeout(timeout),
        )

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        return f"{self.base_url}/{path.lstrip('/')}"

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle API response and raise appropriate exceptions."""
        try:
            body = response.json() if response.content else None
        except json.JSONDecodeError:
            body = response.text

        if response.status_code in (200, 201):
            return body
        if response.status_code == 204:
            return None
        if response.status_code == 400:
            raise ValidationError(
                message=(
                    body.get("error", "Validation error")
                    if isinstance(body, dict)
                    else str(body)
                ),
                status_code=response.status_code,
                response_body=body,
            )
        if response.status_code == 401:
            raise AuthenticationError(
                message="Authentication failed",
                status_code=response.status_code,
                response_body=body,
            )
        if response.status_code == 404:
            raise NotFoundError(
                message=(
                    body.get("error", "Resource not found")
                    if isinstance(body, dict)
                    else str(body)
                ),
                status_code=response.status_code,
                response_body=body,
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
            )
        raise DakeraError(
            message=f"Unexpected status code: {response.status_code}",
            status_code=response.status_code,
            response_body=body,
        )

    async def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make async HTTP request with retry logic."""
        url = self._url(path)
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                )
                return self._handle_response(response)
            except httpx.ConnectError as e:
                last_exception = ConnectionError(f"Failed to connect to {url}: {e}")
            except httpx.TimeoutException as e:
                last_exception = TimeoutError(f"Request timed out: {e}")
            except (RateLimitError, ServerError) as e:
                if attempt == self.max_retries - 1:
                    raise
                last_exception = e
            except DakeraError:
                raise

        if last_exception:
            raise last_exception
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
        alpha: float = 0.5,
        filter: FilterDict | None = None,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search combining vector and full-text."""
        data: dict[str, Any] = {"vector": vector, "query": query, "top_k": top_k, "alpha": alpha}
        if filter:
            data["filter"] = filter
        response = await self._request(
            "POST",
            f"/v1/namespaces/{namespace}/fulltext/hybrid",
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

    async def delete_namespace(self, namespace: str) -> None:
        """Delete a namespace and all its data."""
        await self._request("DELETE", f"/v1/namespaces/{namespace}")

    # =========================================================================
    # Admin / Stats Operations
    # =========================================================================

    async def health(self) -> dict[str, Any]:
        """Check server health status."""
        return await self._request("GET", "/health")

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
    ) -> dict[str, Any]:
        """Store a memory for an agent."""
        data: dict[str, Any] = {"content": content, "memory_type": memory_type}
        if importance is not None:
            data["importance"] = importance
        if metadata is not None:
            data["metadata"] = metadata
        if session_id is not None:
            data["session_id"] = session_id
        return await self._request("POST", f"/v1/agents/{agent_id}/memories", data=data)

    async def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None,
        min_importance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories for an agent."""
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        result = await self._request("POST", f"/v1/agents/{agent_id}/memories/recall", data=data)
        return result.get("memories", result) if isinstance(result, dict) else result

    async def get_memory(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Get a specific memory."""
        return await self._request("GET", f"/v1/agents/{agent_id}/memories/{memory_id}")

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
        return await self._request("PUT", f"/v1/agents/{agent_id}/memories/{memory_id}", data=data)

    async def forget(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Delete a memory."""
        return await self._request("DELETE", f"/v1/agents/{agent_id}/memories/{memory_id}")

    async def search_memories(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
        min_importance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories for an agent."""
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        result = await self._request("POST", f"/v1/agents/{agent_id}/memories/search", data=data)
        return result.get("memories", result) if isinstance(result, dict) else result

    async def update_importance(
        self,
        agent_id: str,
        memory_ids: list[str],
        importance: float,
    ) -> dict[str, Any]:
        """Update importance of memories."""
        return await self._request(
            "PUT",
            f"/v1/agents/{agent_id}/memories/importance",
            data={"memory_ids": memory_ids, "importance": importance},
        )

    async def consolidate(
        self,
        agent_id: str,
        memory_type: str | None = None,
        threshold: float | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Consolidate memories for an agent."""
        data: dict[str, Any] = {"dry_run": dry_run}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if threshold is not None:
            data["threshold"] = threshold
        return await self._request("POST", f"/v1/agents/{agent_id}/memories/consolidate", data=data)

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
    # Session Operations
    # =========================================================================

    async def start_session(
        self,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new session."""
        data: dict[str, Any] = {"agent_id": agent_id}
        if metadata is not None:
            data["metadata"] = metadata
        return await self._request("POST", "/v1/sessions/start", data=data)

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """End a session."""
        return await self._request("POST", f"/v1/sessions/{session_id}/end")

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
