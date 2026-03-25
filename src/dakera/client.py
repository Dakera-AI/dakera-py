"""
Dakera Client

Main client class for interacting with Dakera server.
"""

import json
import random
import time
from collections.abc import Generator
from typing import Any, Optional
from urllib.parse import urljoin

import requests

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
    BatchForgetRequest,
    BatchForgetResponse,
    BatchRecallRequest,
    BatchRecallResponse,
    BatchTextQueryResponse,
    ConfigureNamespaceRequest,
    ConfigureNamespaceResponse,
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
    RateLimitHeaders,
    ReadConsistency,
    RetryConfig,
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


class DakeraClient:
    """
    Client for interacting with Dakera AI memory platform.

    Example:
        >>> client = DakeraClient("http://localhost:3000")
        >>> client.upsert("my-namespace", vectors=[
        ...     {"id": "vec1", "values": [0.1, 0.2, 0.3]},
        ...     {"id": "vec2", "values": [0.4, 0.5, 0.6]},
        ... ])
        >>> results = client.query("my-namespace", vector=[0.1, 0.2, 0.3], top_k=5)
        >>> for result in results.results:
        ...     print(f"{result.id}: {result.score}")
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        connect_timeout: Optional[float] = None,
        max_retries: int = 3,
        retry_config: Optional[RetryConfig] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Initialize Dakera client.

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
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = connect_timeout if connect_timeout is not None else timeout

        # Build effective RetryConfig
        if retry_config is not None:
            self._retry_config = retry_config
        else:
            self._retry_config = RetryConfig(max_retries=max_retries)

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

        if headers:
            self._session.headers.update(headers)

        # OPS-1: last seen rate-limit headers (updated after every response)
        self._last_rate_limit_headers: Optional[RateLimitHeaders] = None

    @property
    def last_rate_limit_headers(self) -> Optional[RateLimitHeaders]:
        """Rate-limit headers from the most recent API response (OPS-1).

        Returns ``None`` until the first successful request has been made.
        """
        return self._last_rate_limit_headers

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _handle_response(self, response: requests.Response) -> Any:
        """Handle API response and raise appropriate exceptions."""
        # OPS-1: capture rate-limit headers before consuming the body
        self._last_rate_limit_headers = RateLimitHeaders.from_headers(
            dict(response.headers)
        )

        try:
            body = response.json() if response.content else None
        except json.JSONDecodeError:
            body = response.text

        if response.status_code == 200 or response.status_code == 201:
            return body
        elif response.status_code == 204:
            return None

        raw_code = body.get("code") if isinstance(body, dict) else None
        try:
            error_code = ErrorCode(raw_code) if raw_code is not None else ErrorCode.UNKNOWN
        except ValueError:
            error_code = ErrorCode.UNKNOWN

        if response.status_code == 400:
            raise ValidationError(
                message=(
                    body.get("error", "Validation error")
                    if isinstance(body, dict) else str(body)
                ),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        elif response.status_code == 401:
            raise AuthenticationError(
                message=(
                    body.get("error", "Authentication failed")
                    if isinstance(body, dict) else "Authentication failed"
                ),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        elif response.status_code == 403:
            raise AuthorizationError(
                message=(
                    body.get("error", "Forbidden")
                    if isinstance(body, dict) else "Forbidden"
                ),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        elif response.status_code == 404:
            raise NotFoundError(
                message=(
                    body.get("error", "Resource not found")
                    if isinstance(body, dict) else str(body)
                ),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message="Rate limit exceeded",
                status_code=response.status_code,
                response_body=body,
                retry_after=int(retry_after) if retry_after else None,
            )
        elif response.status_code >= 500:
            raise ServerError(
                message=body.get("error", "Server error") if isinstance(body, dict) else str(body),
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )
        else:
            raise DakeraError(
                message=f"Unexpected status code: {response.status_code}",
                status_code=response.status_code,
                response_body=body,
                code=error_code,
            )

    @staticmethod
    def _compute_backoff(rc: RetryConfig, attempt: int) -> float:
        """Compute exponential backoff delay for the given attempt index."""
        delay = min(rc.max_delay, rc.base_delay * (2 ** attempt))
        if rc.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Make HTTP request with retry logic and exponential backoff."""
        url = self._url(path)
        rc = self._retry_config
        request_timeout = (self.connect_timeout, self.timeout)

        for attempt in range(rc.max_retries):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=request_timeout,
                )
                return self._handle_response(response)
            except requests.exceptions.ConnectionError as e:
                if attempt == rc.max_retries - 1:
                    raise ConnectionError(f"Failed to connect to {url}: {e}") from e
            except requests.exceptions.Timeout as e:
                if attempt == rc.max_retries - 1:
                    raise TimeoutError(f"Request timed out: {e}") from e
            except RateLimitError as e:
                if attempt == rc.max_retries - 1:
                    raise
                # Respect Retry-After header when present
                wait = (
                    float(e.retry_after) if e.retry_after is not None
                    else self._compute_backoff(rc, attempt)
                )
                time.sleep(wait)
                continue
            except ServerError:
                if attempt == rc.max_retries - 1:
                    raise
            except DakeraError:
                raise

            time.sleep(self._compute_backoff(rc, attempt))

        raise DakeraError("Request failed after retries")

    # =========================================================================
    # Vector Operations
    # =========================================================================

    def upsert(
        self,
        namespace: str,
        vectors: list[VectorInput],
    ) -> dict[str, Any]:
        """
        Upsert vectors into a namespace.

        Args:
            namespace: Target namespace
            vectors: List of vectors to upsert. Each vector should have 'id' and 'values',
                    optionally 'metadata'.

        Returns:
            Response containing upsert status

        Example:
            >>> client.upsert("my-namespace", vectors=[
            ...     {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"label": "a"}},
            ...     Vector(id="vec2", values=[0.4, 0.5, 0.6]),
            ... ])
        """
        vector_dicts = []
        for v in vectors:
            if isinstance(v, Vector):
                vector_dicts.append(v.to_dict())
            else:
                vector_dicts.append(v)

        return self._request(
            "POST",
            f"/v1/namespaces/{namespace}/vectors",
            data={"vectors": vector_dicts},
        )

    def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int = 10,
        filter: Optional[FilterDict] = None,
        include_values: bool = False,
        include_metadata: bool = True,
        distance_metric: Optional[DistanceMetric] = None,
        consistency: Optional[ReadConsistency] = None,
        staleness_config: Optional[StalenessConfig] = None,
    ) -> SearchResult:
        """
        Query vectors by similarity.

        Args:
            namespace: Target namespace
            vector: Query vector
            top_k: Number of results to return (default: 10)
            filter: Optional metadata filter
            include_values: Include vector values in results (default: False)
            include_metadata: Include metadata in results (default: True)
            distance_metric: Distance metric for similarity (cosine, euclidean, dot_product)
            consistency: Read consistency level (strong, eventual, bounded_staleness)
            staleness_config: Configuration for bounded staleness reads

        Returns:
            SearchResult containing matching vectors

        Example:
            >>> results = client.query("my-namespace", vector=[0.1, 0.2, 0.3], top_k=5)
            >>> for r in results.results:
            ...     print(f"{r.id}: {r.score}")
            >>> # With consistency options
            >>> results = client.query(
            ...     "my-namespace",
            ...     vector=[0.1, 0.2, 0.3],
            ...     consistency=ReadConsistency.STRONG,
            ...     distance_metric=DistanceMetric.COSINE,
            ... )
        """
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

        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/query",
            data=data,
        )
        return SearchResult.from_dict(response)

    def delete(
        self,
        namespace: str,
        ids: Optional[list[str]] = None,
        filter: Optional[FilterDict] = None,
        delete_all: bool = False,
    ) -> dict[str, Any]:
        """
        Delete vectors from a namespace.

        Args:
            namespace: Target namespace
            ids: List of vector IDs to delete
            filter: Delete vectors matching this filter
            delete_all: Delete all vectors in namespace

        Returns:
            Response containing deletion status

        Example:
            >>> client.delete("my-namespace", ids=["vec1", "vec2"])
            >>> client.delete("my-namespace", filter={"label": "obsolete"})
            >>> client.delete("my-namespace", delete_all=True)
        """
        data: dict[str, Any] = {}
        if ids:
            data["ids"] = ids
        if filter:
            data["filter"] = filter
        if delete_all:
            data["delete_all"] = True

        return self._request(
            "POST",
            f"/v1/namespaces/{namespace}/delete",
            data=data,
        )

    def fetch(
        self,
        namespace: str,
        ids: list[str],
        include_values: bool = True,
        include_metadata: bool = True,
    ) -> list[Vector]:
        """
        Fetch vectors by ID.

        Args:
            namespace: Target namespace
            ids: List of vector IDs to fetch
            include_values: Include vector values (default: True)
            include_metadata: Include metadata (default: True)

        Returns:
            List of Vector objects

        Example:
            >>> vectors = client.fetch("my-namespace", ids=["vec1", "vec2"])
        """
        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/fetch",
            data={
                "ids": ids,
                "include_values": include_values,
                "include_metadata": include_metadata,
            },
        )
        vectors = response.get("vectors", [])
        return [Vector.from_dict(v) for v in vectors]

    def batch_query(
        self,
        namespace: str,
        queries: list[dict[str, Any]],
    ) -> list[SearchResult]:
        """
        Execute multiple queries in a single request.

        Args:
            namespace: Target namespace
            queries: List of query specifications, each containing 'vector' and optional
                    'top_k', 'filter', 'include_values', 'include_metadata'

        Returns:
            List of SearchResult objects

        Example:
            >>> results = client.batch_query("my-namespace", queries=[
            ...     {"vector": [0.1, 0.2, 0.3], "top_k": 5},
            ...     {"vector": [0.4, 0.5, 0.6], "top_k": 3},
            ... ])
        """
        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/batch-query",
            data={"queries": queries},
        )
        return [SearchResult.from_dict(r) for r in response.get("results", [])]

    # =========================================================================
    # Text-Based Inference Operations (Auto-Embedding)
    # =========================================================================

    def upsert_text(
        self,
        namespace: str,
        documents: list[TextDocumentInput],
        model: Optional[EmbeddingModel] = None,
    ) -> TextUpsertResponse:
        """
        Upsert text documents with automatic embedding generation.

        The text is embedded using the specified model (default: MiniLM)
        and stored as vectors.

        Args:
            namespace: Target namespace
            documents: List of text documents to upsert
            model: Embedding model to use (default: minilm)

        Returns:
            TextUpsertResponse containing upsert status and timing info

        Example:
            >>> response = client.upsert_text("my-namespace", documents=[
            ...     {"id": "doc1", "text": "Hello world", "metadata": {"label": "greeting"}},
            ...     TextDocument(id="doc2", text="Goodbye world"),
            ... ])
            >>> print(f"Upserted {response.upserted_count} documents")
        """
        doc_dicts = []
        for d in documents:
            if isinstance(d, TextDocument):
                doc_dicts.append(d.to_dict())
            else:
                doc_dicts.append(d)

        data: dict[str, Any] = {"documents": doc_dicts}
        if model:
            data["model"] = model.value

        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/upsert-text",
            data=data,
        )
        return TextUpsertResponse.from_dict(response)

    def query_text(
        self,
        namespace: str,
        text: str,
        top_k: int = 10,
        filter: Optional[FilterDict] = None,
        include_text: bool = True,
        include_vectors: bool = False,
        model: Optional[EmbeddingModel] = None,
    ) -> TextQueryResponse:
        """
        Query using natural language text with automatic embedding.

        The query text is embedded and used for similarity search.

        Args:
            namespace: Target namespace
            text: Query text to search for
            top_k: Number of results to return (default: 10)
            filter: Optional metadata filter
            include_text: Include original text in results (default: True)
            include_vectors: Include vectors in results (default: False)
            model: Embedding model to use (default: minilm)

        Returns:
            TextQueryResponse containing results and timing info

        Example:
            >>> response = client.query_text("my-namespace", text="greeting message")
            >>> for result in response.results:
            ...     print(f"{result.id}: {result.score} - {result.text}")
        """
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

        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/query-text",
            data=data,
        )
        return TextQueryResponse.from_dict(response)

    def batch_query_text(
        self,
        namespace: str,
        queries: list[str],
        top_k: int = 10,
        filter: Optional[FilterDict] = None,
        include_vectors: bool = False,
        model: Optional[EmbeddingModel] = None,
    ) -> BatchTextQueryResponse:
        """
        Batch query using multiple text queries with automatic embedding.

        Args:
            namespace: Target namespace
            queries: List of query texts
            top_k: Number of results per query (default: 10)
            filter: Optional metadata filter applied to all queries
            include_vectors: Include vectors in results (default: False)
            model: Embedding model to use (default: minilm)

        Returns:
            BatchTextQueryResponse containing results for each query

        Example:
            >>> response = client.batch_query_text("my-namespace", queries=[
            ...     "greeting message",
            ...     "farewell message",
            ... ])
            >>> for i, query_results in enumerate(response.results):
            ...     print(f"Query {i}: {len(query_results)} results")
        """
        data: dict[str, Any] = {
            "queries": queries,
            "top_k": top_k,
            "include_vectors": include_vectors,
        }
        if filter:
            data["filter"] = filter
        if model:
            data["model"] = model.value

        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/batch-query-text",
            data=data,
        )
        return BatchTextQueryResponse.from_dict(response)

    # =========================================================================
    # Full-Text Search Operations
    # =========================================================================

    def index_documents(
        self,
        namespace: str,
        documents: list[DocumentInput],
    ) -> dict[str, Any]:
        """
        Index documents for full-text search.

        Args:
            namespace: Target namespace
            documents: List of documents to index

        Returns:
            Response containing indexing status

        Example:
            >>> client.index_documents("my-namespace", documents=[
            ...     {"id": "doc1", "content": "Hello world"},
            ...     Document(id="doc2", content="Goodbye world"),
            ... ])
        """
        doc_dicts = []
        for d in documents:
            if isinstance(d, Document):
                doc_dicts.append(d.to_dict())
            else:
                doc_dicts.append(d)

        return self._request(
            "POST",
            f"/v1/namespaces/{namespace}/fulltext/index",
            data={"documents": doc_dicts},
        )

    def fulltext_search(
        self,
        namespace: str,
        query: str,
        top_k: int = 10,
        filter: Optional[FilterDict] = None,
    ) -> list[FullTextSearchResult]:
        """
        Perform full-text search.

        Args:
            namespace: Target namespace
            query: Search query string
            top_k: Number of results to return (default: 10)
            filter: Optional metadata filter

        Returns:
            List of FullTextSearchResult objects

        Example:
            >>> results = client.fulltext_search("my-namespace", query="hello world")
        """
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if filter:
            data["filter"] = filter

        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/fulltext/search",
            data=data,
        )
        return [FullTextSearchResult.from_dict(r) for r in response.get("results", [])]

    def hybrid_search(
        self,
        namespace: str,
        query: str,
        vector: Optional[list[float]] = None,
        top_k: int = 10,
        alpha: float = 0.5,
        filter: Optional[FilterDict] = None,
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search combining vector and full-text.

        When ``vector`` is omitted the server falls back to BM25-only full-text
        search. When provided, results are blended with vector similarity
        according to ``alpha``.

        Args:
            namespace: Target namespace
            query: Text query string
            vector: Optional query vector. Omit for BM25-only search.
            top_k: Number of results to return (default: 10)
            alpha: Balance between vector (0) and text (1) search (default: 0.5)
            filter: Optional metadata filter

        Returns:
            List of HybridSearchResult objects

        Example:
            >>> # Hybrid (vector + text)
            >>> results = client.hybrid_search(
            ...     "my-namespace",
            ...     query="hello world",
            ...     vector=[0.1, 0.2, 0.3],
            ...     alpha=0.7,
            ... )
            >>> # BM25-only (no vector)
            >>> results = client.hybrid_search("my-namespace", query="hello world")
        """
        data: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "alpha": alpha,
        }
        if vector is not None:
            data["vector"] = vector
        if filter:
            data["filter"] = filter

        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/hybrid",
            data=data,
        )
        return [HybridSearchResult.from_dict(r) for r in response.get("results", [])]

    # =========================================================================
    # Namespace Operations
    # =========================================================================

    def list_namespaces(self) -> list[NamespaceInfo]:
        """
        List all namespaces.

        Returns:
            List of NamespaceInfo objects
        """
        response = self._request("GET", "/v1/namespaces")
        return [NamespaceInfo.from_dict(ns) for ns in response.get("namespaces", [])]

    def get_namespace(self, namespace: str) -> NamespaceInfo:
        """
        Get namespace information.

        Args:
            namespace: Namespace name

        Returns:
            NamespaceInfo object
        """
        response = self._request("GET", f"/v1/namespaces/{namespace}")
        return NamespaceInfo.from_dict(response)

    def create_namespace(
        self,
        namespace: str,
        dimensions: Optional[int] = None,
        index_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> NamespaceInfo:
        """
        Create a new namespace.

        Args:
            namespace: Namespace name
            dimensions: Vector dimensions (optional, can be inferred from first upsert)
            index_type: Index type (e.g., "flat", "hnsw", "ivf")
            metadata: Optional namespace metadata

        Returns:
            NamespaceInfo object
        """
        data: dict[str, Any] = {"name": namespace}
        if dimensions:
            data["dimensions"] = dimensions
        if index_type:
            data["index_type"] = index_type
        if metadata:
            data["metadata"] = metadata

        response = self._request("POST", "/v1/namespaces", data=data)
        return NamespaceInfo.from_dict(response)

    def configure_namespace(
        self,
        namespace: str,
        dimension: int,
        distance: Optional[DistanceMetric] = None,
    ) -> ConfigureNamespaceResponse:
        """
        Create or update a namespace configuration (upsert semantics).

        Creates the namespace if it does not exist, or updates its distance
        metric configuration if it already exists.  Replaces the need for
        separate create + patch calls.  Requires ``Scope::Write``.

        Args:
            namespace: Namespace name
            dimension: Vector dimension. Must match existing dimension on updates.
            distance: Distance metric (default: cosine).

        Returns:
            ConfigureNamespaceResponse with ``created=True`` if newly created.
        """
        req = ConfigureNamespaceRequest(dimension=dimension, distance=distance)
        response = self._request("PUT", f"/v1/namespaces/{namespace}", data=req.to_dict())
        return ConfigureNamespaceResponse.from_dict(response)

    def delete_namespace(self, namespace: str) -> None:
        """
        Delete a namespace and all its data.

        Args:
            namespace: Namespace name
        """
        self._request("DELETE", f"/v1/namespaces/{namespace}")

    # =========================================================================
    # Admin Operations
    # =========================================================================

    def health(self) -> dict[str, Any]:
        """
        Check server health status.

        Returns:
            Health status dictionary
        """
        return self._request("GET", "/health")

    def get_index_stats(self, namespace: str) -> IndexStats:
        """
        Get index statistics for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            IndexStats object
        """
        response = self._request("GET", f"/v1/namespaces/{namespace}/stats")
        return IndexStats.from_dict(response)

    def compact(self, namespace: str) -> dict[str, Any]:
        """
        Trigger compaction for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Compaction status
        """
        return self._request("POST", f"/v1/namespaces/{namespace}/compact")

    def flush(self, namespace: str) -> dict[str, Any]:
        """
        Flush pending writes for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Flush status
        """
        return self._request("POST", f"/v1/namespaces/{namespace}/flush")

    # =========================================================================
    # Memory Operations
    # =========================================================================

    def store_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "episodic",
        importance: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        expires_at: Optional[int] = None,
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
            ttl_seconds: Optional TTL in seconds. The memory is hard-deleted after
                this many seconds from creation.
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
        if ttl_seconds is not None:
            data["ttl_seconds"] = ttl_seconds
        if expires_at is not None:
            data["expires_at"] = expires_at
        return self._request("POST", f"/v1/agents/{agent_id}/memories", data=data)

    def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Recall memories for an agent."""
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        result = self._request("POST", f"/v1/agents/{agent_id}/memories/recall", data=data)
        return result.get("memories", result) if isinstance(result, dict) else result

    def get_memory(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Get a specific memory."""
        return self._request("GET", f"/v1/agents/{agent_id}/memories/{memory_id}")

    def update_memory(
        self,
        agent_id: str,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        memory_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update an existing memory."""
        data: dict[str, Any] = {}
        if content is not None:
            data["content"] = content
        if metadata is not None:
            data["metadata"] = metadata
        if memory_type is not None:
            data["memory_type"] = memory_type
        return self._request("PUT", f"/v1/agents/{agent_id}/memories/{memory_id}", data=data)

    def forget(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Delete a memory."""
        return self._request("DELETE", f"/v1/agents/{agent_id}/memories/{memory_id}")

    def batch_recall(self, request: BatchRecallRequest) -> BatchRecallResponse:
        """Bulk-recall memories using filter predicates (CE-2).

        Uses ``POST /v1/memories/recall/batch`` — no embedding required.

        Args:
            request: Batch recall parameters including ``agent_id``, optional
                ``filter`` predicates, and ``limit``.

        Returns:
            :class:`BatchRecallResponse` containing matched memories, total
            count in the namespace, and count after filtering.

        Example:
            >>> filt = BatchMemoryFilter(tags=["preferences"], min_importance=0.7)
            >>> resp = client.batch_recall(BatchRecallRequest("agent-1", filter=filt, limit=50))
            >>> print(f"Found {resp.filtered} memories")
        """
        result = self._request("POST", "/v1/memories/recall/batch", data=request.to_dict())
        return BatchRecallResponse.from_dict(result)

    def batch_forget(self, request: BatchForgetRequest) -> BatchForgetResponse:
        """Bulk-delete memories using filter predicates (CE-2).

        Uses ``DELETE /v1/memories/forget/batch``.  The server requires at
        least one filter predicate to be set as a safety guard.

        Args:
            request: Batch forget parameters including ``agent_id`` and
                ``filter`` predicates (at least one required).

        Returns:
            :class:`BatchForgetResponse` with the number of deleted memories.

        Example:
            >>> filt = BatchMemoryFilter(created_before=1700000000)
            >>> resp = client.batch_forget(BatchForgetRequest("agent-1", filter=filt))
            >>> print(f"Deleted {resp.deleted_count} memories")
        """
        result = self._request(
            "DELETE", "/v1/memories/forget/batch", data=request.to_dict()
        )
        return BatchForgetResponse.from_dict(result)

    def search_memories(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Search memories for an agent."""
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        result = self._request("POST", f"/v1/agents/{agent_id}/memories/search", data=data)
        return result.get("memories", result) if isinstance(result, dict) else result

    def update_importance(
        self,
        agent_id: str,
        memory_ids: list[str],
        importance: float,
    ) -> dict[str, Any]:
        """Update importance of memories."""
        data = {"memory_ids": memory_ids, "importance": importance}
        return self._request("PUT", f"/v1/agents/{agent_id}/memories/importance", data=data)

    def consolidate(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Consolidate memories for an agent."""
        data: dict[str, Any] = {"dry_run": dry_run}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if threshold is not None:
            data["threshold"] = threshold
        return self._request("POST", f"/v1/agents/{agent_id}/memories/consolidate", data=data)

    def memory_feedback(
        self,
        agent_id: str,
        memory_id: str,
        feedback: str,
        relevance_score: Optional[float] = None,
    ) -> dict[str, Any]:
        """Submit feedback on a memory recall."""
        data: dict[str, Any] = {"memory_id": memory_id, "feedback": feedback}
        if relevance_score is not None:
            data["relevance_score"] = relevance_score
        return self._request("POST", f"/v1/agents/{agent_id}/memories/feedback", data=data)

    # =========================================================================
    # Session Operations
    # =========================================================================

    def start_session(
        self,
        agent_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Start a new session."""
        data: dict[str, Any] = {"agent_id": agent_id}
        if metadata is not None:
            data["metadata"] = metadata
        return self._request("POST", "/v1/sessions/start", data=data)

    def end_session(self, session_id: str) -> dict[str, Any]:
        """End a session."""
        return self._request("POST", f"/v1/sessions/{session_id}/end")

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get session details."""
        return self._request("GET", f"/v1/sessions/{session_id}")

    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        active_only: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
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
        return self._request("GET", "/v1/sessions", params=params)

    def session_memories(self, session_id: str) -> list[dict[str, Any]]:
        """Get memories for a session."""
        return self._request("GET", f"/v1/sessions/{session_id}/memories")

    # =========================================================================
    # Agent Operations
    # =========================================================================

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents."""
        return self._request("GET", "/v1/agents")

    def agent_memories(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get memories for an agent."""
        params: dict[str, Any] = {}
        if memory_type is not None:
            params["memory_type"] = memory_type
        if limit is not None:
            params["limit"] = limit
        return self._request("GET", f"/v1/agents/{agent_id}/memories", params=params)

    def agent_stats(self, agent_id: str) -> dict[str, Any]:
        """Get stats for an agent."""
        return self._request("GET", f"/v1/agents/{agent_id}/stats")

    def agent_sessions(
        self,
        agent_id: str,
        active_only: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get sessions for an agent."""
        params: dict[str, Any] = {}
        if active_only is not None:
            params["active_only"] = str(active_only).lower()
        if limit is not None:
            params["limit"] = limit
        return self._request("GET", f"/v1/agents/{agent_id}/sessions", params=params)

    # =========================================================================
    # Cache Warming Operations
    # =========================================================================

    def warm_cache(
        self,
        namespace: str,
        vector_ids: Optional[list[str]] = None,
        priority: WarmingPriority = WarmingPriority.NORMAL,
        target_tier: WarmingTargetTier = WarmingTargetTier.L2,
        background: bool = False,
        ttl_hint_seconds: Optional[int] = None,
        access_pattern: AccessPatternHint = AccessPatternHint.RANDOM,
        max_vectors: Optional[int] = None,
    ) -> WarmCacheResponse:
        """
        Warm cache for vectors in a namespace.

        Args:
            namespace: Target namespace
            vector_ids: Specific vector IDs to warm (None = all vectors)
            priority: Warming priority level (default: NORMAL)
            target_tier: Target cache tier (l1, l2, or both; default: L2)
            background: Run warming in background (default: False)
            ttl_hint_seconds: TTL hint for cached entries
            access_pattern: Access pattern hint for optimization
            max_vectors: Maximum number of vectors to warm

        Returns:
            WarmCacheResponse with warming status

        Example:
            >>> response = client.warm_cache(
            ...     "my-namespace",
            ...     priority=WarmingPriority.HIGH,
            ...     target_tier=WarmingTargetTier.BOTH,
            ... )
            >>> print(f"Warmed {response.entries_warmed} entries")
        """
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

        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/cache/warm",
            data=request.to_dict(),
        )
        return WarmCacheResponse.from_dict(response)

    # =========================================================================
    # Advanced Search Operations
    # =========================================================================

    def multi_vector_search(
        self,
        namespace: str,
        positive: list[list[float]],
        negative: Optional[list[list[float]]] = None,
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        mmr_lambda: Optional[float] = None,
        mmr_prefetch_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Multi-vector search with positive/negative vectors and optional MMR.

        Args:
            namespace: Target namespace
            positive: List of positive query vectors
            negative: Optional list of negative query vectors
            top_k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Include metadata in results
            include_vectors: Include vector values in results
            mmr_lambda: MMR diversity parameter (0.0 = max diversity, 1.0 = max relevance)
            mmr_prefetch_k: Number of candidates to prefetch for MMR

        Returns:
            Dict with results and search metadata
        """
        data: dict[str, Any] = {
            "positive": positive,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_vectors": include_vectors,
        }
        if negative is not None:
            data["negative"] = negative
        if filter is not None:
            data["filter"] = filter
        if mmr_lambda is not None:
            data["mmr_lambda"] = mmr_lambda
        if mmr_prefetch_k is not None:
            data["mmr_prefetch_k"] = mmr_prefetch_k
        return self._request("POST", f"/v1/namespaces/{namespace}/multi-vector", data=data)

    def unified_query(
        self,
        namespace: str,
        vector: Optional[list[float]] = None,
        text: Optional[str] = None,
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        vector_weight: Optional[float] = None,
        text_weight: Optional[float] = None,
        fusion_method: Optional[str] = None,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """
        Unified query combining vector and text search.

        Args:
            namespace: Target namespace
            vector: Optional query vector
            text: Optional text query
            top_k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Include metadata in results
            include_vectors: Include vector values in results
            vector_weight: Weight for vector search component
            text_weight: Weight for text search component
            fusion_method: Fusion method (e.g. "rrf", "linear")
            rerank: Whether to rerank results

        Returns:
            Dict with results and search metadata
        """
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
        if filter is not None:
            data["filter"] = filter
        if vector_weight is not None:
            data["vector_weight"] = vector_weight
        if text_weight is not None:
            data["text_weight"] = text_weight
        if fusion_method is not None:
            data["fusion_method"] = fusion_method
        return self._request("POST", f"/v1/namespaces/{namespace}/unified-query", data=data)

    def aggregate(
        self,
        namespace: str,
        vector: Optional[list[float]] = None,
        group_by: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
        top_groups: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Aggregate vectors with grouping.

        Args:
            namespace: Target namespace
            vector: Optional query vector for similarity-based aggregation
            group_by: Metadata field to group by
            metrics: List of aggregation metrics (e.g. ["count", "avg_score"])
            top_k: Number of results per group
            filter: Optional metadata filter
            top_groups: Maximum number of groups to return

        Returns:
            Dict with aggregation groups and metadata
        """
        data: dict[str, Any] = {}
        if vector is not None:
            data["vector"] = vector
        if group_by is not None:
            data["group_by"] = group_by
        if metrics is not None:
            data["metrics"] = metrics
        if top_k is not None:
            data["top_k"] = top_k
        if filter is not None:
            data["filter"] = filter
        if top_groups is not None:
            data["top_groups"] = top_groups
        return self._request("POST", f"/v1/namespaces/{namespace}/aggregate", data=data)

    def export_vectors(
        self,
        namespace: str,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
        include_vectors: bool = True,
    ) -> dict[str, Any]:
        """
        Export vectors with pagination.

        Args:
            namespace: Target namespace
            cursor: Pagination cursor from previous response
            limit: Maximum number of vectors to return
            filter: Optional metadata filter
            include_vectors: Include vector values in export

        Returns:
            Dict with exported vectors and next_cursor for pagination
        """
        data: dict[str, Any] = {
            "include_vectors": include_vectors,
        }
        if cursor is not None:
            data["cursor"] = cursor
        if limit is not None:
            data["limit"] = limit
        if filter is not None:
            data["filter"] = filter
        return self._request("POST", f"/v1/namespaces/{namespace}/export", data=data)

    def explain_query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Explain query execution plan and performance.

        Args:
            namespace: Target namespace
            vector: Query vector
            top_k: Number of results
            filter: Optional metadata filter
            include_metadata: Include metadata in results

        Returns:
            Dict with query plan, execution steps, and timing information
        """
        data: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }
        if filter is not None:
            data["filter"] = filter
        return self._request("POST", f"/v1/namespaces/{namespace}/explain", data=data)

    def upsert_columns(
        self,
        namespace: str,
        ids: list[str],
        vectors: list[list[float]],
        attributes: Optional[dict[str, list[Any]]] = None,
        ttl_seconds: Optional[int] = None,
        dimension: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Column-format vector upsert for efficient bulk operations.

        Args:
            namespace: Target namespace
            ids: List of vector IDs
            vectors: List of vector value arrays
            attributes: Optional column attributes (key -> list of values)
            ttl_seconds: Optional TTL in seconds for all vectors
            dimension: Optional expected dimension for validation

        Returns:
            Dict with upsert status
        """
        data: dict[str, Any] = {
            "ids": ids,
            "vectors": vectors,
        }
        if attributes is not None:
            data["attributes"] = attributes
        if ttl_seconds is not None:
            data["ttl_seconds"] = ttl_seconds
        if dimension is not None:
            data["dimension"] = dimension
        return self._request("POST", f"/v1/namespaces/{namespace}/upsert-columns", data=data)

    # =========================================================================
    # Knowledge Graph Operations
    # =========================================================================

    def knowledge_graph(
        self,
        agent_id: str,
        memory_id: Optional[str] = None,
        depth: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> dict[str, Any]:
        """Build a knowledge graph from a seed memory."""
        data: dict[str, Any] = {"agent_id": agent_id}
        if memory_id is not None:
            data["memory_id"] = memory_id
        if depth is not None:
            data["depth"] = depth
        if min_similarity is not None:
            data["min_similarity"] = min_similarity
        return self._request("POST", "/v1/knowledge/graph", data=data)

    def full_knowledge_graph(
        self,
        agent_id: str,
        max_nodes: Optional[int] = None,
        min_similarity: Optional[float] = None,
        cluster_threshold: Optional[float] = None,
        max_edges_per_node: Optional[int] = None,
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
        return self._request("POST", "/v1/knowledge/graph/full", data=data)

    def summarize(
        self,
        agent_id: str,
        memory_ids: Optional[list[str]] = None,
        target_type: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Summarize memories."""
        data: dict[str, Any] = {"agent_id": agent_id, "dry_run": dry_run}
        if memory_ids is not None:
            data["memory_ids"] = memory_ids
        if target_type is not None:
            data["target_type"] = target_type
        return self._request("POST", "/v1/knowledge/summarize", data=data)

    def deduplicate(
        self,
        agent_id: str,
        threshold: Optional[float] = None,
        memory_type: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Deduplicate memories."""
        data: dict[str, Any] = {"agent_id": agent_id, "dry_run": dry_run}
        if threshold is not None:
            data["threshold"] = threshold
        if memory_type is not None:
            data["memory_type"] = memory_type
        return self._request("POST", "/v1/knowledge/deduplicate", data=data)

    # =========================================================================
    # Analytics Operations
    # =========================================================================

    def analytics_overview(
        self,
        period: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get analytics overview."""
        params: dict[str, Any] = {}
        if period is not None:
            params["period"] = period
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("GET", "/v1/analytics/overview", params=params)

    def analytics_latency(
        self,
        period: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get latency analytics."""
        params: dict[str, Any] = {}
        if period is not None:
            params["period"] = period
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("GET", "/v1/analytics/latency", params=params)

    def analytics_throughput(
        self,
        period: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get throughput analytics."""
        params: dict[str, Any] = {}
        if period is not None:
            params["period"] = period
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("GET", "/v1/analytics/throughput", params=params)

    def analytics_storage(
        self,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get storage analytics."""
        params: dict[str, Any] = {}
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("GET", "/v1/analytics/storage", params=params)

    # =========================================================================
    # Admin Operations (Extended)
    # =========================================================================

    def ops_stats(self) -> dict[str, Any]:
        """Get server stats (version, total_vectors, namespace_count, uptime_seconds, timestamp, state).

        Requires Read scope — works with read-only API keys, unlike cluster_status.
        The ``state`` field is ``"healthy"`` when storage is accessible, ``"degraded"`` otherwise.
        """
        return self._request("GET", "/v1/ops/stats")

    def cluster_status(self) -> dict[str, Any]:
        """Get cluster status."""
        return self._request("GET", "/v1/admin/cluster/status")

    def cluster_nodes(self) -> list[dict[str, Any]]:
        """Get cluster nodes."""
        return self._request("GET", "/v1/admin/cluster/nodes")

    def optimize_namespace(self, namespace: str) -> dict[str, Any]:
        """Optimize a namespace."""
        return self._request("POST", f"/v1/admin/namespaces/{namespace}/optimize")

    def index_stats(self, namespace: str) -> dict[str, Any]:
        """Get index stats for a namespace."""
        return self._request("GET", f"/v1/admin/namespaces/{namespace}/index/stats")

    def rebuild_indexes(self, namespace: str) -> dict[str, Any]:
        """Rebuild indexes for a namespace."""
        return self._request("POST", f"/v1/admin/namespaces/{namespace}/index/rebuild")

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._request("GET", "/v1/admin/cache/stats")

    def cache_clear(self, namespace: Optional[str] = None) -> dict[str, Any]:
        """Clear cache, optionally for a specific namespace."""
        data: Optional[dict[str, Any]] = None
        if namespace is not None:
            data = {"namespace": namespace}
        return self._request("POST", "/v1/admin/cache/clear", data=data)

    def get_config(self) -> dict[str, Any]:
        """Get server configuration."""
        return self._request("GET", "/v1/admin/config")

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Update server configuration."""
        return self._request("PUT", "/v1/admin/config", data=config)

    def get_quotas(self) -> dict[str, Any]:
        """Get quota settings."""
        return self._request("GET", "/v1/admin/quotas")

    def update_quotas(self, quotas: dict[str, Any]) -> dict[str, Any]:
        """Update quota settings."""
        return self._request("PUT", "/v1/admin/quotas", data=quotas)

    def slow_queries(
        self,
        limit: Optional[int] = None,
        min_duration_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get slow queries."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if min_duration_ms is not None:
            params["min_duration_ms"] = min_duration_ms
        return self._request("GET", "/v1/admin/slow-queries", params=params if params else None)

    def create_backup(self, include_data: bool = True) -> dict[str, Any]:
        """Create a backup."""
        return self._request("POST", "/v1/admin/backups", data={"include_data": include_data})

    def list_backups(self) -> list[dict[str, Any]]:
        """List all backups."""
        return self._request("GET", "/v1/admin/backups")

    def restore_backup(self, backup_id: str) -> dict[str, Any]:
        """Restore a backup."""
        return self._request("POST", f"/v1/admin/backups/{backup_id}/restore")

    def delete_backup(self, backup_id: str) -> dict[str, Any]:
        """Delete a backup."""
        return self._request("DELETE", f"/v1/admin/backups/{backup_id}")

    def configure_ttl(
        self,
        namespace: str,
        ttl_seconds: int,
        strategy: Optional[str] = None,
    ) -> dict[str, Any]:
        """Configure TTL for a namespace."""
        data: dict[str, Any] = {"ttl_seconds": ttl_seconds}
        if strategy is not None:
            data["strategy"] = strategy
        return self._request("POST", f"/v1/admin/namespaces/{namespace}/ttl", data=data)

    def autopilot_status(self) -> dict[str, Any]:
        """Get AutoPilot status: current config and last-run statistics (PILOT-1)."""
        return self._request("GET", "/v1/admin/autopilot/status")

    def autopilot_update_config(
        self,
        enabled: Optional[bool] = None,
        dedup_threshold: Optional[float] = None,
        dedup_interval_hours: Optional[int] = None,
        consolidation_interval_hours: Optional[int] = None,
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
        return self._request("PUT", "/v1/admin/autopilot/config", data=data)

    def autopilot_trigger(self, action: str) -> dict[str, Any]:
        """Manually trigger an AutoPilot cycle (PILOT-3).

        Args:
            action: One of ``"dedup"``, ``"consolidate"``, or ``"all"``.
        """
        return self._request("POST", "/v1/admin/autopilot/trigger", data={"action": action})

    def decay_config(self) -> dict[str, Any]:
        """Get current decay engine configuration (DECAY-1).

        Returns the active decay strategy, half-life, and minimum importance
        threshold. Requires Admin scope.
        """
        return self._request("GET", "/v1/admin/decay/config")

    def decay_update_config(
        self,
        strategy: Optional[str] = None,
        half_life_hours: Optional[float] = None,
        min_importance: Optional[float] = None,
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
        return self._request("PUT", "/v1/admin/decay/config", data=data)

    def decay_stats(self) -> dict[str, Any]:
        """Get decay engine activity counters and last-cycle snapshot (DECAY-2).

        Returns cumulative totals (memories decayed/deleted, cycles run) and
        per-cycle statistics from the most recent run. Requires Admin scope.
        """
        return self._request("GET", "/v1/admin/decay/stats")

    # =========================================================================
    # API Key Operations
    # =========================================================================

    def create_key(
        self,
        name: str,
        permissions: Optional[list[str]] = None,
        expires_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new API key."""
        data: dict[str, Any] = {"name": name}
        if permissions is not None:
            data["permissions"] = permissions
        if expires_at is not None:
            data["expires_at"] = expires_at
        return self._request("POST", "/v1/keys", data=data)

    def list_keys(self) -> list[dict[str, Any]]:
        """List all API keys."""
        return self._request("GET", "/v1/keys")

    def get_key(self, key_id: str) -> dict[str, Any]:
        """Get an API key by ID."""
        return self._request("GET", f"/v1/keys/{key_id}")

    def delete_key(self, key_id: str) -> dict[str, Any]:
        """Delete an API key."""
        return self._request("DELETE", f"/v1/keys/{key_id}")

    def deactivate_key(self, key_id: str) -> dict[str, Any]:
        """Deactivate an API key."""
        return self._request("POST", f"/v1/keys/{key_id}/deactivate")

    def rotate_key(self, key_id: str) -> dict[str, Any]:
        """Rotate an API key."""
        return self._request("POST", f"/v1/keys/{key_id}/rotate")

    def key_usage(self, key_id: str) -> dict[str, Any]:
        """Get usage statistics for an API key."""
        return self._request("GET", f"/v1/keys/{key_id}/usage")

    # =========================================================================
    # SSE Streaming (CE-1)
    # =========================================================================

    def _parse_sse_block(self, block: str) -> Optional[DakeraEvent]:
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

    def stream_namespace_events(
        self,
        namespace: str,
        timeout: Optional[float] = None,
    ) -> Generator[DakeraEvent, None, None]:
        """Stream SSE events scoped to *namespace*.

        Opens a long-lived HTTP connection to ``GET /v1/namespaces/{namespace}/events``
        and yields :class:`~dakera.models.DakeraEvent` objects as they arrive.
        The generator runs until the connection is closed by the server or the
        caller breaks out of the loop.

        Requires a Read-scoped API key.

        Args:
            namespace: The namespace to subscribe to.
            timeout: Optional read timeout in seconds.  Defaults to no timeout
                so the connection stays open indefinitely.

        Yields:
            :class:`~dakera.models.DakeraEvent` — one per SSE event.

        Example::

            for event in client.stream_namespace_events("my-ns"):
                print(event.type, event)
                if event.type == "stream_lagged":
                    break  # reconnect
        """
        url = self._url(f"/v1/namespaces/{namespace}/events")
        yield from self._stream_sse(url, timeout)

    def stream_global_events(
        self,
        timeout: Optional[float] = None,
    ) -> Generator[DakeraEvent, None, None]:
        """Stream all system events from the global event bus.

        Opens a long-lived HTTP connection to ``GET /ops/events`` and yields
        :class:`~dakera.models.DakeraEvent` objects as they arrive.

        Requires an Admin-scoped API key.

        Args:
            timeout: Optional read timeout in seconds.

        Yields:
            :class:`~dakera.models.DakeraEvent` — one per SSE event.
        """
        url = self._url("/ops/events")
        yield from self._stream_sse(url, timeout)

    def _stream_sse(
        self,
        url: str,
        timeout: Optional[float],
    ) -> Generator[DakeraEvent, None, None]:
        """Low-level SSE streaming helper."""
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        response = self._session.get(
            url,
            headers=headers,
            stream=True,
            timeout=timeout,
        )
        # For streaming responses we cannot call _handle_response (it would
        # buffer the entire body). Instead check the status code directly.
        if not response.ok:
            # Consume the (small) error body and delegate to _handle_response.
            _ = response.content  # buffers the error payload
            self._handle_response(response)  # always raises

        buffer = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
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

    def stream_memory_events(
        self,
        timeout: Optional[float] = None,
    ) -> Generator[MemoryEvent, None, None]:
        """Stream memory lifecycle events from the DASH-B SSE endpoint.

        Opens a long-lived HTTP connection to ``GET /v1/events/stream`` and
        yields :class:`~dakera.models.MemoryEvent` objects as they arrive.

        Requires a Read-scoped API key.

        Event types: ``stored``, ``recalled``, ``forgotten``, ``consolidated``,
        ``importance_updated``, ``session_started``, ``session_ended``.

        Args:
            timeout: Optional read timeout in seconds.  ``None`` (default)
                means no timeout — the stream stays open indefinitely.

        Yields:
            :class:`~dakera.models.MemoryEvent` — one per SSE event.

        Example::

            for event in client.stream_memory_events():
                if event.event_type == "stored":
                    print(f"[{event.agent_id}] stored {event.memory_id}")
        """
        url = self._url("/v1/events/stream")
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        response = self._session.get(
            url,
            headers=headers,
            stream=True,
            timeout=timeout,
        )
        if not response.ok:
            _ = response.content
            self._handle_response(response)

        buffer = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
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
                    import json

                    try:
                        payload = json.loads("\n".join(data_lines))
                        yield MemoryEvent.from_dict(payload)
                    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                        pass

    # =========================================================================
    # DASH-A: Cross-Agent Network
    # =========================================================================

    def cross_agent_network(
        self,
        agent_ids: Optional[list[str]] = None,
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

            graph = client.cross_agent_network(min_similarity=0.5)
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

        data = self._request("POST", "/v1/knowledge/network/cross-agent", data=payload)
        return CrossAgentNetworkResponse.from_dict(data)

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> "DakeraClient":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close session."""
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
