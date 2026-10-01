"""
Dakera Client

Main client class for interacting with Dakera server.
"""

import json
import os
import random
import time
from collections.abc import Generator
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any, BinaryIO, Union
from urllib.parse import urljoin

import requests

from dakera._util import _attachment_body, _audit_export_response, _memory_job_body
from dakera.exceptions import (
    AuthorizationError,
    ConnectionError,
    DakeraError,
    ErrorCode,
    NotFoundError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    TimeoutError,
    error_from_response,
    parse_retry_after,
)
from dakera.models import (
    AccessPatternHint,
    AgentFeedbackSummary,
    AttachmentContent,
    AttachmentInfo,
    AttachmentJob,
    AttachmentUploadResponse,
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
    JobAccepted,
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
    Record,
    RecordUpsertResponse,
    RecordView,
    RetryConfig,
    # SEC-3
    RotateEncryptionKeyResponse,
    RouteResponse,
    # CE-10
    RoutingMode,
    SearchResult,
    ServerCapabilities,
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
    wire_value,
)

# DAK-7617: default User-Agent so the engine can attribute Python SDK usage.
# Read from installed package metadata to avoid a circular import of ``dakera``
# (``__version__`` is defined after this module is imported in ``__init__``).
try:
    _USER_AGENT = f"dakera-py/{_pkg_version('dakera')}"
except PackageNotFoundError:  # pragma: no cover
    _USER_AGENT = "dakera-py/unknown"


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
        api_key: str | None = None,
        timeout: float = 30.0,
        connect_timeout: float | None = None,
        max_retries: int = 3,
        retry_config: RetryConfig | None = None,
        headers: dict[str, str] | None = None,
        ode_url: str | None = None,
        preflight: bool = False,
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
            ode_url: Base URL of the dakera-ode sidecar
                (e.g., ``"http://localhost:8080"``).  Required to call
                :meth:`extract_entities`.
            preflight: Validate the requested embedding model, index kind and
                distance metric against ``GET /v1/capabilities`` *before* sending
                a request, raising :class:`~dakera.exceptions.UnsupportedCapabilityError`
                naming what the server supports.  Capabilities are fetched lazily
                on first use and cached (see :meth:`capabilities`).  A server that
                predates the endpoint (404) disables the check silently.  When
                ``False`` (default) the check still runs whenever capabilities
                have already been fetched through :meth:`capabilities`.
        """
        self.base_url = base_url.rstrip("/")
        self.ode_url = ode_url.rstrip("/") if ode_url else None
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = connect_timeout if connect_timeout is not None else timeout

        # Build effective RetryConfig
        if retry_config is not None:
            self._retry_config = retry_config
        else:
            self._retry_config = RetryConfig(max_retries=max_retries)

        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
        )

        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

        if headers:
            self._session.headers.update(headers)

        # OPS-1: last seen rate-limit headers (updated after every response)
        self._last_rate_limit_headers: RateLimitHeaders | None = None

        # R9: per-instance capabilities cache + pre-flight validation switch
        self._preflight_enabled = preflight
        self._capabilities: ServerCapabilities | None = None
        self._capabilities_unavailable = False

    @property
    def last_rate_limit_headers(self) -> RateLimitHeaders | None:
        """Rate-limit headers from the most recent API response (OPS-1).

        Returns ``None`` until the first successful request has been made.
        """
        return self._last_rate_limit_headers

    # =========================================================================
    # Server capabilities (R9 / DAK-10004)
    # =========================================================================

    def capabilities(self, refresh: bool = False) -> ServerCapabilities:
        """
        What the connected server can do — ``GET /v1/capabilities`` (server v0.12+).

        Returns the models the server can load (and which one is active), index
        kinds, distance metrics, the search mode it runs, whether the R2
        ``records`` surface is enabled and whether a re-embed is still pending.
        The document is cached on this client instance; pass ``refresh=True`` to
        fetch it again.  Unknown fields and unknown strings in the document are
        kept (as unknown enum members) rather than rejected.

        Raises:
            NotFoundError: the server predates ``/v1/capabilities``.

        Example:
            >>> caps = client.capabilities()
            >>> caps.model_names
            ['bge-large', 'minilm', ...]
            >>> caps.supports_records, caps.reembed_pending
            (False, False)
        """
        if self._capabilities is None or refresh:
            response = self._request("GET", "/v1/capabilities")
            self._capabilities = ServerCapabilities.from_dict(response or {})
            self._capabilities_unavailable = False
        return self._capabilities

    def require_supported(self, kind: str, value: Any) -> None:
        """
        Raise :class:`~dakera.exceptions.UnsupportedCapabilityError` unless the
        server advertises ``value`` for ``kind``.

        ``kind`` is one of ``"model"``, ``"index_kind"``, ``"distance_metric"``,
        ``"search_mode"``, ``"query_language"``.  Fetches (and caches)
        capabilities on first use.  ``search_mode`` is process-wide on the
        server (``DAKERA_SEARCH_MODE``), so this is the pre-flight for tooling
        that configures it rather than for a per-request field.
        """
        self.capabilities().require(kind, value)

    def _preflight(self, kind: str, value: Any) -> None:
        """Validate ``value`` against cached capabilities before a request.

        Uses the cache when populated; fetches only when ``preflight=True`` was
        passed to the constructor.  A 404 (pre-0.12 server) disables the check
        for the lifetime of this client.
        """
        caps = self._capabilities
        if caps is None:
            if not self._preflight_enabled or self._capabilities_unavailable:
                return
            try:
                caps = self.capabilities()
            except NotFoundError:
                self._capabilities_unavailable = True
                return
        caps.require(kind, value)

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _handle_response(self, response: requests.Response, raw: bool = False) -> Any:
        """Handle API response and raise appropriate exceptions.

        With ``raw=True`` a successful response is returned as-is (binary
        downloads); errors are mapped the same way either way.
        """
        # OPS-1: capture rate-limit headers before consuming the body
        self._last_rate_limit_headers = RateLimitHeaders.from_headers(dict(response.headers))

        if raw and 200 <= response.status_code < 300:
            return response

        try:
            body = response.json() if response.content else None
        except (json.JSONDecodeError, ValueError):
            body = response.text

        if response.status_code == 204:
            return None
        if 200 <= response.status_code < 300:  # 200, 201 and the v0.12 job routes' 202
            return body

        raise error_from_response(response.status_code, body, response.headers.get("Retry-After"))

    @classmethod
    def _retry_delay(cls, rc: RetryConfig, retry_after: float | None, attempt: int) -> float:
        """Delay before the next attempt: the server's ``Retry-After`` when it sent
        one (capped at ``rc.max_delay``), else exponential backoff."""
        if retry_after is not None:
            return min(float(retry_after), rc.max_delay)
        return cls._compute_backoff(rc, attempt)

    @staticmethod
    def _compute_backoff(rc: RetryConfig, attempt: int) -> float:
        """Compute exponential backoff delay for the given attempt index."""
        delay = min(rc.max_delay, rc.base_delay * (2**attempt))
        if rc.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        raw: bool = False,
    ) -> Any:
        """Make HTTP request with retry logic and exponential backoff.

        ``content`` sends a raw body (with ``headers``, e.g. its ``Content-Type``)
        instead of JSON; ``raw=True`` returns the successful response object
        instead of its parsed body. A ``503`` / ``429`` waits for the server's
        ``Retry-After`` (capped at the retry config's ``max_delay``) before the
        next attempt; other transient errors use exponential backoff.
        """
        url = self._url(path)
        rc = self._retry_config
        request_timeout = (self.connect_timeout, self.timeout)

        for attempt in range(rc.max_retries):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=data if content is None else None,
                    data=content,
                    headers=headers,
                    params=params,
                    timeout=request_timeout,
                )
                return self._handle_response(response, raw=raw)
            except requests.exceptions.ConnectionError as e:
                if attempt == rc.max_retries - 1:
                    raise ConnectionError(f"Failed to connect to {url}: {e}") from e
            except requests.exceptions.Timeout as e:
                if attempt == rc.max_retries - 1:
                    raise TimeoutError(f"Request timed out: {e}") from e
            except (RateLimitError, ServiceUnavailableError) as e:
                if attempt == rc.max_retries - 1:
                    raise
                time.sleep(self._retry_delay(rc, e.retry_after, attempt))
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
        filter: FilterDict | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
        distance_metric: DistanceMetric | None = None,
        consistency: ReadConsistency | None = None,
        staleness_config: StalenessConfig | None = None,
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
            "include_vectors": include_values,
            "include_metadata": include_metadata,
        }
        if filter:
            data["filter"] = filter
        if distance_metric:
            self._preflight("distance_metric", distance_metric)
            data["distance_metric"] = wire_value(distance_metric)
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
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
    ) -> dict[str, Any]:
        """
        Delete vectors from a namespace, by ID or by metadata filter.

        ``ids`` use ``POST /v1/namespaces/{ns}/vectors/delete`` (answer:
        ``{"deleted_count"}``); a ``filter`` uses ``POST .../vectors/bulk-delete``
        (answer: ``{"deleted", "failed", "errors"}``). The server has no
        delete-everything route.

        Raises:
            ValueError: ``delete_all`` was requested, or neither ``ids`` nor
                ``filter`` was given.

        Example:
            >>> client.delete("my-namespace", ids=["vec1", "vec2"])
            >>> client.delete("my-namespace", filter={"label": "obsolete"})
        """
        if delete_all:
            raise ValueError(
                "delete_all is not supported by the Dakera server; "
                "delete by ids or by filter, or delete the namespace"
            )
        if ids:
            return self._request(
                "POST", f"/v1/namespaces/{namespace}/vectors/delete", data={"ids": ids}
            )
        if filter:
            return self._request(
                "POST", f"/v1/namespaces/{namespace}/vectors/bulk-delete", data={"filter": filter}
            )
        raise ValueError("delete() needs ids or a filter")

    def bulk_update_vectors(
        self,
        namespace: str,
        filter: FilterDict,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Bulk update vector metadata matching a filter.

        Args:
            namespace: Target namespace
            filter: Filter to select vectors
            update: Metadata fields to set on matched vectors

        Returns:
            Dict with ``updated``, ``failed``, and ``errors`` fields
        """
        return self._request(
            "POST",
            f"/v1/namespaces/{namespace}/vectors/bulk-update",
            data={"filter": filter, "update": update},
        )

    def bulk_delete_vectors(
        self,
        namespace: str,
        filter: FilterDict,
    ) -> dict[str, Any]:
        """Bulk delete vectors matching a filter.

        Args:
            namespace: Target namespace
            filter: Filter to select vectors for deletion

        Returns:
            Dict with ``deleted``, ``failed``, and ``errors`` fields
        """
        return self._request(
            "POST",
            f"/v1/namespaces/{namespace}/vectors/bulk-delete",
            data={"filter": filter},
        )

    def count_vectors(
        self,
        namespace: str,
        filter: FilterDict | None = None,
    ) -> dict[str, Any]:
        """Count vectors in a namespace, optionally filtered.

        Args:
            namespace: Target namespace
            filter: Optional filter to narrow the count

        Returns:
            Dict with ``count`` and ``namespace`` fields
        """
        data: dict[str, Any] = {}
        if filter is not None:
            data["filter"] = filter
        return self._request(
            "POST",
            f"/v1/namespaces/{namespace}/vectors/count",
            data=data,
        )

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
        model: EmbeddingModel | str | None = None,
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
            self._preflight("model", model)
            data["model"] = wire_value(model)

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
        filter: FilterDict | None = None,
        include_text: bool = True,
        include_vectors: bool = False,
        model: EmbeddingModel | str | None = None,
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
            self._preflight("model", model)
            data["model"] = wire_value(model)

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
        filter: FilterDict | None = None,
        include_vectors: bool = False,
        model: EmbeddingModel | str | None = None,
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
            self._preflight("model", model)
            data["model"] = wire_value(model)

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
            ...     Document(id="doc2", text="Goodbye world"),
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
        filter: FilterDict | None = None,
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
        vector: list[float] | None = None,
        top_k: int = 10,
        vector_weight: float = 0.5,
        filter: FilterDict | None = None,
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search combining vector and full-text.

        When ``vector`` is omitted the server falls back to BM25-only full-text
        search. When provided, results are blended with vector similarity
        according to ``vector_weight``.

        Args:
            namespace: Target namespace
            query: Text query string
            vector: Optional query vector. Omit for BM25-only search.
            top_k: Number of results to return (default: 10)
            vector_weight: Balance between vector (0) and text (1) search (default: 0.5)
            filter: Optional metadata filter

        Returns:
            List of HybridSearchResult objects

        Example:
            >>> # Hybrid (vector + text)
            >>> results = client.hybrid_search(
            ...     "my-namespace",
            ...     query="hello world",
            ...     vector=[0.1, 0.2, 0.3],
            ...     vector_weight=0.7,
            ... )
            >>> # BM25-only (no vector)
            >>> results = client.hybrid_search("my-namespace", query="hello world")
        """
        data: dict[str, Any] = {
            "text": query,
            "top_k": top_k,
            "vector_weight": vector_weight,
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
        namespaces = response.get("namespaces", [])
        result = []
        for ns in namespaces:
            if isinstance(ns, str):
                result.append(NamespaceInfo(name=ns, vector_count=0))
            else:
                result.append(NamespaceInfo.from_dict(ns))
        return result

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
        dimensions: int | None = None,
        index_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        distance: "DistanceMetric | str | None" = None,
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
            data["dimension"] = dimensions
        if index_type:
            self._preflight("index_kind", index_type)
            data["index_type"] = index_type
        if metadata:
            data["metadata"] = metadata
        if distance:
            self._preflight("distance_metric", distance)
            data["distance"] = wire_value(distance)

        response = self._request("POST", "/v1/namespaces", data=data)
        return NamespaceInfo.from_dict(response)

    def configure_namespace(
        self,
        namespace: str,
        dimension: int,
        distance: DistanceMetric | None = None,
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
        if distance is not None:
            self._preflight("distance_metric", distance)
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

    def health_ready(self) -> dict[str, Any]:
        """K8s readiness probe — checks storage and dependencies."""
        return self._request("GET", "/health/ready")

    def health_live(self) -> dict[str, Any]:
        """K8s liveness probe — checks process is alive."""
        return self._request("GET", "/health/live")

    # =========================================================================
    # Health: readiness wait (server v0.12 /health/ready + /health/live)
    # =========================================================================

    def is_ready(self) -> bool:
        """Whether the server answers ``GET /health/ready`` with ``200``.

        One probe, no retries. A ``503`` (starting, storage unreachable,
        embedding warm-up failed) or an unreachable server is **not ready**: a
        starting v0.12 server answers ``503`` + ``Retry-After`` on its health
        routes while models load, and that must never count as healthy.
        """
        ready, _ = self._probe_ready()
        return ready

    def _probe_ready(self) -> tuple[bool, float | None]:
        """One ``/health/ready`` probe → ``(ready, retry_after_seconds)``."""
        try:
            response = self._session.request(
                "GET", self._url("/health/ready"), timeout=(self.connect_timeout, self.timeout)
            )
        except requests.exceptions.RequestException:
            return False, None
        if response.status_code == 200:
            return True, None
        return False, parse_retry_after(response.headers.get("Retry-After"))

    def wait_until_ready(self, timeout: float = 60.0, poll_interval: float = 1.0) -> dict[str, Any]:
        """Block until ``GET /health/ready`` answers ``200`` and return its body.

        Polls every ``poll_interval`` seconds, or as long as the server's
        ``Retry-After`` asks when it sent one. Use this after starting a server
        instead of treating any HTTP answer as healthy.

        Raises:
            TimeoutError: the server was not ready within ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            ready, retry_after = self._probe_ready()
            if ready:
                return self.health_ready()
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"server not ready after {timeout:g}s ({self.base_url})")
            delay = retry_after if retry_after is not None else poll_interval
            time.sleep(max(0.0, min(max(delay, 0.05), deadline - now)))

    # =========================================================================
    # Attachments (server v0.12, DAKERA_ATTACHMENTS — else 501 FEATURE_DISABLED)
    # =========================================================================

    def upload_attachment(
        self,
        namespace: str,
        data: "bytes | str | os.PathLike[str] | BinaryIO",
        content_type: str | None = None,
    ) -> AttachmentUploadResponse:
        """Upload a file to a namespace (content-addressed).

        Args:
            namespace: Namespace to store it in. To attach it to a memory use the
                agent's own namespace, ``_dakera_agent_{agent_id}``; the
                transcribe / index jobs accept an attachment from any namespace
                the key can read.
            data: The bytes, a path to a file, or a binary file object.
            content_type: Media type. Defaults to a guess from the file name, else
                ``application/octet-stream``.

        Returns:
            :class:`AttachmentUploadResponse` — ``attachment_ref`` is
            ``sha256:<hex>``; ``created`` is ``False`` when the namespace already
            held the same bytes.

        Raises:
            PayloadTooLargeError: over ``DAKERA_ATTACHMENT_MAX_BYTES`` (25 MiB by
                default) or the namespace quota.
            FeatureNotAvailableError: the server has attachments off (501).
        """
        body, guessed = _attachment_body(data)
        headers = {"Content-Type": content_type or guessed}
        result = self._request(
            "POST", f"/v1/namespaces/{namespace}/attachments", content=body, headers=headers
        )
        return AttachmentUploadResponse.from_dict(result)

    def list_attachments(self, namespace: str) -> list[AttachmentInfo]:
        """List a namespace's attachments (reference, media type, size; no bytes)."""
        result = self._request("GET", f"/v1/namespaces/{namespace}/attachments")
        items = result.get("attachments", []) if isinstance(result, dict) else []
        return [AttachmentInfo.from_dict(a) for a in items]

    def download_attachment(self, namespace: str, attachment_ref: str) -> AttachmentContent:
        """Download an attachment's bytes with the media type it was uploaded with."""
        response = self._request(
            "GET", f"/v1/namespaces/{namespace}/attachments/{attachment_ref}", raw=True
        )
        etag = response.headers.get("ETag")
        return AttachmentContent(
            data=response.content,
            content_type=response.headers.get("Content-Type", "application/octet-stream"),
            etag=etag.strip('"') if etag else None,
        )

    def delete_attachment(self, namespace: str, attachment_ref: str) -> None:
        """Delete an attachment.

        Raises:
            ConflictError: a memory still references it (409) — forget the memory
                instead; the attachment goes with the last memory that references it.
        """
        self._request("DELETE", f"/v1/namespaces/{namespace}/attachments/{attachment_ref}")

    def transcribe_attachment(
        self,
        namespace: str,
        attachment_ref: str,
        agent_id: str,
        *,
        memory_type: str | None = None,
        session_id: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
        expires_at: int | None = None,
        id: str | None = None,
        lang: str | None = None,
    ) -> JobAccepted:
        """Start a speech-to-text job on a WAV attachment (``202``).

        The job transcribes the audio and stores the transcript as a memory of
        ``agent_id`` (embedded and full-text indexed, ``attachment_ref`` set to the
        audio). Poll :meth:`get_transcription_job` or call
        :meth:`wait_for_transcription`. Needs read on ``namespace`` and write on
        the agent's namespace. Anything but WAV is a ``400`` before a job exists.
        """
        body = _memory_job_body(
            agent_id, None, memory_type, session_id, importance, tags, metadata,
            ttl_seconds, expires_at, id, lang,
        )  # fmt: skip
        result = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/attachments/{attachment_ref}/transcribe",
            data=body,
        )
        return JobAccepted.from_dict(result)

    def get_transcription_job(
        self, namespace: str, attachment_ref: str, job_id: str
    ) -> AttachmentJob:
        """Status of a transcription job. Jobs live in server memory: after a server
        restart this is a ``NotFoundError`` (``code == JOB_NOT_FOUND``); the memory a
        completed job stored is kept — look it up by ``JobAccepted.memory_id``."""
        result = self._request(
            "GET",
            f"/v1/namespaces/{namespace}/attachments/{attachment_ref}/transcribe/{job_id}",
        )
        return AttachmentJob.from_dict(result)

    def wait_for_transcription(
        self,
        namespace: str,
        attachment_ref: str,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> AttachmentJob:
        """Poll a transcription job until it completes, fails or is cancelled.

        Returns the final :class:`AttachmentJob` (check ``succeeded`` /
        ``error_code``). Raises :class:`~dakera.exceptions.TimeoutError` after
        ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_transcription_job(namespace, attachment_ref, job_id)
            if job.is_done:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"transcription job {job_id} not done after {timeout:g}s")
            time.sleep(poll_interval)

    def index_attachment(
        self,
        namespace: str,
        attachment_ref: str,
        agent_id: str,
        *,
        content: str | None = None,
        memory_type: str | None = None,
        session_id: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
        expires_at: int | None = None,
        id: str | None = None,
        lang: str | None = None,
    ) -> JobAccepted:
        """Start an image-indexing job on a PNG attachment (``202``).

        Needs ``DAKERA_VISION`` **and** ``DAKERA_ATTACHMENTS`` on the server (else
        501). The image is embedded from its pixels; ``content`` is only the
        caption stored as the memory's text (default ``[image sha256:...]``).
        """
        body = _memory_job_body(
            agent_id, content, memory_type, session_id, importance, tags, metadata,
            ttl_seconds, expires_at, id, lang,
        )  # fmt: skip
        result = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/attachments/{attachment_ref}/index",
            data=body,
        )
        return JobAccepted.from_dict(result)

    def get_index_job(self, namespace: str, attachment_ref: str, job_id: str) -> AttachmentJob:
        """Status of an image-index job (see :meth:`get_transcription_job`)."""
        result = self._request(
            "GET",
            f"/v1/namespaces/{namespace}/attachments/{attachment_ref}/index/{job_id}",
        )
        return AttachmentJob.from_dict(result)

    def wait_for_index(
        self,
        namespace: str,
        attachment_ref: str,
        job_id: str,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ) -> AttachmentJob:
        """Poll an image-index job until it completes, fails or is cancelled."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_index_job(namespace, attachment_ref, job_id)
            if job.is_done:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"index job {job_id} not done after {timeout:g}s")
            time.sleep(poll_interval)

    # =========================================================================
    # Records: one vector plus named representations (server v0.12, DAKERA_RECORDS)
    # =========================================================================

    def upsert_records(
        self, namespace: str, records: "list[Record | dict[str, Any]]"
    ) -> RecordUpsertResponse:
        """Upsert records: a primary dense vector (indexed, searched) plus named
        extra representations (``token_multivector`` / ``patch_multivector`` /
        ``dense``) stored beside it and deleted with it through the vector routes.

        Raises:
            FeatureNotAvailableError: ``DAKERA_RECORDS`` is off (501).
            PayloadTooLargeError: a record is over ``DAKERA_RECORD_MAX_VECTORS`` /
                ``DAKERA_RECORD_MAX_BYTES`` (413).

        Example:
            >>> client.upsert_records("docs", [Record(
            ...     id="r1", values=[0.1, 0.2, 0.3, 0.4],
            ...     representations=[Representation(
            ...         "tokens", [[0.1, 0.2], [0.3, 0.4]],
            ...         kind=RepresentationKind.TOKEN_MULTIVECTOR,
            ...         store_as=BlockDType.F16)])])
        """
        payload = [r.to_dict() if isinstance(r, Record) else r for r in records]
        result = self._request(
            "POST", f"/v1/namespaces/{namespace}/records", data={"records": payload}
        )
        return RecordUpsertResponse.from_dict(result)

    def get_record(
        self, namespace: str, record_id: str, include_vectors: bool = False
    ) -> RecordView:
        """Read a record: a manifest of its representations (name, kind, model,
        shape, dtype, bytes); the vectors themselves only with ``include_vectors``.
        There is no record delete route — delete the id through the vector routes."""
        params = {"include_vectors": "true"} if include_vectors else None
        result = self._request(
            "GET", f"/v1/namespaces/{namespace}/records/{record_id}", params=params
        )
        return RecordView.from_dict(result)

    def replace_namespace_ner_config(
        self,
        namespace: str,
        extract_entities: bool,
        entity_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Replace a namespace's entity-extraction config (``PUT .../config``,
        server v0.12+).

        Unlike :meth:`configure_namespace_ner` (``PATCH``, which merges), ``PUT`` is
        a full replacement: an omitted ``entity_types`` clears the list. A v0.11
        server answers ``405``.
        """
        body = {"extract_entities": extract_entities, "entity_types": entity_types or []}
        return self._request("PUT", f"/v1/namespaces/{namespace}/config", data=body)

    def get_index_stats(self, namespace: str) -> IndexStats:
        """
        Get index statistics for a namespace (from ``GET /v1/namespaces/{ns}``).

        Args:
            namespace: Namespace name

        Returns:
            IndexStats object (``total_vectors`` = the namespace's ``vector_count``,
            ``index_type`` = the index actually serving it, ``disk_usage_bytes`` =
            the server's storage estimate). For every namespace at once use
            :meth:`index_stats` (Admin scope).
        """
        response = self._request("GET", f"/v1/namespaces/{namespace}")
        return IndexStats.from_dict(response)

    def compact(self, namespace: str | None = None, force: bool = False) -> dict[str, Any]:
        """
        Trigger storage compaction (``POST /ops/compact``, Admin scope).

        Args:
            namespace: Compact only this namespace (``None`` = all).
            force: Compact every segment holding garbage, ignoring the backend's
                threshold.

        Returns:
            ``{"job_id", "message", "report"}``. A backend without on-request
            compaction answers ``501``
            (:class:`~dakera.exceptions.FeatureNotAvailableError`).
        """
        data: dict[str, Any] = {"force": force}
        if namespace is not None:
            data["namespace"] = namespace
        return self._request("POST", "/ops/compact", data=data)

    # =========================================================================
    # Memory Operations
    # =========================================================================

    def store_memory(
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
        valid_from: int | None = None,
        lang: str | None = None,
        attachment_ref: str | None = None,
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
            ttl_seconds: Optional TTL in seconds. The memory is hard-deleted after
                this many seconds from creation.
            expires_at: Optional explicit expiry as a Unix timestamp (seconds).
                Takes precedence over ``ttl_seconds`` when both are provided.
            valid_from: Optional bi-temporal validity start as a Unix timestamp
                (seconds). Indicates when this memory becomes temporally valid,
                independent of ingest time. Defaults to ingest time when omitted.
                Requires server v0.11.98+ (DAK-7424).
            lang: Language of ``content`` for the write-time derivations (event
                dates, entity tags): an ISO 639-1 code or name, optionally with a
                region (``"de"``, ``"pt-BR"``). Omitted ⇒ the server-wide default.
                Server v0.12+; an unsupported value is a ``400`` naming the
                supported languages (see ``capabilities().query_languages``).
            attachment_ref: ``sha256:<hex>`` of an attachment already uploaded to
                this agent's namespace (``_dakera_agent_{agent_id}``) with
                :meth:`upload_attachment`. Server v0.12+, needs
                ``DAKERA_ATTACHMENTS``.
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
        if valid_from is not None:
            data["valid_from"] = valid_from
        if lang is not None:
            data["lang"] = lang
        if attachment_ref is not None:
            data["attachment_ref"] = attachment_ref
        data["agent_id"] = agent_id
        response = self._request("POST", "/v1/memory/store", data=data)
        # Server wraps the memory in {"memory": {...}, "embedding_time_ms": ...}
        if isinstance(response, dict) and "memory" in response:
            return response["memory"]
        return response

    def recall(
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
        routing: "RoutingMode | str | None" = None,
        rerank: bool | None = None,
        fusion: "FusionStrategy | str | None" = None,
        vector_weight: float | None = None,
        iterations: int | None = None,
        neighborhood: bool | None = None,
        lang: str | None = None,
        tags: list[str] | None = None,
    ) -> RecallResponse:
        """Recall memories for an agent.

        Args:
            agent_id: The agent whose memories to recall.
            query: Semantic query text.
            top_k: Number of primary results to return (default: 5).
            memory_type: Filter by memory type.
            min_importance: Minimum importance threshold.
            tags: Only memories carrying at least one of these tags.
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
        if lang is not None:
            data["lang"] = lang
        if tags is not None:
            data["tags"] = tags
        data["agent_id"] = agent_id
        result = self._request("POST", "/v1/memory/recall", data=data)
        if isinstance(result, dict):
            return RecallResponse.from_dict(result)
        return RecallResponse(memories=result)

    def get_memory(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Get a specific memory."""
        return self._request("GET", f"/v1/memory/get/{memory_id}", params={"agent_id": agent_id})

    def update_memory(
        self,
        agent_id: str,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_type: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing memory.

        ``lang`` (server v0.12+) is the language of the content; when it differs
        from the recorded one the text-derived data is re-derived.
        """
        data: dict[str, Any] = {}
        if content is not None:
            data["content"] = content
        if metadata is not None:
            data["metadata"] = metadata
        if memory_type is not None:
            data["memory_type"] = memory_type
        if lang is not None:
            data["lang"] = lang
        return self._request(
            "PUT",
            f"/v1/memory/update/{memory_id}",
            data=data,
            params={"agent_id": agent_id},
        )

    def forget(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Delete a memory."""
        data = {"agent_id": agent_id, "memory_ids": [memory_id]}
        return self._request("POST", "/v1/memory/forget", data=data)

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
        result = self._request("DELETE", "/v1/memories/forget/batch", data=request.to_dict())
        return BatchForgetResponse.from_dict(result)

    def store_memories_batch(self, request: BatchStoreMemoryRequest) -> BatchStoreMemoryResponse:
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
            >>> resp = client.store_memories_batch(BatchStoreMemoryRequest("agent-1", items))
            >>> print(f"Stored {resp.stored_count} memories")
        """
        result = self._request("POST", "/v1/memories/store/batch", data=request.to_dict())
        return BatchStoreMemoryResponse.from_dict(result)

    def search_memories(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
        min_importance: float | None = None,
        routing: "RoutingMode | str | None" = None,
        rerank: bool | None = None,
        lang: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories for an agent.

        ``lang`` (server v0.12+): language of ``query`` for rule-based routing.
        ``tags``: only memories carrying at least one of these tags.
        """
        data: dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        if routing is not None:
            data["routing"] = routing.value if hasattr(routing, "value") else routing
        if rerank is not None:
            data["rerank"] = rerank
        if lang is not None:
            data["lang"] = lang
        if tags is not None:
            data["tags"] = tags
        data["agent_id"] = agent_id
        result = self._request("POST", "/v1/memory/search", data=data)
        items = result.get("memories", result) if isinstance(result, dict) else result
        if isinstance(items, list):
            return [
                {**item["memory"], "score": item.get("score")}
                if isinstance(item, dict) and isinstance(item.get("memory"), dict)
                else item
                for item in items
            ]
        return items

    def compress_agent(self, agent_id: str) -> "CompressResponse":
        """Compress the memory namespace for an agent (CE-12).

        Runs a server-side compression pass that removes low-value or redundant
        memories, returning statistics about the operation.

        Args:
            agent_id: The agent whose namespace to compress.

        Returns:
            :class:`CompressResponse` with before/after counts and timing.
        """
        result = self._request("POST", f"/v1/agents/{agent_id}/compress")
        return CompressResponse.from_dict(result)

    def update_importance(
        self,
        agent_id: str,
        memory_ids: list[str],
        importance: float,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Update importance of memories."""
        results = []
        for mid in memory_ids:
            data = {"agent_id": agent_id, "memory_id": mid, "importance": importance}
            results.append(self._request("POST", "/v1/memory/importance", data=data))
        return results[0] if len(results) == 1 else results

    def consolidate(
        self,
        agent_id: str,
        memory_type: str | None = None,
        threshold: float | None = None,
        dry_run: bool = False,
        config: "ConsolidationConfig | None" = None,
    ) -> dict[str, Any]:
        """Consolidate memories for an agent (CE-6).

        Args:
            agent_id: Agent whose memories to consolidate.
            memory_type: Optional filter — only consolidate this memory type.
            threshold: Similarity threshold for grouping (0–1).
            dry_run: Preview changes without applying them.
            config: Optional :class:`~dakera.ConsolidationConfig` to select the
                clustering algorithm (``"dbscan"`` or ``"greedy"``) and tune its
                parameters.  When omitted the server uses its default algorithm.

        Returns:
            Dict with ``consolidated_count``, ``removed_count``, ``new_memories``
            and optionally a ``log`` list of :class:`~dakera.ConsolidationLogEntry`
            steps.
        """
        data: dict[str, Any] = {"dry_run": dry_run}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if threshold is not None:
            data["threshold"] = threshold
        if config is not None:
            data["config"] = config.to_dict()
        data["agent_id"] = agent_id
        return self._request("POST", "/v1/memory/consolidate", data=data)

    def consolidate_agent(self, agent_id: str) -> dict[str, Any]:
        """Consolidate memories directly for an agent (DBSCAN clustering).

        Unlike :meth:`consolidate` which uses the generic memory endpoint, this
        calls the agent-scoped consolidation endpoint that runs the full
        DBSCAN-based clustering pipeline for the agent.

        Returns:
            Dict with ``agent_id``, ``memories_scanned``, ``clusters_found``,
            ``memories_deprecated``, ``anchor_ids``, ``deprecated_ids``.
        """
        return self._request("POST", f"/v1/agents/{agent_id}/consolidate")

    def get_consolidation_log(self, agent_id: str) -> list[dict[str, Any]]:
        """Get the consolidation execution log for an agent.

        Returns:
            List of log entries with ``timestamp``, ``clusters_found``,
            ``memories_deprecated``, ``anchor_ids``, ``deprecated_ids``.
        """
        return self._request("GET", f"/v1/agents/{agent_id}/consolidation/log")

    def patch_consolidation_config(
        self,
        agent_id: str,
        enabled: bool | None = None,
        epsilon: float | None = None,
        min_samples: int | None = None,
        soft_deprecation_days: int | None = None,
    ) -> dict[str, Any]:
        """Update the consolidation configuration for an agent.

        Returns:
            Updated consolidation config dict.
        """
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        if epsilon is not None:
            data["epsilon"] = epsilon
        if min_samples is not None:
            data["min_samples"] = min_samples
        if soft_deprecation_days is not None:
            data["soft_deprecation_days"] = soft_deprecation_days
        return self._request("PATCH", f"/v1/agents/{agent_id}/consolidation/config", data=data)

    def memory_feedback(
        self,
        agent_id: str,
        memory_id: str,
        feedback: str,
        relevance_score: float | None = None,
    ) -> dict[str, Any]:
        """Submit feedback on a memory (``POST /v1/memory/feedback``).

        Args:
            feedback: The signal: ``"upvote"``, ``"downvote"``, ``"flag"`` (or the
                aliases ``"positive"`` / ``"negative"``).
            relevance_score: Ignored — the server has no such field. Kept so existing
                calls keep working; use :meth:`feedback_memory` for the path-based
                INT-1 route.

        Returns:
            ``{"memory_id", "new_importance", "signal"}``.
        """
        data: dict[str, Any] = {"agent_id": agent_id, "memory_id": memory_id, "signal": feedback}
        return self._request("POST", "/v1/memory/feedback", data=data)

    # =========================================================================
    # Memory Feedback Loop — INT-1
    # =========================================================================

    def feedback_memory(
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
        result = self._request("POST", f"/v1/memories/{memory_id}/feedback", data=data)
        return FeedbackResponse.from_dict(result)

    def get_memory_feedback_history(self, memory_id: str) -> FeedbackHistoryResponse:
        """Get the full feedback history for a memory (INT-1).

        Args:
            memory_id: The memory whose feedback history to retrieve.

        Returns:
            :class:`FeedbackHistoryResponse` with ordered list of feedback events.
        """
        result = self._request("GET", f"/v1/memories/{memory_id}/feedback")
        return FeedbackHistoryResponse.from_dict(result)

    def evaluate_tif(self, memory_id: str) -> TifScore:
        """Compute a T-I-F reliability score for a memory (T-I-F RFC Phase 3).

        Fetches the memory's full feedback history and reduces it to a
        :class:`TifScore` with truth/indeterminacy/falsity proportions and a
        human-readable :attr:`~TifScore.classification`.

        Args:
            memory_id: The memory to score.

        Returns:
            :class:`TifScore` derived from the memory's feedback history.
        """
        history = self.get_memory_feedback_history(memory_id)
        return TifScore.from_feedback_history(history)

    def get_agent_feedback_summary(self, agent_id: str) -> AgentFeedbackSummary:
        """Get aggregate feedback counts and health score for an agent (INT-1).

        Args:
            agent_id: The agent to summarise feedback for.

        Returns:
            :class:`AgentFeedbackSummary` with upvote/downvote/flag counts and health score.
        """
        result = self._request("GET", f"/v1/agents/{agent_id}/feedback/summary")
        return AgentFeedbackSummary.from_dict(result)

    def patch_memory_importance(
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
        result = self._request("PATCH", f"/v1/memories/{memory_id}/importance", data=data)
        return FeedbackResponse.from_dict(result)

    def get_feedback_health(self, agent_id: str) -> FeedbackHealthResponse:
        """Get overall feedback health score for an agent (INT-1).

        The health score is the mean importance of all non-expired memories (0.0–1.0).
        A higher score indicates a healthier, more relevant memory store.

        Args:
            agent_id: The agent to get health score for.

        Returns:
            :class:`FeedbackHealthResponse` with health score, memory count, and avg importance.
        """
        result = self._request("GET", "/v1/feedback/health", params={"agent_id": agent_id})
        return FeedbackHealthResponse.from_dict(result)

    # =========================================================================
    # Memory Knowledge Graph Operations (CE-5 / SDK-9)
    # =========================================================================

    def memory_graph(
        self,
        memory_id: str,
        depth: int = 1,
        types: list[str] | None = None,
    ) -> MemoryGraph:
        """Traverse the knowledge graph from a memory node.

        Requires CE-5 (Memory Knowledge Graph) on the server.

        Args:
            memory_id: Root memory ID to start traversal from.
            depth: Maximum traversal depth (default: 1; the server caps it at 5).
            types: Filter by edge types — any of ``"related_to"``,
                ``"shares_entity"``, ``"precedes"``, ``"linked_by"``.
                ``None`` returns all edge types. Applied client-side: the
                server returns every type.
        """
        # The server takes only `depth` (capped at 5) and returns every edge
        # type, so `types` is applied here.
        params: dict[str, Any] = {"depth": depth}
        result = self._request("GET", f"/v1/memories/{memory_id}/graph", params=params)
        graph = MemoryGraph.from_dict(result)
        return graph.only_edge_types(types) if types else graph

    def memory_path(
        self,
        source_id: str,
        target_id: str,
    ) -> GraphPath:
        """Find the shortest path between two memories in the knowledge graph.

        Requires CE-5 (Memory Knowledge Graph) on the server.
        """
        params: dict[str, Any] = {"to": target_id}
        result = self._request("GET", f"/v1/memories/{source_id}/path", params=params)
        return GraphPath.from_dict(result)

    def memory_link(
        self,
        source_id: str,
        target_id: str,
        edge_type: Union[str, EdgeType] = EdgeType.LINKED_BY,
        *,
        agent_id: str,
        label: str | None = None,
    ) -> GraphLinkResponse:
        """Create an explicit edge between two memories.

        Requires CE-5 (Memory Knowledge Graph) on the server.

        Args:
            source_id: Source memory ID.
            target_id: Target memory ID.
            edge_type: Accepted for compatibility; the server records every
                explicit link as ``linked_by`` and this value is not sent.
            agent_id: Agent that owns both memories (required by the server).
            label: Optional human-readable label stored with the link.
        """
        # The server reads {target_id, agent_id, label?} and records every
        # explicit link as `linked_by`; `edge_type` is accepted for
        # compatibility and not sent.
        if not agent_id:
            raise ValueError("memory_link() needs agent_id: the server requires it")
        del edge_type  # always recorded as linked_by by the server
        data: dict[str, Any] = {"target_id": target_id, "agent_id": agent_id}
        if label is not None:
            data["label"] = label
        result = self._request("POST", f"/v1/memories/{source_id}/links", data=data)
        if isinstance(result, dict) and "error" in result:
            raise AuthorizationError(
                message=result.get("message") or result.get("error", "Forbidden"),
                status_code=403,
                response_body=result,
                code=ErrorCode.UNKNOWN,
            )
        return GraphLinkResponse.from_dict(result)

    def agent_graph_export(
        self,
        agent_id: str,
        format: str = "json",
    ) -> GraphExport:
        """Export the full knowledge graph for an agent.

        Requires CE-5 (Memory Knowledge Graph) on the server.

        Args:
            agent_id: Agent whose graph to export.
            format: Sent as given; the server always answers JSON
                (``{agent_id, namespace, node_count, edge_count, edges}``).
                For GraphML use :meth:`knowledge_export` with ``format="graphml"``.
        """
        params: dict[str, Any] = {"format": format}
        result = self._request("GET", f"/v1/agents/{agent_id}/graph/export", params=params)
        return GraphExport.from_dict(result)

    # =========================================================================
    # Entity Extraction Operations (CE-4)
    # =========================================================================

    def get_namespace_entity_config(self, namespace: str) -> dict[str, Any]:
        """Get entity extraction configuration for a namespace.

        Returns:
            Dict with ``namespace``, ``extract_entities``, ``entity_types``.
        """
        return self._request("GET", f"/v1/namespaces/{namespace}/config")

    def get_namespace_extractor(self, namespace: str) -> dict[str, Any]:
        """Get the extractor provider configuration for a namespace.

        Returns:
            Dict with ``provider``, ``model``, ``base_url``.
        """
        return self._request("GET", f"/v1/namespaces/{namespace}/extractor")

    def configure_namespace_ner(
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
                "location", "date"]``.  ``None`` keeps existing types; ``[]`` clears
                them (a v0.12 server's ``PATCH`` merges what it is sent; use
                :meth:`replace_namespace_ner_config` for a full replacement).

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
        return self._request("PATCH", f"/v1/namespaces/{namespace}/config", data=config.to_dict())

    def extract_entities(
        self,
        text: str,
        entity_types: list[str] | None = None,
        lang: str | None = None,
    ) -> EntityExtractionResponse:
        """Extract entities from arbitrary text without storing a memory.

        Uses the same GLiNER + rule-based pipeline as automatic extraction.
        Requires ``Scope::Read``.

        Args:
            text: Text to extract entities from.
            entity_types: Entity types to extract.  ``None`` uses server
                defaults (person, org, location, date, url, email).
            lang: Language of ``text`` for the rule-based date rules (server v0.12+).

        Returns:
            :class:`EntityExtractionResponse` with extracted entities.

        Note:
            Requires CE-4 (GLiNER) on the server.
        """
        data: dict[str, Any] = {"content": text}
        if entity_types is not None:
            data["entity_types"] = entity_types
        if lang is not None:
            data["lang"] = lang
        result = self._request("POST", "/v1/memories/extract", data=data)
        return EntityExtractionResponse.from_dict(result)

    def memory_entities(self, memory_id: str) -> MemoryEntitiesResponse:
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
        result = self._request("GET", f"/v1/memory/entities/{memory_id}")
        return MemoryEntitiesResponse.from_dict(result, memory_id=memory_id)

    # =========================================================================
    # Session Operations
    # =========================================================================

    def start_session(
        self,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new session. Returns the session dict (unwrapped from the server response)."""
        data: dict[str, Any] = {"agent_id": agent_id}
        if metadata is not None:
            data["metadata"] = metadata
        result = self._request("POST", "/v1/sessions/start", data=data)
        return result["session"]

    def end_session(self, session_id: str, summary: str | None = None) -> dict[str, Any]:
        """End a session."""
        data: dict[str, Any] = {}
        if summary is not None:
            data["summary"] = summary
        return self._request("POST", f"/v1/sessions/{session_id}/end", data=data)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get session details."""
        return self._request("GET", f"/v1/sessions/{session_id}")

    def list_sessions(
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
        return self._request("GET", "/v1/sessions", params=params)

    def session_memories(self, session_id: str) -> list[dict[str, Any]]:
        """Get memories for a session."""
        result = self._request("GET", f"/v1/sessions/{session_id}/memories")
        if isinstance(result, dict):
            return result.get("memories", [])
        return result

    # =========================================================================
    # Agent Operations
    # =========================================================================

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents."""
        return self._request("GET", "/v1/agents")

    def agent_memories(
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
        return self._request("GET", f"/v1/agents/{agent_id}/memories", params=params)

    def agent_stats(self, agent_id: str) -> dict[str, Any]:
        """Get stats for an agent."""
        return self._request("GET", f"/v1/agents/{agent_id}/stats")

    def agent_sessions(
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
        return self._request("GET", f"/v1/agents/{agent_id}/sessions", params=params)

    def wake_up(
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
        result = self._request("GET", f"/v1/agents/{agent_id}/wake-up", params=params)
        return WakeUpResponse.from_dict(result)

    # =========================================================================
    # Cache Warming Operations
    # =========================================================================

    def warm_cache(
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
        negative: list[list[float]] | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        mmr_lambda: float | None = None,
        mmr_prefetch_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Multi-vector search with positive/negative vectors and optional MMR
        (``POST /v1/namespaces/{ns}/multi-vector``).

        Args:
            namespace: Target namespace
            positive: List of positive query vectors
            negative: Optional list of negative query vectors
            top_k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Include metadata in results
            include_vectors: Include vector values in results
            mmr_lambda: Enables MMR re-ranking; 0.0 = max diversity, 1.0 = max relevance
            mmr_prefetch_k: Ignored — the server has no such field (kept for
                compatibility).

        Returns:
            Dict with results and search metadata
        """
        data: dict[str, Any] = {
            "positive_vectors": positive,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_vectors": include_vectors,
        }
        if negative is not None:
            data["negative_vectors"] = negative
        if filter:
            data["filter"] = filter
        if mmr_lambda is not None:
            data["enable_mmr"] = True
            data["mmr_lambda"] = mmr_lambda
        return self._request("POST", f"/v1/namespaces/{namespace}/multi-vector", data=data)

    def unified_query(
        self,
        namespace: str,
        vector: list[float] | None = None,
        text: str | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        vector_weight: float | None = None,
        text_weight: float | None = None,
        fusion_method: str | None = None,
        rerank: bool = False,
        rank_by: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Unified query (``POST /v1/namespaces/{ns}/unified-query``): rank by a
        ``rank_by`` expression.

        Pass ``rank_by`` yourself (``["ANN", [..]]``, ``["text", "BM25", "query"]``,
        ``["Sum", [..]]``, ``["Product", weight, expr]``, ``["field", "asc"]``) or
        give ``vector`` and / or ``text`` and one is built: both are combined with
        ``Sum``, each wrapped in ``Product`` when ``vector_weight`` /
        ``text_weight`` is set.

        ``fusion_method`` and ``rerank`` are ignored (the server has no such
        fields; kept for compatibility).

        Returns:
            Dict with ``results`` (each with ``$dist``) and ``next_cursor``.
        """
        if rank_by is None:
            parts: list[Any] = []
            if vector is not None:
                expr: list[Any] = ["ANN", vector]
                parts.append(
                    ["Product", vector_weight, expr] if vector_weight is not None else expr
                )
            if text is not None:
                expr = ["text", "BM25", text]
                if text_weight is not None:
                    expr = ["Product", text_weight, expr]
                parts.append(expr)
            if not parts:
                raise ValueError("unified_query() needs rank_by, vector or text")
            rank_by = parts[0] if len(parts) == 1 else ["Sum", parts]
        data: dict[str, Any] = {
            "rank_by": rank_by,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_vectors": include_vectors,
        }
        if filter:
            data["filter"] = filter
        return self._request("POST", f"/v1/namespaces/{namespace}/unified-query", data=data)

    def aggregate(
        self,
        namespace: str,
        vector: list[float] | None = None,
        group_by: "str | list[str] | None" = None,
        metrics: list[str] | None = None,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        top_groups: int | None = None,
        aggregate_by: dict[str, list[Any]] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate vectors' metadata (``POST /v1/namespaces/{ns}/aggregate``).

        Args:
            namespace: Target namespace.
            group_by: Attribute(s) to group by.
            aggregate_by: Named aggregates exactly as the server takes them:
                ``{"n": ["Count"], "avg_price": ["Avg", "price"]}`` (``Count``,
                ``Sum``, ``Avg``, ``Min``, ``Max``).
            metrics: Shorthand for ``aggregate_by``: ``"count"`` or
                ``"sum:field"`` / ``"avg:field"`` / ``"min:field"`` / ``"max:field"``.
            filter: Optional metadata filter.
            limit: Maximum groups (server default 100); ``top_groups`` is the older
                name for the same thing.
            vector, top_k: Ignored — the server's aggregation is a metadata scan
                (kept for compatibility).

        With neither ``aggregate_by`` nor ``metrics`` a ``count`` is computed.
        """
        agg: dict[str, list[Any]] = dict(aggregate_by or {})
        for metric in metrics or []:
            name, _, field = metric.partition(":")
            kind = name.lower()
            if kind == "count":
                agg.setdefault("count", ["Count"])
            elif kind in ("sum", "avg", "min", "max") and field:
                agg[f"{kind}_{field}"] = [kind.capitalize(), field]
            else:
                raise ValueError(
                    f"unsupported metric {metric!r}: use 'count' or 'sum|avg|min|max:<field>'"
                )
        data: dict[str, Any] = {"aggregate_by": agg or {"count": ["Count"]}}
        if group_by is not None:
            data["group_by"] = [group_by] if isinstance(group_by, str) else list(group_by)
        if filter:
            data["filter"] = filter
        if limit is None:
            limit = top_groups
        if limit is not None:
            data["limit"] = limit
        return self._request("POST", f"/v1/namespaces/{namespace}/aggregate", data=data)

    def export_vectors(
        self,
        namespace: str,
        cursor: str | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
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
        vector: list[float] | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        query_type: str = "vector_search",
        text_query: str | None = None,
        execute: bool = False,
    ) -> dict[str, Any]:
        """
        Explain query execution plan (``POST /v1/namespaces/{ns}/explain``).

        Args:
            namespace: Target namespace
            vector: Query vector (``vector_search`` / ``hybrid_search``)
            top_k: Number of results
            filter: Optional metadata filter
            include_metadata: Ignored — the server has no such field (kept for
                compatibility).
            query_type: ``vector_search`` (default), ``full_text_search``,
                ``hybrid_search``, ``multi_vector`` or ``batch_query``.
            text_query: Text query for ``full_text_search`` / ``hybrid_search``.
            execute: Also run the query and report measured ``actual_stats``.

        Returns:
            Dict with the query plan, execution steps and timing information
        """
        data: dict[str, Any] = {"query_type": query_type, "top_k": top_k}
        if vector is not None:
            data["vector"] = vector
        if text_query is not None:
            data["text_query"] = text_query
        if filter:
            data["filter"] = filter
        if execute:
            data["execute"] = True
        return self._request("POST", f"/v1/namespaces/{namespace}/explain", data=data)

    def upsert_columns(
        self,
        namespace: str,
        ids: list[str],
        vectors: list[list[float]],
        attributes: dict[str, list[Any]] | None = None,
        ttl_seconds: int | None = None,
        dimension: int | None = None,
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
        memory_id: str | None = None,
        depth: int | None = None,
        min_similarity: float | None = None,
    ) -> dict[str, Any]:
        """Build a knowledge graph from a seed memory."""
        if not memory_id:
            raise ValueError(
                "knowledge_graph() needs memory_id: the server builds the graph from a seed memory"
            )
        data: dict[str, Any] = {"agent_id": agent_id, "memory_id": memory_id}
        if depth is not None:
            data["depth"] = depth
        if min_similarity is not None:
            data["min_similarity"] = min_similarity
        return self._request("POST", "/v1/knowledge/graph", data=data)

    def full_knowledge_graph(
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
        return self._request("POST", "/v1/knowledge/graph/full", data=data)

    def summarize(
        self,
        agent_id: str,
        memory_ids: list[str] | None = None,
        target_type: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Summarize memories."""
        # The server needs at least two memory ids and has no dry run: it
        # always writes the summary memory.
        if dry_run:
            raise ValueError(
                "summarize() has no dry run on the server: it always stores the summary"
            )
        if not memory_ids or len(memory_ids) < 2:
            raise ValueError("summarize() needs at least two memory_ids")
        data: dict[str, Any] = {"agent_id": agent_id, "memory_ids": memory_ids}
        if target_type is not None:
            data["target_type"] = target_type
        return self._request("POST", "/v1/knowledge/summarize", data=data)

    def deduplicate(
        self,
        agent_id: str,
        threshold: float | None = None,
        memory_type: str | None = None,
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
    # KG-2: Graph Query & Export Operations
    # =========================================================================

    def knowledge_query(
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
        result = self._request("GET", "/v1/knowledge/query", params=params)
        return KgQueryResponse.from_dict(result)

    def knowledge_path(
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
        result = self._request("GET", "/v1/knowledge/path", params=params)
        return KgPathResponse.from_dict(result)

    def knowledge_export(
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
        result = self._request("GET", "/v1/knowledge/export", params=params)
        return KgExportResponse.from_dict(result)

    # =========================================================================
    # Analytics Operations
    # =========================================================================

    def analytics_overview(
        self,
        period: str | None = None,
        namespace: str | None = None,
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
        period: str | None = None,
        namespace: str | None = None,
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
        period: str | None = None,
        namespace: str | None = None,
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
        namespace: str | None = None,
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
        """Get server stats (version, total_vectors, namespace_count,
        uptime_seconds, timestamp, state).

        Requires Read scope — works with read-only API keys, unlike cluster_status.
        The ``state`` field is ``"healthy"`` when storage is accessible, ``"degraded"`` otherwise.
        """
        return self._request("GET", "/v1/ops/stats")

    def ops_metrics(self) -> str:
        """Get Prometheus metrics in text exposition format (INFRA-3).

        Requires Admin scope. Returns the raw Prometheus text exposition
        format string suitable for scraping by a Prometheus server.
        """
        return self._request("GET", "/v1/ops/metrics")

    def debug_config(self) -> dict[str, Any]:
        """Return all active DAKERA_* env vars (non-secret) from the running server (DAK-7477).

        Requires Admin scope. Returns a dict of DAKERA_* environment variables
        set in the server process, plus ``_version`` and optionally ``_build_sha``.
        Secret-bearing keys (TOKEN, KEY, SECRET, PASSWORD, CRED, URL, URI, DSN)
        are filtered server-side.

        Used by bench harnesses to verify the server is running with the exact
        feature-flag configuration requested before scoring.
        """
        return self._request("GET", "/debug/config")

    def cluster_status(self) -> dict[str, Any]:
        """Get cluster status."""
        return self._request("GET", "/v1/admin/cluster/status")

    def cluster_nodes(self) -> list[dict[str, Any]]:
        """Get cluster nodes."""
        return self._request("GET", "/v1/admin/cluster/nodes")

    def optimize_namespace(self, namespace: str) -> dict[str, Any]:
        """Optimize a namespace."""
        return self._request("POST", f"/v1/admin/namespaces/{namespace}/optimize")

    def index_stats(self) -> dict[str, Any]:
        """Get index statistics across all namespaces."""
        return self._request("GET", "/v1/admin/indexes/stats")

    def rebuild_indexes(self, namespace: str | None = None) -> dict[str, Any]:
        """Rebuild indexes, optionally for a specific namespace."""
        data: dict[str, Any] = {}
        if namespace is not None:
            data["namespace"] = namespace
        return self._request("POST", "/v1/admin/indexes/rebuild", data=data or None)

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._request("GET", "/v1/admin/cache/stats")

    def cache_clear(self, namespace: str | None = None) -> dict[str, Any]:
        """Clear cache, optionally for a specific namespace."""
        data: dict[str, Any] | None = None
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

    def update_quotas(self, quotas: dict[str, Any], namespace: str | None = None) -> dict[str, Any]:
        """Set a quota configuration.

        Uses ``PUT /v1/admin/quotas/{namespace}`` for one namespace, or
        ``PUT /v1/admin/quotas/default`` (the default applied to namespaces
        without their own quota) when ``namespace`` is omitted. The server has no
        ``PUT /v1/admin/quotas``.

        Args:
            quotas: The quota config: ``max_vectors``, ``max_storage_bytes``,
                ``max_dimensions``, ``max_metadata_bytes``, ``enforcement``
                (``"none"``, ``"soft"`` or ``"hard"``). A ``{"config": {...}}``
                wrapper is accepted as well.
            namespace: Target namespace; ``None`` sets the default quota.

        Note:
            v0.12 enforces quotas: a write over a ``hard`` quota is a ``413``
            (:class:`~dakera.exceptions.PayloadTooLargeError`, ``.is_quota``).
        """
        config = quotas["config"] if set(quotas) == {"config"} else quotas
        path = "/v1/admin/quotas/default" if namespace is None else f"/v1/admin/quotas/{namespace}"
        return self._request("PUT", path, data={"config": config})

    def slow_queries(
        self,
        limit: int | None = None,
        min_duration_ms: int | None = None,
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
        """Restore a backup (``POST /v1/admin/backups/restore``; needs global
        ``super_admin`` on v0.12). See also admin_restore_backup for full restore
        options."""
        return self._request("POST", "/v1/admin/backups/restore", data={"backup_id": backup_id})

    def delete_backup(self, backup_id: str) -> dict[str, Any]:
        """Delete a backup."""
        return self._request("DELETE", f"/v1/admin/backups/{backup_id}")

    def autopilot_status(self) -> dict[str, Any]:
        """Get AutoPilot status: current config and last-run statistics (PILOT-1)."""
        return self._request("GET", "/v1/admin/autopilot/status")

    def autopilot_update_config(
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
        return self._request("PUT", "/v1/admin/decay/config", data=data)

    def decay_stats(self) -> dict[str, Any]:
        """Get decay engine activity counters and last-cycle snapshot (DECAY-2).

        Returns cumulative totals (memories decayed/deleted, cycles run) and
        per-cycle statistics from the most recent run. Requires Admin scope.
        """
        return self._request("GET", "/v1/admin/decay/stats")

    def get_kpis(self) -> "KpiSnapshot":
        """Return a point-in-time product KPI snapshot (OBS-2).

        Calls ``GET /v1/kpis``. Returns 8 operational metrics covering
        latency, error rate, and retention. Sub-millisecond — served from
        in-memory counters. Requires Admin scope.

        Returns:
            :class:`~dakera.models.KpiSnapshot` with all 8 KPI fields.
        """
        result = self._request("GET", "/v1/kpis")
        return KpiSnapshot.from_dict(result)

    def rotate_encryption_key(
        self,
        new_key: str,
        namespace: str | None = None,
    ) -> RotateEncryptionKeyResponse:
        """Re-encrypt all memory content blobs with a new AES-256-GCM key (SEC-3).

        After this call the new key is active in the running process.
        The operator must update ``DAKERA_ENCRYPTION_KEY`` and restart the
        server to make the rotation durable across restarts.

        Only blobs stored with the ``$enc$v1$`` prefix are re-encrypted.
        Plaintext blobs (stored before encryption was enabled) are encrypted
        with the new key as part of the rotation.

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
        result = self._request("POST", "/v1/admin/encryption/rotate-key", data=data)
        return RotateEncryptionKeyResponse.from_dict(result)

    # =========================================================================
    # API Key Operations
    # =========================================================================

    def create_key(
        self,
        name: str,
        permissions: list[str] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Create a new API key."""
        data: dict[str, Any] = {"name": name}
        if permissions is not None:
            data["permissions"] = permissions
        if expires_at is not None:
            data["expires_at"] = expires_at
        return self._request("POST", "/admin/keys", data=data)

    def list_keys(self) -> list[dict[str, Any]]:
        """List all API keys."""
        return self._request("GET", "/admin/keys")

    def get_key(self, key_id: str) -> dict[str, Any]:
        """Get an API key by ID."""
        return self._request("GET", f"/admin/keys/{key_id}")

    def delete_key(self, key_id: str) -> dict[str, Any]:
        """Delete an API key."""
        return self._request("DELETE", f"/admin/keys/{key_id}")

    def deactivate_key(self, key_id: str) -> dict[str, Any]:
        """Deactivate an API key."""
        return self._request("POST", f"/admin/keys/{key_id}/deactivate")

    def rotate_key(self, key_id: str) -> dict[str, Any]:
        """Rotate an API key."""
        return self._request("POST", f"/admin/keys/{key_id}/rotate")

    def key_usage(self, key_id: str) -> dict[str, Any]:
        """Get usage statistics for an API key."""
        return self._request("GET", f"/admin/keys/{key_id}/usage")

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

    def stream_namespace_events(
        self,
        namespace: str,
        timeout: float | None = None,
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
        timeout: float | None = None,
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
        timeout: float | None,
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
        timeout: float | None = None,
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
    # Namespace API Keys — SEC-1
    # =========================================================================

    def create_namespace_key(
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
        result = self._request("POST", f"/v1/namespaces/{namespace}/keys", data=data)
        return CreateNamespaceKeyResponse.from_dict(result)

    def list_namespace_keys(self, namespace: str) -> ListNamespaceKeysResponse:
        """List all API keys scoped to a namespace (SEC-1).

        Args:
            namespace: The namespace whose keys to list.

        Returns:
            :class:`ListNamespaceKeysResponse` with key metadata (no secrets).
        """
        result = self._request("GET", f"/v1/namespaces/{namespace}/keys")
        return ListNamespaceKeysResponse.from_dict(result)

    def delete_namespace_key(self, namespace: str, key_id: str) -> dict[str, Any]:
        """Revoke a namespace-scoped API key (SEC-1).

        Args:
            namespace: The namespace the key belongs to.
            key_id: The key to revoke.

        Returns:
            Dict with ``success`` and ``message`` fields.
        """
        return self._request("DELETE", f"/v1/namespaces/{namespace}/keys/{key_id}")

    def get_namespace_key_usage(self, namespace: str, key_id: str) -> NamespaceKeyUsageResponse:
        """Get usage statistics for a namespace-scoped API key (SEC-1).

        Args:
            namespace: The namespace the key belongs to.
            key_id: The key whose usage to retrieve.

        Returns:
            :class:`NamespaceKeyUsageResponse` with request counts and latency.
        """
        result = self._request("GET", f"/v1/namespaces/{namespace}/keys/{key_id}/usage")
        return NamespaceKeyUsageResponse.from_dict(result)

    # =========================================================================
    # DX-1: Memory Import / Export
    # =========================================================================

    def import_memories(
        self,
        data: Any,
        format: str = "jsonl",
        agent_id: str | None = None,
        namespace: str | None = None,
    ) -> MemoryImportResponse:
        """Import memories from an external format (DX-1).

        Args:
            data: Serialised memories — a list of dicts (Mem0/JSONL), a
                Zep-compatible dict, or a CSV string depending on *format*.
            format: One of ``"jsonl"``, ``"mem0"``, ``"zep"``, ``"csv"``.
            agent_id: Assign all imported memories to this agent.
            namespace: Target namespace (defaults to the client's namespace).

        Returns:
            :class:`MemoryImportResponse` with counts and any per-row errors.
        """
        body: dict[str, Any] = {"data": data, "format": format}
        if agent_id is not None:
            body["agent_id"] = agent_id
        if namespace is not None:
            body["namespace"] = namespace
        result = self._request("POST", "/v1/import", data=body)
        return MemoryImportResponse.from_dict(result)

    def export_memories(
        self,
        format: str = "jsonl",
        agent_id: str | None = None,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> MemoryExportResponse:
        """Export memories in a portable format (DX-1).

        Args:
            format: One of ``"jsonl"``, ``"mem0"``, ``"zep"``, ``"csv"``.
            agent_id: Export only memories for this agent.
            namespace: Source namespace (defaults to client namespace).
            limit: Maximum number of memories to export.

        Returns:
            :class:`MemoryExportResponse` with the serialised data and count.
        """
        params: dict[str, Any] = {"format": format}
        if agent_id is not None:
            params["agent_id"] = agent_id
        if namespace is not None:
            params["namespace"] = namespace
        if limit is not None:
            params["limit"] = limit
        result = self._request("GET", "/v1/export", params=params)
        return MemoryExportResponse.from_dict(result)

    # =========================================================================
    # OBS-1: Business-Event Audit Log
    # =========================================================================

    def list_audit_events(
        self,
        agent_id: str | None = None,
        event_type: str | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> AuditListResponse:
        """List business-event audit log entries (OBS-1).

        Args:
            agent_id: Filter to events from this agent.
            event_type: Filter to a specific event type string.
            from_ts: Unix timestamp lower bound (inclusive).
            to_ts: Unix timestamp upper bound (exclusive).
            limit: Maximum number of events to return.
            cursor: Pagination cursor from a previous response.

        Returns:
            :class:`AuditListResponse` with events and optional next cursor.
        """
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
        result = self._request("GET", "/v1/audit", params=params)
        return AuditListResponse.from_dict(result)

    def stream_audit_events(
        self,
        agent_id: str | None = None,
        event_type: str | None = None,
        timeout: float | None = None,
    ) -> Generator[DakeraEvent, None, None]:
        """Stream live audit events via SSE (OBS-1).

        Opens a long-lived connection to ``GET /v1/audit/stream`` and yields
        :class:`~dakera.models.DakeraEvent` objects as they arrive.

        Args:
            agent_id: Scope the stream to a specific agent.
            event_type: Scope the stream to a specific event type.
            timeout: Optional read timeout in seconds.

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
        yield from self._stream_sse(url, timeout)

    def export_audit(
        self,
        format: str = "jsonl",
        agent_id: str | None = None,
        event_type: str | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        limit: int | None = None,
    ) -> AuditExportResponse:
        """Bulk-export audit log entries (``GET /v1/audit/export``, Admin scope).

        Args:
            format: ``"jsonl"`` (one JSON event per line), ``"json"`` (a JSON array)
                or ``"csv"``. The server speaks ``json`` and ``csv``; ``jsonl`` is
                produced here from the ``json`` answer.
            agent_id: Filter to a specific agent.
            event_type: Filter to a specific event type.
            from_ts: Unix timestamp lower bound.
            to_ts: Unix timestamp upper bound.
            limit: Maximum events (server default 10 000).

        Returns:
            :class:`AuditExportResponse` with the serialised data and count.
        """
        params: dict[str, Any] = {"format": "csv" if format == "csv" else "json"}
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
        result = self._request("GET", "/v1/audit/export", params=params)
        return _audit_export_response(result, format)

    # =========================================================================
    # EXT-1: External Extraction Providers
    # =========================================================================

    def extract_text(
        self,
        text: str,
        namespace: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> ExtractionResult:
        """Extract entities from text using a pluggable provider (EXT-1).

        The server selects the provider hierarchy: per-request override >
        namespace default > GLiNER (bundled, zero-config).

        Args:
            text: Input text to extract from.
            namespace: Namespace whose default extractor to inherit.
            provider: Override provider — one of ``"gliner"``, ``"openai"``,
                ``"anthropic"``, ``"openrouter"``, ``"ollama"``.
            model: Override model within the chosen provider.

        Returns:
            :class:`ExtractionResult` with entities and provider metadata.
        """
        body: dict[str, Any] = {"text": text}
        if namespace is not None:
            body["namespace"] = namespace
        if provider is not None:
            body["provider"] = provider
        if model is not None:
            body["model"] = model
        result = self._request("POST", "/v1/extract", data=body)
        return ExtractionResult.from_dict(result)

    def configure_namespace_extractor(
        self,
        namespace: str,
        provider: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Set the default extraction provider for a namespace (EXT-1).

        Args:
            namespace: The namespace to configure.
            provider: Default provider — ``"gliner"``, ``"openai"``,
                ``"anthropic"``, ``"openrouter"``, or ``"ollama"``.
            model: Default model within the provider (optional).

        Returns:
            Dict confirming the updated extractor configuration.
        """
        body: dict[str, Any] = {"provider": provider}
        if model is not None:
            body["model"] = model
        return self._request("PATCH", f"/v1/namespaces/{namespace}/extractor", data=body)

    # =========================================================================
    # ODE-2: GLiNER Entity Extraction (dakera-ode sidecar)
    # =========================================================================

    def ode_extract_entities(
        self,
        content: str,
        agent_id: str,
        memory_id: str | None = None,
        entity_types: list[str] | None = None,
    ) -> ExtractEntitiesResponse:
        """Extract named entities from text using the GLiNER sidecar (ODE-2).

        Calls ``POST /ode/extract`` on the dakera-ode sidecar service.
        Requires :attr:`ode_url` to be configured on the client.

        Unlike :meth:`extract_entities` (CE-4, server-side NER), this method
        calls the dedicated GLiNER sidecar and returns character offsets,
        model name, and processing time.

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
                "Pass ode_url='http://localhost:8080' to DakeraClient."
            )
        body: dict[str, Any] = {"content": content, "agent_id": agent_id}
        if memory_id is not None:
            body["memory_id"] = memory_id
        if entity_types is not None:
            body["entity_types"] = entity_types
        response = self._session.post(
            f"{self.ode_url}/ode/extract",
            json=body,
            timeout=(self.connect_timeout, self.timeout),
        )
        data = self._handle_response(response)
        return ExtractEntitiesResponse.from_dict(data)

    # =========================================================================
    # COG-1: Per-namespace Memory Lifecycle Policy
    # =========================================================================

    def get_memory_policy(self, namespace: str) -> MemoryPolicy:
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
        result = self._request("GET", f"/v1/namespaces/{namespace}/memory_policy")
        return MemoryPolicy.from_dict(result)

    def set_memory_policy(self, namespace: str, policy: MemoryPolicy) -> MemoryPolicy:
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
        result = self._request(
            "PUT",
            f"/v1/namespaces/{namespace}/memory_policy",
            data=policy.to_dict(),
        )
        return MemoryPolicy.from_dict(result)

    # =========================================================================
    # CE-54: Fulltext Reindex (Admin)
    # =========================================================================

    def admin_fulltext_reindex(self, namespace: str | None = None) -> FulltextReindexResponse:
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
        result = self._request("POST", "/v1/admin/fulltext/reindex", data=data)
        return FulltextReindexResponse.from_dict(result)

    # =========================================================================
    # Admin — Cluster & Maintenance
    # =========================================================================

    def admin_cluster_replication(self) -> dict[str, Any]:
        """GET /admin/cluster/replication — cluster replication status."""
        return self._request("GET", "/v1/admin/cluster/replication")

    def admin_list_shards(self) -> dict[str, Any]:
        """GET /admin/cluster/shards — list shards."""
        return self._request("GET", "/v1/admin/cluster/shards")

    def admin_rebalance_shards(
        self,
        shard_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """POST /admin/cluster/shards/rebalance — rebalance shards."""
        data: dict[str, Any] = {"dry_run": dry_run}
        if shard_ids is not None:
            data["shard_ids"] = shard_ids
        return self._request("POST", "/v1/admin/cluster/shards/rebalance", data=data)

    def admin_maintenance_status(self) -> dict[str, Any]:
        """GET /admin/cluster/maintenance — maintenance mode status."""
        return self._request("GET", "/v1/admin/cluster/maintenance")

    def admin_enable_maintenance(
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
        return self._request("POST", "/v1/admin/cluster/maintenance/enable", data=data)

    def admin_disable_maintenance(self, force: bool | None = None) -> dict[str, Any]:
        """POST /admin/cluster/maintenance/disable — disable maintenance mode."""
        data: dict[str, Any] = {}
        if force is not None:
            data["force"] = force
        return self._request("POST", "/v1/admin/cluster/maintenance/disable", data=data)

    # =========================================================================
    # Admin — Quotas
    # =========================================================================

    def admin_list_quotas(self) -> dict[str, Any]:
        """GET /admin/quotas — list all namespace quotas."""
        return self._request("GET", "/v1/admin/quotas")

    def admin_get_default_quota(self) -> dict[str, Any]:
        """GET /admin/quotas/default — get default quota configuration."""
        return self._request("GET", "/v1/admin/quotas/default")

    def admin_set_default_quota(self, config: dict[str, Any] | None) -> dict[str, Any]:
        """PUT /admin/quotas/default — set default quota configuration."""
        return self._request("PUT", "/v1/admin/quotas/default", data={"config": config})

    def admin_get_quota(self, namespace: str) -> dict[str, Any]:
        """GET /admin/quotas/{namespace} — get namespace quota."""
        return self._request("GET", f"/v1/admin/quotas/{namespace}")

    def admin_set_quota(self, namespace: str, config: dict[str, Any]) -> dict[str, Any]:
        """PUT /admin/quotas/{namespace} — set namespace quota."""
        return self._request("PUT", f"/v1/admin/quotas/{namespace}", data={"config": config})

    def admin_delete_quota(self, namespace: str) -> dict[str, Any]:
        """DELETE /admin/quotas/{namespace} — remove namespace quota."""
        return self._request("DELETE", f"/v1/admin/quotas/{namespace}")

    def admin_check_quota(
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
        return self._request("POST", f"/v1/admin/quotas/{namespace}/check", data=data)

    # =========================================================================
    # Admin — Slow Queries
    # =========================================================================

    def admin_list_slow_queries(
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
        return self._request("GET", "/v1/admin/slow-queries", params=params)

    def admin_slow_query_summary(self) -> dict[str, Any]:
        """GET /admin/slow-queries/summary — slow query summary."""
        return self._request("GET", "/v1/admin/slow-queries/summary")

    def admin_clear_slow_queries(self, namespace: str | None = None) -> dict[str, Any]:
        """DELETE /admin/slow-queries — clear slow query log."""
        params: dict[str, Any] = {}
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("DELETE", "/v1/admin/slow-queries", params=params)

    def admin_update_slow_query_config(self, **kwargs: Any) -> dict[str, Any]:
        """PATCH /admin/slow-queries/config — update slow query configuration."""
        return self._request("PATCH", "/v1/admin/slow-queries/config", data=kwargs)

    # =========================================================================
    # Admin — Backups
    # =========================================================================

    def admin_list_backups(self) -> dict[str, Any]:
        """GET /admin/backups — list all backups."""
        return self._request("GET", "/v1/admin/backups")

    def admin_create_backup(
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
        return self._request("POST", "/v1/admin/backups", data=data)

    def admin_get_backup(self, backup_id: str) -> dict[str, Any]:
        """GET /admin/backups/{id} — get backup details."""
        return self._request("GET", f"/v1/admin/backups/{backup_id}")

    def admin_delete_backup(self, backup_id: str) -> dict[str, Any]:
        """DELETE /admin/backups/{id} — delete a backup."""
        return self._request("DELETE", f"/v1/admin/backups/{backup_id}")

    def admin_get_backup_schedule(self) -> dict[str, Any]:
        """GET /admin/backups/schedule — get backup schedule."""
        return self._request("GET", "/v1/admin/backups/schedule")

    def admin_update_backup_schedule(self, **kwargs: Any) -> dict[str, Any]:
        """POST /admin/backups/schedule — update backup schedule."""
        return self._request("POST", "/v1/admin/backups/schedule", data=kwargs)

    def admin_restore_backup(
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
        return self._request("POST", "/v1/admin/backups/restore", data=data)

    def admin_get_restore_status(self, restore_id: str) -> dict[str, Any]:
        """GET /admin/backups/restore/{id} — restore operation status."""
        return self._request("GET", f"/v1/admin/backups/restore/{restore_id}")

    # =========================================================================
    # Ops — Diagnostics & Jobs
    # =========================================================================

    def ops_diagnostics(self) -> dict[str, Any]:
        """GET /ops/diagnostics — system diagnostics."""
        return self._request("GET", "/ops/diagnostics")

    def ops_list_jobs(self) -> list[dict[str, Any]]:
        """GET /ops/jobs — list background jobs."""
        return self._request("GET", "/ops/jobs")

    def ops_get_job(self, job_id: str) -> dict[str, Any]:
        """GET /ops/jobs/{id} — get job status."""
        return self._request("GET", f"/ops/jobs/{job_id}")

    def ops_compact(
        self,
        namespace: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """POST /ops/compact — trigger compaction."""
        data: dict[str, Any] = {"force": force}
        if namespace is not None:
            data["namespace"] = namespace
        return self._request("POST", "/ops/compact", data=data)

    def ops_shutdown(self) -> dict[str, Any]:
        """POST /ops/shutdown — request graceful shutdown."""
        return self._request("POST", "/ops/shutdown")

    # =========================================================================
    # Phase 3 — Engine Parity
    # =========================================================================

    def fulltext_stats(self, namespace: str) -> FullTextIndexStats:
        """GET /v1/namespaces/{namespace}/fulltext/stats — full-text index statistics."""
        data = self._request("GET", f"/v1/namespaces/{namespace}/fulltext/stats")
        return FullTextIndexStats.from_dict(data)

    def fulltext_delete(self, namespace: str, ids: list[str]) -> dict[str, Any]:
        """Delete documents from the full-text index.

        Calls ``POST /v1/namespaces/{namespace}/fulltext/delete``.
        """
        return self._request(
            "POST", f"/v1/namespaces/{namespace}/fulltext/delete", data={"ids": ids}
        )

    def ttl_stats(self) -> TtlStatsResponse:
        """GET /admin/ttl/stats — TTL expiration statistics across namespaces."""
        data = self._request("GET", "/v1/admin/ttl/stats")
        return TtlStatsResponse.from_dict(data)

    def ttl_cleanup(self, namespace: str | None = None) -> TtlCleanupResponse:
        """POST /admin/ttl/cleanup — remove expired vectors, optionally scoped to a namespace."""
        body: dict[str, Any] = {}
        if namespace is not None:
            body["namespace"] = namespace
        data = self._request("POST", "/v1/admin/ttl/cleanup", data=body if body else None)
        return TtlCleanupResponse.from_dict(data)

    def route_query(
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
        resp = self._request("POST", "/v1/route", data=data)
        return RouteResponse.from_dict(resp)

    def import_job_status(self, job_id: str) -> ImportJobStatus:
        """GET /v1/import/{job_id}/status — check import job progress."""
        data = self._request("GET", f"/v1/import/{job_id}/status")
        return ImportJobStatus.from_dict(data)

    def download_backup(self, backup_id: str) -> bytes:
        """GET /admin/backups/{id}/download — download a backup as gzip bytes."""
        url = self._url(f"/v1/admin/backups/{backup_id}/download")
        response = self._session.get(url, timeout=(self.connect_timeout, self.timeout))
        if not response.ok:
            self._handle_response(response)
        return response.content

    def upload_backup(self, data: bytes) -> dict[str, Any]:
        """POST /admin/backups/upload — upload a gzip backup."""
        url = self._url("/v1/admin/backups/upload")
        response = self._session.post(
            url,
            data=data,
            headers={"Content-Type": "application/gzip"},
            timeout=(self.connect_timeout, self.timeout),
        )
        if not response.ok:
            self._handle_response(response)
        return response.json()

    def storage_tier_overview(self) -> StorageTierOverview:
        """GET /admin/storage/tiers — storage tier architecture overview."""
        data = self._request("GET", "/v1/admin/storage/tiers")
        return StorageTierOverview.from_dict(data)

    def background_activity(self) -> dict[str, Any]:
        """GET /admin/background-activity — current background tasks and jobs."""
        return self._request("GET", "/v1/admin/background-activity")

    def memory_type_stats(self) -> MemoryTypeStatsResponse:
        """GET /admin/memory-type-stats — memory type distribution statistics."""
        data = self._request("GET", "/v1/admin/memory-type-stats")
        return MemoryTypeStatsResponse.from_dict(data)

    def migrate_namespace_dimensions(
        self,
        target_dimension: int = 1024,
        namespaces: list[str] | None = None,
    ) -> MigrateDimensionsResponse:
        """POST /admin/namespaces/migrate-dimensions — migrate namespace vector dimensions."""
        data: dict[str, Any] = {"target_dimension": target_dimension}
        if namespaces is not None:
            data["namespaces"] = namespaces
        resp = self._request("POST", "/v1/admin/namespaces/migrate-dimensions", data=data)
        return MigrateDimensionsResponse.from_dict(resp)

    def drain_reembed(
        self,
        timeout_secs: int | None = None,
        batch_size: int | None = None,
        min_importance: float | None = None,
    ) -> DrainReembedResponse:
        """``POST /admin/reembed/drain`` — drain static vectors to full ONNX quality (v0.11.82+).

        Runs the re-embedding upgrade loop until zero ``_embedding_kind=static``
        candidates remain across all namespaces (or ``timeout_secs`` elapses).
        Requires Admin scope. Useful as a pre-bench steady-state gate when
        ``DAKERA_TIERED=1``.

        Args:
            timeout_secs: Hard wall-clock cap in seconds (default 600).
            batch_size: Candidates upgraded per cycle (default 10000).
            min_importance: Minimum importance threshold; defaults to 0.0
                (upgrade all statics — differs from background job default of 0.5).

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
        resp = self._request("POST", "/v1/admin/reembed/drain", data=body if body else None)
        return DrainReembedResponse.from_dict(resp)

    def admin_reembed_static_count(self) -> StaticCountResponse:
        """``GET /admin/reembed/static-count`` — static vectors pending re-embedding.

        Returns the number of ``_embedding_kind=static`` vectors that have not yet
        been upgraded to full ONNX quality. Operators can poll this alongside
        :meth:`drain_reembed` to monitor drain progress. A ``static_count`` of 0
        means steady state — no vectors awaiting upgrade. Requires Admin scope.

        Returns:
            :class:`StaticCountResponse` with ``static_count`` field.
        """
        resp = self._request("GET", "/v1/admin/reembed/static-count")
        return StaticCountResponse.from_dict(resp)

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
