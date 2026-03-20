"""
Dakera Client

Main client class for interacting with Dakera server.
"""

import json
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urljoin

import requests

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
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize Dakera client.

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

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

        if headers:
            self._session.headers.update(headers)

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _handle_response(self, response: requests.Response) -> Any:
        """Handle API response and raise appropriate exceptions."""
        try:
            body = response.json() if response.content else None
        except json.JSONDecodeError:
            body = response.text

        if response.status_code == 200 or response.status_code == 201:
            return body
        elif response.status_code == 204:
            return None
        elif response.status_code == 400:
            raise ValidationError(
                message=(
                    body.get("error", "Validation error")
                    if isinstance(body, dict) else str(body)
                ),
                status_code=response.status_code,
                response_body=body,
            )
        elif response.status_code == 401:
            raise AuthenticationError(
                message="Authentication failed",
                status_code=response.status_code,
                response_body=body,
            )
        elif response.status_code == 404:
            raise NotFoundError(
                message=(
                    body.get("error", "Resource not found")
                    if isinstance(body, dict) else str(body)
                ),
                status_code=response.status_code,
                response_body=body,
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
            )
        else:
            raise DakeraError(
                message=f"Unexpected status code: {response.status_code}",
                status_code=response.status_code,
                response_body=body,
            )

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make HTTP request with retry logic."""
        url = self._url(path)
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout,
                )
                return self._handle_response(response)
            except requests.exceptions.ConnectionError as e:
                last_exception = ConnectionError(f"Failed to connect to {url}: {e}")
            except requests.exceptions.Timeout as e:
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

    def upsert(
        self,
        namespace: str,
        vectors: List[VectorInput],
    ) -> Dict[str, Any]:
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
        vector: List[float],
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
        data: Dict[str, Any] = {
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
        ids: Optional[List[str]] = None,
        filter: Optional[FilterDict] = None,
        delete_all: bool = False,
    ) -> Dict[str, Any]:
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
        data: Dict[str, Any] = {}
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
        ids: List[str],
        include_values: bool = True,
        include_metadata: bool = True,
    ) -> List[Vector]:
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
        queries: List[Dict[str, Any]],
    ) -> List[SearchResult]:
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
        documents: List[TextDocumentInput],
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

        data: Dict[str, Any] = {"documents": doc_dicts}
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
        data: Dict[str, Any] = {
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
        queries: List[str],
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
        data: Dict[str, Any] = {
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
        documents: List[DocumentInput],
    ) -> Dict[str, Any]:
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
    ) -> List[FullTextSearchResult]:
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
        data: Dict[str, Any] = {"query": query, "top_k": top_k}
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
        vector: List[float],
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filter: Optional[FilterDict] = None,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector and full-text.

        Args:
            namespace: Target namespace
            vector: Query vector
            query: Text query string
            top_k: Number of results to return (default: 10)
            alpha: Balance between vector (0) and text (1) search (default: 0.5)
            filter: Optional metadata filter

        Returns:
            List of HybridSearchResult objects

        Example:
            >>> results = client.hybrid_search(
            ...     "my-namespace",
            ...     vector=[0.1, 0.2, 0.3],
            ...     query="hello world",
            ...     alpha=0.7,
            ... )
        """
        data: Dict[str, Any] = {
            "vector": vector,
            "query": query,
            "top_k": top_k,
            "alpha": alpha,
        }
        if filter:
            data["filter"] = filter

        response = self._request(
            "POST",
            f"/v1/namespaces/{namespace}/fulltext/hybrid",
            data=data,
        )
        return [HybridSearchResult.from_dict(r) for r in response.get("results", [])]

    # =========================================================================
    # Namespace Operations
    # =========================================================================

    def list_namespaces(self) -> List[NamespaceInfo]:
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
        metadata: Optional[Dict[str, Any]] = None,
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
        data: Dict[str, Any] = {"name": namespace}
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

    def health(self) -> Dict[str, Any]:
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

    def compact(self, namespace: str) -> Dict[str, Any]:
        """
        Trigger compaction for a namespace.

        Args:
            namespace: Namespace name

        Returns:
            Compaction status
        """
        return self._request("POST", f"/v1/namespaces/{namespace}/compact")

    def flush(self, namespace: str) -> Dict[str, Any]:
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
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a memory for an agent."""
        data: Dict[str, Any] = {"content": content, "memory_type": memory_type}
        if importance is not None:
            data["importance"] = importance
        if metadata is not None:
            data["metadata"] = metadata
        if session_id is not None:
            data["session_id"] = session_id
        return self._request("POST", f"/v1/agents/{agent_id}/memories", data=data)

    def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Recall memories for an agent."""
        data: Dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        result = self._request("POST", f"/v1/agents/{agent_id}/memories/recall", data=data)
        return result.get("memories", result) if isinstance(result, dict) else result

    def get_memory(self, agent_id: str, memory_id: str) -> Dict[str, Any]:
        """Get a specific memory."""
        return self._request("GET", f"/v1/agents/{agent_id}/memories/{memory_id}")

    def update_memory(
        self,
        agent_id: str,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        data: Dict[str, Any] = {}
        if content is not None:
            data["content"] = content
        if metadata is not None:
            data["metadata"] = metadata
        if memory_type is not None:
            data["memory_type"] = memory_type
        return self._request("PUT", f"/v1/agents/{agent_id}/memories/{memory_id}", data=data)

    def forget(self, agent_id: str, memory_id: str) -> Dict[str, Any]:
        """Delete a memory."""
        return self._request("DELETE", f"/v1/agents/{agent_id}/memories/{memory_id}")

    def search_memories(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories for an agent."""
        data: Dict[str, Any] = {"query": query, "top_k": top_k}
        if memory_type is not None:
            data["memory_type"] = memory_type
        if min_importance is not None:
            data["min_importance"] = min_importance
        result = self._request("POST", f"/v1/agents/{agent_id}/memories/search", data=data)
        return result.get("memories", result) if isinstance(result, dict) else result

    def update_importance(
        self,
        agent_id: str,
        memory_ids: List[str],
        importance: float,
    ) -> Dict[str, Any]:
        """Update importance of memories."""
        data = {"memory_ids": memory_ids, "importance": importance}
        return self._request("PUT", f"/v1/agents/{agent_id}/memories/importance", data=data)

    def consolidate(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Consolidate memories for an agent."""
        data: Dict[str, Any] = {"dry_run": dry_run}
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
    ) -> Dict[str, Any]:
        """Submit feedback on a memory recall."""
        data: Dict[str, Any] = {"memory_id": memory_id, "feedback": feedback}
        if relevance_score is not None:
            data["relevance_score"] = relevance_score
        return self._request("POST", f"/v1/agents/{agent_id}/memories/feedback", data=data)

    # =========================================================================
    # Session Operations
    # =========================================================================

    def start_session(
        self,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start a new session."""
        data: Dict[str, Any] = {"agent_id": agent_id}
        if metadata is not None:
            data["metadata"] = metadata
        return self._request("POST", "/v1/sessions/start", data=data)

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a session."""
        return self._request("POST", f"/v1/sessions/{session_id}/end")

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session details."""
        return self._request("GET", f"/v1/sessions/{session_id}")

    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        active_only: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List sessions."""
        params: Dict[str, Any] = {}
        if agent_id is not None:
            params["agent_id"] = agent_id
        if active_only is not None:
            params["active_only"] = str(active_only).lower()
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._request("GET", "/v1/sessions", params=params)

    def session_memories(self, session_id: str) -> List[Dict[str, Any]]:
        """Get memories for a session."""
        return self._request("GET", f"/v1/sessions/{session_id}/memories")

    # =========================================================================
    # Agent Operations
    # =========================================================================

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents."""
        return self._request("GET", "/v1/agents")

    def agent_memories(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get memories for an agent."""
        params: Dict[str, Any] = {}
        if memory_type is not None:
            params["memory_type"] = memory_type
        if limit is not None:
            params["limit"] = limit
        return self._request("GET", f"/v1/agents/{agent_id}/memories", params=params)

    def agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get stats for an agent."""
        return self._request("GET", f"/v1/agents/{agent_id}/stats")

    def agent_sessions(
        self,
        agent_id: str,
        active_only: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get sessions for an agent."""
        params: Dict[str, Any] = {}
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
        vector_ids: Optional[List[str]] = None,
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
        positive: List[List[float]],
        negative: Optional[List[List[float]]] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        mmr_lambda: Optional[float] = None,
        mmr_prefetch_k: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        data: Dict[str, Any] = {
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
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        vector_weight: Optional[float] = None,
        text_weight: Optional[float] = None,
        fusion_method: Optional[str] = None,
        rerank: bool = False,
    ) -> Dict[str, Any]:
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
        data: Dict[str, Any] = {
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
        vector: Optional[List[float]] = None,
        group_by: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        top_groups: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        data: Dict[str, Any] = {}
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
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = True,
    ) -> Dict[str, Any]:
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
        data: Dict[str, Any] = {
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
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
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
        data: Dict[str, Any] = {
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
        ids: List[str],
        vectors: List[List[float]],
        attributes: Optional[Dict[str, List[Any]]] = None,
        ttl_seconds: Optional[int] = None,
        dimension: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        data: Dict[str, Any] = {
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
    ) -> Dict[str, Any]:
        """Build a knowledge graph from a seed memory."""
        data: Dict[str, Any] = {"agent_id": agent_id}
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
    ) -> Dict[str, Any]:
        """Build a full knowledge graph for an agent."""
        data: Dict[str, Any] = {"agent_id": agent_id}
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
        memory_ids: Optional[List[str]] = None,
        target_type: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Summarize memories."""
        data: Dict[str, Any] = {"agent_id": agent_id, "dry_run": dry_run}
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
    ) -> Dict[str, Any]:
        """Deduplicate memories."""
        data: Dict[str, Any] = {"agent_id": agent_id, "dry_run": dry_run}
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
    ) -> Dict[str, Any]:
        """Get analytics overview."""
        params: Dict[str, Any] = {}
        if period is not None:
            params["period"] = period
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("GET", "/v1/analytics/overview", params=params)

    def analytics_latency(
        self,
        period: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get latency analytics."""
        params: Dict[str, Any] = {}
        if period is not None:
            params["period"] = period
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("GET", "/v1/analytics/latency", params=params)

    def analytics_throughput(
        self,
        period: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get throughput analytics."""
        params: Dict[str, Any] = {}
        if period is not None:
            params["period"] = period
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("GET", "/v1/analytics/throughput", params=params)

    def analytics_storage(
        self,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get storage analytics."""
        params: Dict[str, Any] = {}
        if namespace is not None:
            params["namespace"] = namespace
        return self._request("GET", "/v1/analytics/storage", params=params)

    # =========================================================================
    # Admin Operations (Extended)
    # =========================================================================

    def cluster_status(self) -> Dict[str, Any]:
        """Get cluster status."""
        return self._request("GET", "/v1/admin/cluster/status")

    def cluster_nodes(self) -> List[Dict[str, Any]]:
        """Get cluster nodes."""
        return self._request("GET", "/v1/admin/cluster/nodes")

    def optimize_namespace(self, namespace: str) -> Dict[str, Any]:
        """Optimize a namespace."""
        return self._request("POST", f"/v1/admin/namespaces/{namespace}/optimize")

    def index_stats(self, namespace: str) -> Dict[str, Any]:
        """Get index stats for a namespace."""
        return self._request("GET", f"/v1/admin/namespaces/{namespace}/index/stats")

    def rebuild_indexes(self, namespace: str) -> Dict[str, Any]:
        """Rebuild indexes for a namespace."""
        return self._request("POST", f"/v1/admin/namespaces/{namespace}/index/rebuild")

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._request("GET", "/v1/admin/cache/stats")

    def cache_clear(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Clear cache, optionally for a specific namespace."""
        data: Optional[Dict[str, Any]] = None
        if namespace is not None:
            data = {"namespace": namespace}
        return self._request("POST", "/v1/admin/cache/clear", data=data)

    def get_config(self) -> Dict[str, Any]:
        """Get server configuration."""
        return self._request("GET", "/v1/admin/config")

    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update server configuration."""
        return self._request("PUT", "/v1/admin/config", data=config)

    def get_quotas(self) -> Dict[str, Any]:
        """Get quota settings."""
        return self._request("GET", "/v1/admin/quotas")

    def update_quotas(self, quotas: Dict[str, Any]) -> Dict[str, Any]:
        """Update quota settings."""
        return self._request("PUT", "/v1/admin/quotas", data=quotas)

    def slow_queries(
        self,
        limit: Optional[int] = None,
        min_duration_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get slow queries."""
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if min_duration_ms is not None:
            params["min_duration_ms"] = min_duration_ms
        return self._request("GET", "/v1/admin/slow-queries", params=params if params else None)

    def create_backup(self, include_data: bool = True) -> Dict[str, Any]:
        """Create a backup."""
        return self._request("POST", "/v1/admin/backups", data={"include_data": include_data})

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all backups."""
        return self._request("GET", "/v1/admin/backups")

    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore a backup."""
        return self._request("POST", f"/v1/admin/backups/{backup_id}/restore")

    def delete_backup(self, backup_id: str) -> Dict[str, Any]:
        """Delete a backup."""
        return self._request("DELETE", f"/v1/admin/backups/{backup_id}")

    def configure_ttl(
        self,
        namespace: str,
        ttl_seconds: int,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure TTL for a namespace."""
        data: Dict[str, Any] = {"ttl_seconds": ttl_seconds}
        if strategy is not None:
            data["strategy"] = strategy
        return self._request("POST", f"/v1/admin/namespaces/{namespace}/ttl", data=data)

    # =========================================================================
    # API Key Operations
    # =========================================================================

    def create_key(
        self,
        name: str,
        permissions: Optional[List[str]] = None,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new API key."""
        data: Dict[str, Any] = {"name": name}
        if permissions is not None:
            data["permissions"] = permissions
        if expires_at is not None:
            data["expires_at"] = expires_at
        return self._request("POST", "/v1/keys", data=data)

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys."""
        return self._request("GET", "/v1/keys")

    def get_key(self, key_id: str) -> Dict[str, Any]:
        """Get an API key by ID."""
        return self._request("GET", f"/v1/keys/{key_id}")

    def delete_key(self, key_id: str) -> Dict[str, Any]:
        """Delete an API key."""
        return self._request("DELETE", f"/v1/keys/{key_id}")

    def deactivate_key(self, key_id: str) -> Dict[str, Any]:
        """Deactivate an API key."""
        return self._request("POST", f"/v1/keys/{key_id}/deactivate")

    def rotate_key(self, key_id: str) -> Dict[str, Any]:
        """Rotate an API key."""
        return self._request("POST", f"/v1/keys/{key_id}/rotate")

    def key_usage(self, key_id: str) -> Dict[str, Any]:
        """Get usage statistics for an API key."""
        return self._request("GET", f"/v1/keys/{key_id}/usage")

    # =========================================================================
    # SSE Streaming (CE-1)
    # =========================================================================

    def _parse_sse_block(self, block: str) -> Optional[DakeraEvent]:
        """Parse a single SSE event block into a :class:`~dakera.models.DakeraEvent`."""
        data_lines: List[str] = []
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
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass

    # =========================================================================
    # DASH-A: Cross-Agent Network
    # =========================================================================

    def cross_agent_network(
        self,
        agent_ids: Optional[List[str]] = None,
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
        payload: Dict[str, Any] = {
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
