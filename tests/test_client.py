"""Tests for Dakera client."""

from unittest.mock import patch

import pytest
import responses

from dakera import (
    AsyncDakeraClient,
    DakeraClient,
    DakeraEvent,
    Document,
    MemoryEvent,
    NotFoundError,
    ServerError,
    ValidationError,
    Vector,
)


@pytest.fixture
def client():
    """Create a test client."""
    return DakeraClient("http://localhost:3000")


@pytest.fixture
def mock_responses():
    """Enable responses mocking."""
    with responses.RequestsMock() as rsps:
        yield rsps


class TestVectorOperations:
    """Tests for vector operations."""

    def test_upsert_with_dicts(self, client, mock_responses):
        """Test upserting vectors using dictionaries."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/vectors",
            json={"upserted_count": 2},
            status=200,
        )

        result = client.upsert(
            "test-ns",
            vectors=[
                {"id": "vec1", "values": [0.1, 0.2, 0.3]},
                {"id": "vec2", "values": [0.4, 0.5, 0.6]},
            ],
        )

        assert result["upserted_count"] == 2
        assert len(mock_responses.calls) == 1

    def test_upsert_with_vector_objects(self, client, mock_responses):
        """Test upserting vectors using Vector dataclass."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/vectors",
            json={"upserted_count": 1},
            status=200,
        )

        result = client.upsert(
            "test-ns",
            vectors=[Vector(id="vec1", values=[0.1, 0.2, 0.3], metadata={"key": "value"})],
        )

        assert result["upserted_count"] == 1

    def test_query(self, client, mock_responses):
        """Test querying vectors."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/query",
            json={
                "results": [
                    {"id": "vec1", "score": 0.95, "metadata": {"key": "value"}},
                    {"id": "vec2", "score": 0.85},
                ],
                "total_searched": 100,
            },
            status=200,
        )

        result = client.query("test-ns", vector=[0.1, 0.2, 0.3], top_k=10)

        assert len(result.results) == 2
        assert result.results[0].id == "vec1"
        assert result.results[0].score == 0.95
        assert result.results[0].metadata == {"key": "value"}
        assert result.total_searched == 100

    def test_query_with_filter(self, client, mock_responses):
        """Test querying vectors with metadata filter."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/query",
            json={"results": []},
            status=200,
        )

        result = client.query(
            "test-ns",
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            filter={"category": {"$eq": "test"}},
        )

        assert len(result.results) == 0
        request_body = mock_responses.calls[0].request.body
        assert b"filter" in request_body

    def test_delete_by_ids(self, client, mock_responses):
        """Test deleting vectors by IDs."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/delete",
            json={"deleted_count": 2},
            status=200,
        )

        result = client.delete("test-ns", ids=["vec1", "vec2"])

        assert result["deleted_count"] == 2

    def test_delete_by_filter(self, client, mock_responses):
        """Test deleting vectors by filter."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/delete",
            json={"deleted_count": 5},
            status=200,
        )

        result = client.delete("test-ns", filter={"status": {"$eq": "obsolete"}})

        assert result["deleted_count"] == 5

    def test_fetch(self, client, mock_responses):
        """Test fetching vectors by ID."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/fetch",
            json={
                "vectors": [
                    {"id": "vec1", "values": [0.1, 0.2, 0.3]},
                    {"id": "vec2", "values": [0.4, 0.5, 0.6]},
                ]
            },
            status=200,
        )

        vectors = client.fetch("test-ns", ids=["vec1", "vec2"])

        assert len(vectors) == 2
        assert vectors[0].id == "vec1"
        assert vectors[0].values == [0.1, 0.2, 0.3]

    def test_batch_query(self, client, mock_responses):
        """Test batch querying."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/batch-query",
            json={
                "results": [
                    {"results": [{"id": "vec1", "score": 0.9}]},
                    {"results": [{"id": "vec2", "score": 0.8}]},
                ]
            },
            status=200,
        )

        results = client.batch_query(
            "test-ns",
            queries=[
                {"vector": [0.1, 0.2, 0.3], "top_k": 1},
                {"vector": [0.4, 0.5, 0.6], "top_k": 1},
            ],
        )

        assert len(results) == 2
        assert results[0].results[0].id == "vec1"


class TestFullTextOperations:
    """Tests for full-text search operations."""

    def test_index_documents(self, client, mock_responses):
        """Test indexing documents."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/fulltext/index",
            json={"indexed_count": 2},
            status=200,
        )

        result = client.index_documents(
            "test-ns",
            documents=[
                {"id": "doc1", "content": "Hello world"},
                Document(id="doc2", content="Goodbye world"),
            ],
        )

        assert result["indexed_count"] == 2

    def test_fulltext_search(self, client, mock_responses):
        """Test full-text search."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/fulltext/search",
            json={
                "results": [
                    {"id": "doc1", "score": 2.5, "content": "Hello world"},
                ]
            },
            status=200,
        )

        results = client.fulltext_search("test-ns", query="hello", top_k=10)

        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].score == 2.5

    def test_hybrid_search(self, client, mock_responses):
        """Test hybrid search."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/hybrid",
            json={
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.85,
                        "vector_score": 0.9,
                        "text_score": 0.8,
                    },
                ]
            },
            status=200,
        )

        results = client.hybrid_search(
            "test-ns",
            vector=[0.1, 0.2, 0.3],
            query="hello",
            top_k=10,
            alpha=0.5,
        )

        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].vector_score == 0.9
        assert results[0].text_score == 0.8
        # Verify correct endpoint was called
        assert len(mock_responses.calls) == 1
        assert "/v1/namespaces/test-ns/hybrid" in mock_responses.calls[0].request.url

    def test_hybrid_search_bm25_only(self, client, mock_responses):
        """Test hybrid search with no vector (BM25-only fallback)."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/hybrid",
            json={
                "results": [{"id": "doc2", "score": 0.75, "vector_score": 0, "text_score": 0.75}]
            },
            status=200,
        )

        results = client.hybrid_search("test-ns", query="hello")

        assert len(results) == 1
        assert results[0].id == "doc2"
        # Verify correct endpoint was called
        assert "/v1/namespaces/test-ns/hybrid" in mock_responses.calls[-1].request.url
        # Verify vector was not sent in request body
        sent_body = mock_responses.calls[-1].request.body
        import json
        assert "vector" not in json.loads(sent_body)


class TestNamespaceOperations:
    """Tests for namespace operations."""

    def test_list_namespaces(self, client, mock_responses):
        """Test listing namespaces."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces",
            json={
                "namespaces": [
                    {"name": "ns1", "vector_count": 100},
                    {"name": "ns2", "vector_count": 200},
                ]
            },
            status=200,
        )

        namespaces = client.list_namespaces()

        assert len(namespaces) == 2
        assert namespaces[0].name == "ns1"
        assert namespaces[0].vector_count == 100

    def test_get_namespace(self, client, mock_responses):
        """Test getting namespace info."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/test-ns",
            json={
                "name": "test-ns",
                "vector_count": 1000,
                "dimensions": 384,
                "index_type": "hnsw",
            },
            status=200,
        )

        info = client.get_namespace("test-ns")

        assert info.name == "test-ns"
        assert info.vector_count == 1000
        assert info.dimensions == 384
        assert info.index_type == "hnsw"

    def test_create_namespace(self, client, mock_responses):
        """Test creating a namespace."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces",
            json={
                "name": "new-ns",
                "vector_count": 0,
                "dimensions": 384,
                "index_type": "hnsw",
            },
            status=200,
        )

        info = client.create_namespace("new-ns", dimensions=384, index_type="hnsw")

        assert info.name == "new-ns"
        assert info.dimensions == 384

    def test_delete_namespace(self, client, mock_responses):
        """Test deleting a namespace."""
        mock_responses.add(
            responses.DELETE,
            "http://localhost:3000/v1/namespaces/test-ns",
            status=204,
        )

        client.delete_namespace("test-ns")

        assert len(mock_responses.calls) == 1


class TestErrorHandling:
    """Tests for error handling."""

    def test_not_found_error(self, client, mock_responses):
        """Test 404 error handling."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/nonexistent",
            json={"error": "Namespace not found"},
            status=404,
        )

        with pytest.raises(NotFoundError) as exc_info:
            client.get_namespace("nonexistent")

        assert exc_info.value.status_code == 404

    def test_validation_error(self, client, mock_responses):
        """Test 400 error handling."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/query",
            json={"error": "Invalid vector dimensions"},
            status=400,
        )

        with pytest.raises(ValidationError) as exc_info:
            client.query("test-ns", vector=[0.1])

        assert exc_info.value.status_code == 400

    def test_server_error(self, client, mock_responses):
        """Test 500 error handling."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"error": "Internal server error"},
            status=500,
        )

        with pytest.raises(ServerError) as exc_info:
            client.health()

        assert exc_info.value.status_code == 500


class TestClientConfiguration:
    """Tests for client configuration."""

    def test_custom_timeout(self, mock_responses):
        """Test custom timeout configuration."""
        client = DakeraClient("http://localhost:3000", timeout=60.0)
        assert client.timeout == 60.0

    def test_api_key_header(self, mock_responses):
        """Test API key is added to headers."""
        client = DakeraClient("http://localhost:3000", api_key="test-key")
        assert "Authorization" in client._session.headers
        assert client._session.headers["Authorization"] == "Bearer test-key"

    def test_custom_headers(self, mock_responses):
        """Test custom headers are added."""
        client = DakeraClient(
            "http://localhost:3000",
            headers={"X-Custom": "value"},
        )
        assert client._session.headers["X-Custom"] == "value"

    def test_context_manager(self, mock_responses):
        """Test context manager support."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"status": "ok"},
            status=200,
        )

        with DakeraClient("http://localhost:3000") as client:
            result = client.health()
            assert result["status"] == "ok"


class TestModels:
    """Tests for data models."""

    def test_vector_to_dict(self):
        """Test Vector.to_dict()."""
        vector = Vector(id="vec1", values=[0.1, 0.2], metadata={"key": "value"})
        d = vector.to_dict()

        assert d["id"] == "vec1"
        assert d["values"] == [0.1, 0.2]
        assert d["metadata"] == {"key": "value"}

    def test_vector_from_dict(self):
        """Test Vector.from_dict()."""
        vector = Vector.from_dict(
            {"id": "vec1", "values": [0.1, 0.2], "metadata": {"key": "value"}}
        )

        assert vector.id == "vec1"
        assert vector.values == [0.1, 0.2]
        assert vector.metadata == {"key": "value"}

    def test_document_to_dict(self):
        """Test Document.to_dict()."""
        doc = Document(id="doc1", content="Hello", metadata={"type": "greeting"})
        d = doc.to_dict()

        assert d["id"] == "doc1"
        assert d["content"] == "Hello"
        assert d["metadata"] == {"type": "greeting"}


class TestRetryConfig:
    """Tests for RetryConfig and retry behavior."""

    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        from dakera import RetryConfig
        rc = RetryConfig()
        assert rc.max_retries == 3
        assert rc.base_delay == 0.1
        assert rc.max_delay == 60.0
        assert rc.jitter is True

    def test_retry_config_custom(self):
        """Test RetryConfig accepts custom values."""
        from dakera import RetryConfig
        rc = RetryConfig(max_retries=5, base_delay=0.5, max_delay=30.0, jitter=False)
        assert rc.max_retries == 5
        assert rc.base_delay == 0.5
        assert rc.max_delay == 30.0
        assert rc.jitter is False

    def test_client_accepts_retry_config(self):
        """Test DakeraClient accepts RetryConfig."""
        from dakera import RetryConfig
        rc = RetryConfig(max_retries=5, base_delay=0.2, jitter=False)
        client = DakeraClient("http://localhost:3000", retry_config=rc)
        assert client._retry_config.max_retries == 5
        assert client._retry_config.base_delay == 0.2

    def test_client_connect_timeout(self):
        """Test DakeraClient accepts connect_timeout."""
        client = DakeraClient("http://localhost:3000", timeout=30.0, connect_timeout=5.0)
        assert client.connect_timeout == 5.0
        assert client.timeout == 30.0

    def test_client_connect_timeout_defaults_to_timeout(self):
        """Test connect_timeout defaults to timeout when not set."""
        client = DakeraClient("http://localhost:3000", timeout=15.0)
        assert client.connect_timeout == 15.0

    def test_max_retries_param_sets_retry_config(self):
        """Test max_retries param is reflected in _retry_config."""
        client = DakeraClient("http://localhost:3000", max_retries=7)
        assert client._retry_config.max_retries == 7

    def test_retry_on_server_error(self, mock_responses):
        """Test that 5xx errors are retried up to max_retries."""
        from dakera import RetryConfig
        rc = RetryConfig(max_retries=3, base_delay=0.0, jitter=False)
        client = DakeraClient("http://localhost:3000", retry_config=rc)

        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"error": "internal error"},
            status=500,
        )
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"error": "internal error"},
            status=500,
        )
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"healthy": True},
            status=200,
        )

        result = client.health()
        assert result["healthy"] is True
        assert len(mock_responses.calls) == 3

    def test_retry_exhausted_raises(self, mock_responses):
        """Test that exhausting retries re-raises the last error."""
        from dakera import RetryConfig, ServerError
        rc = RetryConfig(max_retries=2, base_delay=0.0, jitter=False)
        client = DakeraClient("http://localhost:3000", retry_config=rc)

        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"error": "server error"},
            status=500,
        )
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"error": "server error"},
            status=500,
        )

        with pytest.raises(ServerError):
            client.health()
        assert len(mock_responses.calls) == 2

    def test_no_retry_on_4xx(self, mock_responses):
        """Test that 4xx errors (except 429) are NOT retried."""
        from dakera import NotFoundError, RetryConfig
        rc = RetryConfig(max_retries=3, base_delay=0.0, jitter=False)
        client = DakeraClient("http://localhost:3000", retry_config=rc)

        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/does-not-exist",
            json={"error": "not found"},
            status=404,
        )

        with pytest.raises(NotFoundError):
            client.get_namespace("does-not-exist")
        assert len(mock_responses.calls) == 1

    def test_retry_after_respected_on_429(self, mock_responses):
        """Test that Retry-After header is used as wait time on 429."""
        import time  # noqa: PLC0415

        from dakera import RetryConfig
        rc = RetryConfig(max_retries=2, base_delay=10.0, jitter=False)
        client = DakeraClient("http://localhost:3000", retry_config=rc)

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/ns/vectors",
            headers={"Retry-After": "0"},
            json={"error": "rate limited"},
            status=429,
        )
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/ns/vectors",
            json={"upserted_count": 1},
            status=200,
        )

        start = time.monotonic()
        result = client.upsert("ns", vectors=[{"id": "v1", "values": [0.1]}])
        elapsed = time.monotonic() - start

        assert result["upserted_count"] == 1
        # Retry-After=0, not the 10s base_delay — should be fast
        assert elapsed < 2.0
        assert len(mock_responses.calls) == 2


class TestBatchMemoryOperations:
    """Tests for CE-2 batch recall/forget (v0.7.0)."""

    def test_batch_recall_sends_correct_request(self, client, mock_responses):
        """batch_recall() POSTs to /v1/memories/recall/batch and returns BatchRecallResponse."""
        from dakera import BatchMemoryFilter, BatchRecallRequest, BatchRecallResponse

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/recall/batch",
            json={
                "memories": [
                    {
                        "id": "mem_1",
                        "agent_id": "qa",
                        "content": "test memory",
                        "importance": 0.8,
                        "memory_type": "episodic",
                        "tags": ["test"],
                        "created_at": 1700000000,
                        "last_accessed_at": 1700000000,
                        "access_count": 1,
                    }
                ],
                "total": 10,
                "filtered": 1,
            },
            status=200,
        )

        filt = BatchMemoryFilter(tags=["test"], min_importance=0.5)
        req = BatchRecallRequest("qa", filter=filt, limit=50)
        resp = client.batch_recall(req)

        assert isinstance(resp, BatchRecallResponse)
        assert resp.total == 10
        assert resp.filtered == 1
        assert len(resp.memories) == 1
        assert resp.memories[0].id == "mem_1"
        # Verify correct HTTP method + URL
        assert mock_responses.calls[0].request.method == "POST"
        assert "/v1/memories/recall/batch" in mock_responses.calls[0].request.url

    def test_batch_recall_no_filter(self, client, mock_responses):
        """batch_recall() works with no filter (returns all up to limit)."""
        from dakera import BatchRecallRequest, BatchRecallResponse

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/recall/batch",
            json={"memories": [], "total": 0, "filtered": 0},
            status=200,
        )

        req = BatchRecallRequest("agent-x")
        resp = client.batch_recall(req)

        assert isinstance(resp, BatchRecallResponse)
        assert resp.total == 0
        assert resp.filtered == 0
        assert resp.memories == []

    def test_batch_forget_sends_correct_request(self, client, mock_responses):
        """batch_forget() DELETEs /v1/memories/forget/batch and returns BatchForgetResponse."""
        from dakera import BatchForgetRequest, BatchForgetResponse, BatchMemoryFilter

        mock_responses.add(
            responses.DELETE,
            "http://localhost:3000/v1/memories/forget/batch",
            json={"deleted_count": 5},
            status=200,
        )

        filt = BatchMemoryFilter(created_before=1700000000)
        req = BatchForgetRequest("qa", filter=filt)
        resp = client.batch_forget(req)

        assert isinstance(resp, BatchForgetResponse)
        assert resp.deleted_count == 5
        assert mock_responses.calls[0].request.method == "DELETE"
        assert "/v1/memories/forget/batch" in mock_responses.calls[0].request.url

    def test_batch_recall_request_to_dict(self):
        """BatchRecallRequest.to_dict() serializes correctly."""
        from dakera import BatchMemoryFilter, BatchRecallRequest

        filt = BatchMemoryFilter(tags=["qa"], min_importance=0.7, memory_type="episodic")
        req = BatchRecallRequest("agent-1", filter=filt, limit=25)
        d = req.to_dict()

        assert d["agent_id"] == "agent-1"
        assert d["limit"] == 25
        assert d["filter"]["tags"] == ["qa"]
        assert d["filter"]["min_importance"] == 0.7
        assert d["filter"]["memory_type"] == "episodic"

    def test_batch_forget_request_default_filter(self):
        """BatchForgetRequest gets a default empty BatchMemoryFilter when filter is None."""
        from dakera import BatchForgetRequest, BatchMemoryFilter

        req = BatchForgetRequest("agent-1")
        assert req.filter is not None
        assert isinstance(req.filter, BatchMemoryFilter)

    def test_batch_forget_response_from_dict(self):
        """BatchForgetResponse.from_dict() parses deleted_count correctly."""
        from dakera import BatchForgetResponse

        resp = BatchForgetResponse.from_dict({"deleted_count": 42})
        assert resp.deleted_count == 42

    def test_batch_recall_response_from_dict(self):
        """BatchRecallResponse.from_dict() parses total/filtered/memories."""
        from dakera import BatchRecallResponse

        resp = BatchRecallResponse.from_dict(
            {"memories": [], "total": 100, "filtered": 0}
        )
        assert resp.total == 100
        assert resp.filtered == 0
        assert resp.memories == []


class TestRateLimitHeaders:
    """Tests for OPS-1 RateLimitHeaders (v0.7.0)."""

    def test_from_headers_parses_all_fields(self):
        """RateLimitHeaders.from_headers() parses all known header names."""
        from dakera import RateLimitHeaders

        rl = RateLimitHeaders.from_headers(
            {
                "X-RateLimit-Limit": "1000",
                "X-RateLimit-Remaining": "750",
                "X-RateLimit-Reset": "1700000060",
                "X-Quota-Used": "500",
                "X-Quota-Limit": "10000",
            }
        )
        assert rl.limit == 1000
        assert rl.remaining == 750
        assert rl.reset == 1700000060
        assert rl.quota_used == 500
        assert rl.quota_limit == 10000

    def test_from_headers_missing_fields_are_none(self):
        """RateLimitHeaders.from_headers() returns None for missing headers."""
        from dakera import RateLimitHeaders

        rl = RateLimitHeaders.from_headers({})
        assert rl.limit is None
        assert rl.remaining is None
        assert rl.reset is None
        assert rl.quota_used is None
        assert rl.quota_limit is None

    def test_from_headers_ignores_non_numeric_values(self):
        """RateLimitHeaders.from_headers() returns None for non-numeric header values."""
        from dakera import RateLimitHeaders

        rl = RateLimitHeaders.from_headers({"X-RateLimit-Limit": "not-a-number"})
        assert rl.limit is None

    def test_last_rate_limit_headers_populated_after_request(self, client, mock_responses):
        """client.last_rate_limit_headers is populated from response headers."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"status": "healthy", "version": "0.7.0"},
            headers={
                "X-RateLimit-Limit": "500",
                "X-RateLimit-Remaining": "499",
                "X-RateLimit-Reset": "1700000120",
            },
            status=200,
        )

        client.health()
        rl = client.last_rate_limit_headers

        assert rl is not None
        assert rl.limit == 500
        assert rl.remaining == 499
        assert rl.reset == 1700000120

    def test_last_rate_limit_headers_initially_none(self):
        """client.last_rate_limit_headers is None before any request."""
        fresh_client = DakeraClient("http://localhost:3000")
        assert fresh_client.last_rate_limit_headers is None


class TestAutoPilotOperations:
    """Tests for AutoPilot management API (PILOT-1/2/3) — v0.7.2."""

    def test_autopilot_status(self, client, mock_responses):
        """autopilot_status() GETs /v1/admin/autopilot/status and returns config + stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/autopilot/status",
            json={
                "config": {
                    "enabled": True,
                    "dedup_threshold": 0.93,
                    "dedup_interval_hours": 6,
                    "consolidation_interval_hours": 12,
                },
                "last_dedup_at": 1700000000,
                "last_consolidation_at": 1700000100,
                "total_dedup_removed": 42,
                "total_consolidated": 10,
            },
            status=200,
        )

        result = client.autopilot_status()

        assert result["config"]["enabled"] is True
        assert result["config"]["dedup_threshold"] == 0.93
        assert result["total_dedup_removed"] == 42
        assert mock_responses.calls[0].request.method == "GET"
        assert "/v1/admin/autopilot/status" in mock_responses.calls[0].request.url

    def test_autopilot_update_config(self, client, mock_responses):
        """autopilot_update_config() PUTs /v1/admin/autopilot/config with updated fields."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/autopilot/config",
            json={
                "success": True,
                "config": {
                    "enabled": False,
                    "dedup_threshold": 0.90,
                    "dedup_interval_hours": 8,
                    "consolidation_interval_hours": 24,
                },
                "message": "AutoPilot config updated",
            },
            status=200,
        )

        result = client.autopilot_update_config(enabled=False, dedup_threshold=0.90)

        assert result["success"] is True
        assert result["config"]["enabled"] is False
        assert result["config"]["dedup_threshold"] == 0.90
        assert mock_responses.calls[0].request.method == "PUT"
        assert "/v1/admin/autopilot/config" in mock_responses.calls[0].request.url

    def test_autopilot_update_config_serializes_only_set_fields(self, client, mock_responses):
        """autopilot_update_config() omits unset optional fields from the request body."""
        import json as _json

        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/autopilot/config",
            json={"success": True, "config": {}, "message": "ok"},
            status=200,
        )

        client.autopilot_update_config(dedup_interval_hours=4)

        body = _json.loads(mock_responses.calls[0].request.body)
        assert "dedup_interval_hours" in body
        assert body["dedup_interval_hours"] == 4
        assert "enabled" not in body
        assert "dedup_threshold" not in body

    def test_autopilot_trigger_dedup(self, client, mock_responses):
        """autopilot_trigger('dedup') POSTs /v1/admin/autopilot/trigger with action=dedup."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/autopilot/trigger",
            json={
                "success": True,
                "action": "dedup",
                "dedup": {
                    "namespaces_processed": 3,
                    "memories_scanned": 500,
                    "duplicates_removed": 12,
                },
                "message": "Dedup cycle completed",
            },
            status=200,
        )

        result = client.autopilot_trigger("dedup")

        assert result["success"] is True
        assert result["action"] == "dedup"
        assert result["dedup"]["duplicates_removed"] == 12
        assert mock_responses.calls[0].request.method == "POST"
        assert "/v1/admin/autopilot/trigger" in mock_responses.calls[0].request.url

    def test_autopilot_trigger_all(self, client, mock_responses):
        """autopilot_trigger('all') sends action=all and returns dedup+consolidation."""
        import json as _json

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/autopilot/trigger",
            json={
                "success": True,
                "action": "all",
                "dedup": {
                    "namespaces_processed": 2,
                    "memories_scanned": 300,
                    "duplicates_removed": 5,
                },
                "consolidation": {
                    "namespaces_processed": 2,
                    "memories_scanned": 300,
                    "clusters_merged": 4,
                    "memories_consolidated": 8,
                },
                "message": "Full AutoPilot cycle completed",
            },
            status=200,
        )

        result = client.autopilot_trigger("all")

        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["action"] == "all"
        assert result["consolidation"]["clusters_merged"] == 4


class TestDecayOperations:
    """Tests for Decay Engine management API (DECAY-1/2) — v0.7.3."""

    def test_decay_config(self, client, mock_responses):
        """decay_config() GETs /v1/admin/decay/config and returns strategy/half-life/min."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/decay/config",
            json={
                "strategy": "exponential",
                "half_life_hours": 168.0,
                "min_importance": 0.05,
            },
            status=200,
        )

        result = client.decay_config()

        assert result["strategy"] == "exponential"
        assert result["half_life_hours"] == 168.0
        assert result["min_importance"] == 0.05
        assert mock_responses.calls[0].request.method == "GET"
        assert "/v1/admin/decay/config" in mock_responses.calls[0].request.url

    def test_decay_update_config(self, client, mock_responses):
        """decay_update_config() PUTs /v1/admin/decay/config with updated fields."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/decay/config",
            json={
                "success": True,
                "config": {
                    "strategy": "linear",
                    "half_life_hours": 72.0,
                    "min_importance": 0.1,
                },
                "message": "Decay config updated",
            },
            status=200,
        )

        result = client.decay_update_config(strategy="linear", half_life_hours=72.0)

        assert result["success"] is True
        assert result["config"]["strategy"] == "linear"
        assert result["config"]["half_life_hours"] == 72.0
        assert mock_responses.calls[0].request.method == "PUT"
        assert "/v1/admin/decay/config" in mock_responses.calls[0].request.url

    def test_decay_update_config_serializes_only_set_fields(self, client, mock_responses):
        """decay_update_config() omits unset optional fields from the request body."""
        import json as _json

        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/decay/config",
            json={"success": True, "config": {}, "message": "ok"},
            status=200,
        )

        client.decay_update_config(min_importance=0.02)

        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["min_importance"] == 0.02
        assert "strategy" not in body
        assert "half_life_hours" not in body

    def test_decay_stats(self, client, mock_responses):
        """decay_stats() GETs /v1/admin/decay/stats and returns counters + last-cycle snapshot."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/decay/stats",
            json={
                "total_decayed": 1024,
                "total_deleted": 128,
                "last_run_at": 1700000000,
                "cycles_run": 42,
                "last_cycle": {
                    "namespaces_processed": 5,
                    "memories_processed": 200,
                    "memories_decayed": 30,
                    "memories_deleted": 5,
                },
            },
            status=200,
        )

        result = client.decay_stats()

        assert result["total_decayed"] == 1024
        assert result["total_deleted"] == 128
        assert result["cycles_run"] == 42
        assert result["last_cycle"]["memories_decayed"] == 30
        assert mock_responses.calls[0].request.method == "GET"
        assert "/v1/admin/decay/stats" in mock_responses.calls[0].request.url

    def test_decay_stats_no_last_cycle(self, client, mock_responses):
        """decay_stats() handles the case where last_cycle is absent (never run)."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/decay/stats",
            json={
                "total_decayed": 0,
                "total_deleted": 0,
                "cycles_run": 0,
            },
            status=200,
        )

        result = client.decay_stats()

        assert result["cycles_run"] == 0
        assert "last_cycle" not in result


class TestStoreMemoryExpiresAt:
    """Tests for expires_at and ttl_seconds params on store_memory (DECAY-3) — v0.7.3."""

    def test_store_memory_with_expires_at(self, client, mock_responses):
        """store_memory() includes expires_at in request body when set."""
        import json as _json

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/agents/agent-1/memories",
            json={"id": "mem_1", "content": "test content"},
            status=200,
        )

        client.store_memory("agent-1", "test content", expires_at=1800000000)

        body = _json.loads(mock_responses.calls[0].request.body)
        assert "expires_at" in body
        assert body["expires_at"] == 1800000000

    def test_store_memory_without_expires_at(self, client, mock_responses):
        """store_memory() omits expires_at from request body when not set."""
        import json as _json

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/agents/agent-1/memories",
            json={"id": "mem_1", "content": "test content"},
            status=200,
        )

        client.store_memory("agent-1", "test content")

        body = _json.loads(mock_responses.calls[0].request.body)
        assert "expires_at" not in body

    def test_store_memory_with_ttl_seconds(self, client, mock_responses):
        """store_memory() includes ttl_seconds in request body when set."""
        import json as _json

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/agents/agent-1/memories",
            json={"id": "mem_1", "content": "ephemeral"},
            status=200,
        )

        client.store_memory("agent-1", "ephemeral", ttl_seconds=3600)

        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["ttl_seconds"] == 3600
        assert "expires_at" not in body


# ===========================================================================
# SSE Connected Event Parsing (DAK-720) — v0.8.3
# ===========================================================================


class TestDakeraEventConnectedParsing:
    """DakeraEvent and MemoryEvent correctly parse the 'connected' handshake event."""

    def test_dakera_event_from_dict_connected(self):
        """DakeraEvent.from_dict() with type='connected' populates type field."""
        data = {"type": "connected", "timestamp": 1700000000000}
        event = DakeraEvent.from_dict(data)
        assert event.type == "connected"

    def test_dakera_event_connected_has_no_namespace(self):
        """DakeraEvent connected event has no namespace or other fields set."""
        data = {"type": "connected", "timestamp": 1700000000000}
        event = DakeraEvent.from_dict(data)
        assert event.namespace is None
        assert event.dimension is None
        assert event.operation_id is None

    def test_memory_event_from_dict_connected_normalizes_type_key(self):
        """MemoryEvent.from_dict() maps 'type' key to 'event_type' for connected events."""
        data = {"type": "connected", "timestamp": 1700000000000}
        event = MemoryEvent.from_dict(data)
        assert event.event_type == "connected"

    def test_memory_event_connected_defaults_agent_id_to_empty(self):
        """MemoryEvent connected event has empty agent_id (no agent_id in payload)."""
        data = {"type": "connected", "timestamp": 1700000000000}
        event = MemoryEvent.from_dict(data)
        assert event.agent_id == ""

    def test_memory_event_connected_preserves_timestamp(self):
        """MemoryEvent connected event carries the timestamp from the SSE payload."""
        ts = 1774296453000
        data = {"type": "connected", "timestamp": ts}
        event = MemoryEvent.from_dict(data)
        assert event.timestamp == ts

    def test_memory_event_regular_event_unaffected(self):
        """MemoryEvent.from_dict() still works correctly for normal events."""
        data = {
            "event_type": "stored",
            "agent_id": "qa",
            "timestamp": 1700000000000,
            "memory_id": "mem_abc",
            "content": "test memory",
            "importance": 0.8,
        }
        event = MemoryEvent.from_dict(data)
        assert event.event_type == "stored"
        assert event.agent_id == "qa"
        assert event.memory_id == "mem_abc"
        assert event.importance == 0.8


# ===========================================================================
# AsyncDakeraClient.store_memory() parity (DAK-747) — v0.8.3
# ===========================================================================


class TestAsyncClientStoreMemoryParity:
    """AsyncDakeraClient.store_memory() now has full parity with sync client for
    ttl_seconds and expires_at (fixed in v0.8.3, DAK-747)."""

    async def test_store_memory_with_ttl_seconds_includes_field(self):
        """store_memory() sends ttl_seconds in request body when set."""
        client = AsyncDakeraClient("http://localhost:3000")
        captured: dict = {}

        async def fake_request(method, path, data=None, **kwargs):
            captured.update(data or {})
            return {"id": "mem_1", "content": "test"}

        with patch.object(client, "_request", side_effect=fake_request):
            await client.store_memory("agent-1", "test content", ttl_seconds=3600)

        assert captured.get("ttl_seconds") == 3600
        assert "expires_at" not in captured

    async def test_store_memory_with_expires_at_includes_field(self):
        """store_memory() sends expires_at in request body when set."""
        client = AsyncDakeraClient("http://localhost:3000")
        captured: dict = {}

        async def fake_request(method, path, data=None, **kwargs):
            captured.update(data or {})
            return {"id": "mem_1", "content": "test"}

        with patch.object(client, "_request", side_effect=fake_request):
            await client.store_memory("agent-1", "test content", expires_at=1800000000)

        assert captured.get("expires_at") == 1800000000
        assert "ttl_seconds" not in captured

    async def test_store_memory_with_both_fields(self):
        """store_memory() sends both ttl_seconds and expires_at when both are set."""
        client = AsyncDakeraClient("http://localhost:3000")
        captured: dict = {}

        async def fake_request(method, path, data=None, **kwargs):
            captured.update(data or {})
            return {"id": "mem_1", "content": "test"}

        with patch.object(client, "_request", side_effect=fake_request):
            await client.store_memory(
                "agent-1", "test content", ttl_seconds=3600, expires_at=1800000000
            )

        assert captured.get("ttl_seconds") == 3600
        assert captured.get("expires_at") == 1800000000

    async def test_store_memory_omits_expiry_fields_when_not_set(self):
        """store_memory() does not include ttl_seconds or expires_at when neither is set."""
        client = AsyncDakeraClient("http://localhost:3000")
        captured: dict = {}

        async def fake_request(method, path, data=None, **kwargs):
            captured.update(data or {})
            return {"id": "mem_1", "content": "test"}

        with patch.object(client, "_request", side_effect=fake_request):
            await client.store_memory("agent-1", "plain content")

        assert "ttl_seconds" not in captured
        assert "expires_at" not in captured

    async def test_store_memory_posts_to_correct_endpoint(self):
        """store_memory() calls POST /v1/agents/{agent_id}/memories."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, data=None, **kwargs):
            calls.append((method, path))
            return {"id": "mem_1"}

        with patch.object(client, "_request", side_effect=fake_request):
            await client.store_memory("my-agent", "content")

        assert calls == [("POST", "/v1/agents/my-agent/memories")]

# ===========================================================================
# Memory Knowledge Graph Tests (CE-5 / SDK-9)
# ===========================================================================

GRAPH_RESPONSE = {
    "root_id": "mem-abc",
    "depth": 2,
    "nodes": [
        {"memory_id": "mem-abc", "content_preview": "Root memory", "importance": 0.9, "depth": 0},
        {"memory_id": "mem-def", "content_preview": "Related memory", "importance": 0.7,
         "depth": 1},
        {"memory_id": "mem-ghi", "content_preview": "Linked memory", "importance": 0.5, "depth": 2},
    ],
    "edges": [
        {
            "id": "edge-1",
            "source_id": "mem-abc",
            "target_id": "mem-def",
            "edge_type": "related_to",
            "weight": 0.92,
            "created_at": 1774000000,
        },
        {
            "id": "edge-2",
            "source_id": "mem-def",
            "target_id": "mem-ghi",
            "edge_type": "linked_by",
            "weight": 1.0,
            "created_at": 1774001000,
        },
    ],
}

PATH_RESPONSE = {
    "source_id": "mem-abc",
    "target_id": "mem-ghi",
    "path": ["mem-abc", "mem-def", "mem-ghi"],
    "hops": 2,
    "edges": [
        {
            "id": "edge-1",
            "source_id": "mem-abc",
            "target_id": "mem-def",
            "edge_type": "related_to",
            "weight": 0.92,
            "created_at": 1774000000,
        },
        {
            "id": "edge-2",
            "source_id": "mem-def",
            "target_id": "mem-ghi",
            "edge_type": "linked_by",
            "weight": 1.0,
            "created_at": 1774001000,
        },
    ],
}

LINK_RESPONSE = {
    "edge": {
        "id": "edge-new",
        "source_id": "mem-abc",
        "target_id": "mem-xyz",
        "edge_type": "linked_by",
        "weight": 1.0,
        "created_at": 1774002000,
    }
}

EXPORT_RESPONSE = {
    "agent_id": "test-agent",
    "format": "json",
    "data": '{"nodes": [], "edges": []}',
    "node_count": 10,
    "edge_count": 7,
}


class TestMemoryGraphSyncClient:
    """Tests for Memory Knowledge Graph sync client methods (CE-5 / SDK-9)."""

    def test_memory_graph_default_depth(self, client, mock_responses):
        """memory_graph() defaults to depth=1."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-abc/graph",
            json=GRAPH_RESPONSE,
            status=200,
        )
        result = client.memory_graph("mem-abc")
        assert result.root_id == "mem-abc"
        assert result.depth == 2
        assert len(result.nodes) == 3
        assert len(result.edges) == 2
        assert "depth=1" in mock_responses.calls[0].request.url

    def test_memory_graph_custom_depth_and_types(self, client, mock_responses):
        """memory_graph() passes depth and types query params."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-abc/graph",
            json=GRAPH_RESPONSE,
            status=200,
        )
        result = client.memory_graph("mem-abc", depth=2, types=["related_to", "linked_by"])
        assert result.root_id == "mem-abc"
        url = mock_responses.calls[0].request.url
        assert "depth=2" in url
        assert "related_to" in url

    def test_memory_graph_edge_types_parsed(self, client, mock_responses):
        """memory_graph() returns edges with correct EdgeType enum values."""
        from dakera import EdgeType
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-abc/graph",
            json=GRAPH_RESPONSE,
            status=200,
        )
        result = client.memory_graph("mem-abc")
        assert result.edges[0].edge_type == EdgeType.RELATED_TO
        assert result.edges[1].edge_type == EdgeType.LINKED_BY

    def test_memory_graph_node_depth_populated(self, client, mock_responses):
        """GraphNode.depth is populated correctly."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-abc/graph",
            json=GRAPH_RESPONSE,
            status=200,
        )
        result = client.memory_graph("mem-abc")
        depths = {n.memory_id: n.depth for n in result.nodes}
        assert depths["mem-abc"] == 0
        assert depths["mem-def"] == 1
        assert depths["mem-ghi"] == 2

    def test_memory_path(self, client, mock_responses):
        """memory_path() returns shortest path between two memories."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-abc/path",
            json=PATH_RESPONSE,
            status=200,
        )
        result = client.memory_path("mem-abc", "mem-ghi")
        assert result.source_id == "mem-abc"
        assert result.target_id == "mem-ghi"
        assert result.path == ["mem-abc", "mem-def", "mem-ghi"]
        assert result.hops == 2
        assert len(result.edges) == 2
        assert "target=mem-ghi" in mock_responses.calls[0].request.url

    def test_memory_link_default_edge_type(self, client, mock_responses):
        """memory_link() defaults to EdgeType.LINKED_BY."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/mem-abc/links",
            json=LINK_RESPONSE,
            status=200,
        )
        result = client.memory_link("mem-abc", "mem-xyz")
        assert result.edge.id == "edge-new"
        assert result.edge.edge_type.value == "linked_by"
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["target_id"] == "mem-xyz"
        assert body["edge_type"] == "linked_by"

    def test_memory_link_custom_edge_type_string(self, client, mock_responses):
        """memory_link() accepts a plain string edge_type."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/mem-abc/links",
            json=LINK_RESPONSE,
            status=200,
        )
        result = client.memory_link("mem-abc", "mem-xyz", edge_type="linked_by")
        assert result.edge.id == "edge-new"

    def test_agent_graph_export_json(self, client, mock_responses):
        """agent_graph_export() defaults to json format."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/test-agent/graph/export",
            json=EXPORT_RESPONSE,
            status=200,
        )
        result = client.agent_graph_export("test-agent")
        assert result.agent_id == "test-agent"
        assert result.format == "json"
        assert result.node_count == 10
        assert result.edge_count == 7
        assert "format=json" in mock_responses.calls[0].request.url

    def test_agent_graph_export_graphml(self, client, mock_responses):
        """agent_graph_export() passes format=graphml."""
        graphml_resp = {**EXPORT_RESPONSE, "format": "graphml", "data": "<graphml/>"}
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/test-agent/graph/export",
            json=graphml_resp,
            status=200,
        )
        result = client.agent_graph_export("test-agent", format="graphml")
        assert result.format == "graphml"
        assert "format=graphml" in mock_responses.calls[0].request.url


@pytest.mark.asyncio
class TestMemoryGraphAsyncClient:
    """Tests for Memory Knowledge Graph async client methods (CE-5 / SDK-9)."""

    async def test_memory_graph_calls_correct_endpoint(self):
        """memory_graph() calls GET /v1/memories/{id}/graph."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, params=None, **kwargs):
            calls.append((method, path, params or {}))
            return GRAPH_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.memory_graph("mem-abc", depth=2)

        assert calls[0][0] == "GET"
        assert calls[0][1] == "/v1/memories/mem-abc/graph"
        assert calls[0][2]["depth"] == 2
        assert result.root_id == "mem-abc"
        assert len(result.nodes) == 3

    async def test_memory_graph_types_filter(self):
        """memory_graph() includes types param when specified."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, params=None, **kwargs):
            calls.append((method, path, params or {}))
            return GRAPH_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            await client.memory_graph("mem-abc", depth=1, types=["related_to"])

        assert calls[0][2]["types"] == "related_to"

    async def test_memory_graph_no_types_filter(self):
        """memory_graph() omits types param when not specified."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, params=None, **kwargs):
            calls.append((method, path, params or {}))
            return GRAPH_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            await client.memory_graph("mem-abc")

        assert "types" not in calls[0][2]

    async def test_memory_path_calls_correct_endpoint(self):
        """memory_path() calls GET /v1/memories/{id}/path?target={id}."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, params=None, **kwargs):
            calls.append((method, path, params or {}))
            return PATH_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.memory_path("mem-abc", "mem-ghi")

        assert calls[0][0] == "GET"
        assert calls[0][1] == "/v1/memories/mem-abc/path"
        assert calls[0][2]["target"] == "mem-ghi"
        assert result.hops == 2
        assert result.path == ["mem-abc", "mem-def", "mem-ghi"]

    async def test_memory_link_default_edge_type(self):
        """memory_link() defaults to linked_by and POSTs to /links."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, data=None, **kwargs):
            calls.append((method, path, data or {}))
            return LINK_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.memory_link("mem-abc", "mem-xyz")

        assert calls[0][0] == "POST"
        assert calls[0][1] == "/v1/memories/mem-abc/links"
        assert calls[0][2]["target_id"] == "mem-xyz"
        assert calls[0][2]["edge_type"] == "linked_by"
        assert result.edge.id == "edge-new"

    async def test_memory_link_enum_edge_type(self):
        """memory_link() accepts EdgeType enum values."""
        from dakera import EdgeType
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, data=None, **kwargs):
            calls.append((method, path, data or {}))
            return LINK_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            await client.memory_link("mem-abc", "mem-xyz", edge_type=EdgeType.LINKED_BY)

        assert calls[0][2]["edge_type"] == "linked_by"

    async def test_agent_graph_export(self):
        """agent_graph_export() calls GET /v1/agents/{id}/graph/export."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, params=None, **kwargs):
            calls.append((method, path, params or {}))
            return EXPORT_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.agent_graph_export("my-agent", format="graphml")

        assert calls[0][0] == "GET"
        assert calls[0][1] == "/v1/agents/my-agent/graph/export"
        assert calls[0][2]["format"] == "graphml"
        assert result.agent_id == "test-agent"
        assert result.node_count == 10

    async def test_agent_graph_export_default_format(self):
        """agent_graph_export() defaults to json format."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, params=None, **kwargs):
            calls.append((method, path, params or {}))
            return EXPORT_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            await client.agent_graph_export("my-agent")

        assert calls[0][2]["format"] == "json"


class TestGraphModels:
    """Unit tests for graph model dataclasses and EdgeType enum."""

    def test_edge_type_values(self):
        """EdgeType enum has all expected values."""
        from dakera import EdgeType
        assert EdgeType.RELATED_TO.value == "related_to"
        assert EdgeType.SHARES_ENTITY.value == "shares_entity"
        assert EdgeType.PRECEDES.value == "precedes"
        assert EdgeType.LINKED_BY.value == "linked_by"

    def test_edge_type_from_string(self):
        """EdgeType can be constructed from string value."""
        from dakera import EdgeType
        assert EdgeType("related_to") == EdgeType.RELATED_TO

    def test_graph_edge_from_dict(self):
        """GraphEdge.from_dict() parses all fields."""
        from dakera import EdgeType, GraphEdge
        edge = GraphEdge.from_dict({
            "id": "e1",
            "source_id": "mem-a",
            "target_id": "mem-b",
            "edge_type": "related_to",
            "weight": 0.88,
            "created_at": 1774000000,
        })
        assert edge.id == "e1"
        assert edge.edge_type == EdgeType.RELATED_TO
        assert edge.weight == 0.88

    def test_graph_node_from_dict(self):
        """GraphNode.from_dict() parses all fields."""
        from dakera import GraphNode
        node = GraphNode.from_dict({
            "memory_id": "mem-x",
            "content_preview": "test content",
            "importance": 0.75,
            "depth": 1,
        })
        assert node.memory_id == "mem-x"
        assert node.depth == 1

    def test_memory_graph_from_dict(self):
        """MemoryGraph.from_dict() parses nodes and edges."""
        from dakera import MemoryGraph
        result = MemoryGraph.from_dict(GRAPH_RESPONSE)
        assert result.root_id == "mem-abc"
        assert len(result.nodes) == 3
        assert len(result.edges) == 2

    def test_graph_path_hops_computed_from_path(self):
        """GraphPath.hops falls back to len(path)-1 if not in response."""
        from dakera import GraphPath
        data = {**PATH_RESPONSE}
        del data["hops"]
        path = GraphPath.from_dict(data)
        assert path.hops == 2  # len(["mem-abc","mem-def","mem-ghi"]) - 1

    def test_graph_export_from_dict(self):
        """GraphExport.from_dict() parses all fields."""
        from dakera import GraphExport
        export = GraphExport.from_dict(EXPORT_RESPONSE)
        assert export.agent_id == "test-agent"
        assert export.format == "json"
        assert export.node_count == 10
        assert export.edge_count == 7


# ===========================================================================
# agents_subscribe (SDK-10)
# ===========================================================================


def make_memory_event(event_type: str, agent_id: str, tags=None, memory_id=None) -> MemoryEvent:
    return MemoryEvent.from_dict({
        "event_type": event_type,
        "agent_id": agent_id,
        "timestamp": 1774533000000,
        **({"memory_id": memory_id} if memory_id else {}),
        **({"tags": tags} if tags else {}),
    })


@pytest.mark.asyncio
class TestAgentsSubscribe:
    """agents_subscribe() filters memory events by agent_id and tags."""

    async def test_filters_by_agent_id(self):
        """Only events for the given agent_id are yielded."""
        events = [
            make_memory_event("stored", "agent-a", memory_id="m1"),
            make_memory_event("stored", "agent-b", memory_id="m2"),
            make_memory_event("recalled", "agent-a", memory_id="m3"),
        ]

        async def fake_stream():
            for e in events:
                yield e

        client = AsyncDakeraClient("http://localhost:3000")
        with patch.object(client, "stream_memory_events", return_value=fake_stream()):
            collected = []
            async for ev in client.agents_subscribe("agent-a", reconnect=False):
                collected.append(ev)

        assert len(collected) == 2
        assert all(e.agent_id == "agent-a" for e in collected)
        assert {e.memory_id for e in collected} == {"m1", "m3"}

    async def test_skips_connected_handshake(self):
        """The 'connected' handshake event is never yielded."""
        events = [
            MemoryEvent.from_dict({"type": "connected", "timestamp": 1774533000000}),
            make_memory_event("stored", "bot", memory_id="m1"),
        ]

        async def fake_stream():
            for e in events:
                yield e

        client = AsyncDakeraClient("http://localhost:3000")
        with patch.object(client, "stream_memory_events", return_value=fake_stream()):
            collected = []
            async for ev in client.agents_subscribe("bot", reconnect=False):
                collected.append(ev)

        assert len(collected) == 1
        assert collected[0].memory_id == "m1"

    async def test_tag_filter(self):
        """Only events with at least one matching tag are yielded."""
        events = [
            make_memory_event("stored", "bot", tags=["important", "work"], memory_id="m1"),
            make_memory_event("stored", "bot", tags=["trivial"], memory_id="m2"),
            make_memory_event("stored", "bot", tags=["important"], memory_id="m3"),
        ]

        async def fake_stream():
            for e in events:
                yield e

        client = AsyncDakeraClient("http://localhost:3000")
        with patch.object(client, "stream_memory_events", return_value=fake_stream()):
            collected = []
            async for ev in client.agents_subscribe("bot", tags=["important"], reconnect=False):
                collected.append(ev)

        assert {e.memory_id for e in collected} == {"m1", "m3"}

    async def test_no_tag_filter_yields_all_agent_events(self):
        """When tags=None, all events for the agent are yielded."""
        events = [
            make_memory_event("stored", "bot", tags=["x"], memory_id="m1"),
            make_memory_event("forgotten", "bot", memory_id="m2"),
        ]

        async def fake_stream():
            for e in events:
                yield e

        client = AsyncDakeraClient("http://localhost:3000")
        with patch.object(client, "stream_memory_events", return_value=fake_stream()):
            collected = []
            async for ev in client.agents_subscribe("bot", reconnect=False):
                collected.append(ev)

        assert len(collected) == 2

    async def test_reconnect_false_raises_on_error(self):
        """With reconnect=False an exception propagates to the caller."""
        async def failing_stream():
            raise ConnectionError("stream dropped")
            yield  # make it a generator

        client = AsyncDakeraClient("http://localhost:3000")
        mock_stream = patch.object(
            client, "stream_memory_events", return_value=failing_stream()
        )
        with mock_stream, pytest.raises(ConnectionError, match="stream dropped"):
            async for _ in client.agents_subscribe("bot", reconnect=False):
                pass

    async def test_reconnect_true_retries_on_error(self):
        """With reconnect=True the generator reconnects after a stream error."""
        call_count = 0

        async def flaky_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("first attempt fails")
            yield make_memory_event("stored", "bot", memory_id="m1")

        client = AsyncDakeraClient("http://localhost:3000")
        mock_stream = patch.object(
            client, "stream_memory_events", side_effect=flaky_stream
        )
        with mock_stream, patch("asyncio.sleep"):
            collected = []
            async for ev in client.agents_subscribe("bot", reconnect=True, reconnect_delay=0):
                collected.append(ev)
                break  # stop after first successful event

        assert call_count == 2
        assert collected[0].memory_id == "m1"
