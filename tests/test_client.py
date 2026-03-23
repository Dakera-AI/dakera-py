"""Tests for Dakera client."""

import pytest
import responses

from dakera import (
    DakeraClient,
    Document,
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
