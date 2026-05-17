"""Tests for Dakera client."""

import json
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
            vector_weight=0.5,
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


class TestCog2AssociativeRecall:
    """Tests for COG-2 associative recall (include_associated)."""

    def test_recall_include_associated_sends_flag(self, client, mock_responses):
        """recall(include_associated=True) sends include_associated in request body."""
        import responses as resp_lib

        mock_responses.add(
            resp_lib.POST,
            "http://localhost:3000/v1/memory/recall",
            json={
                "memories": [
                    {
                        "id": "mem_1",
                        "content": "primary memory",
                        "memory_type": "episodic",
                        "importance": 0.8,
                        "score": 0.95,
                    }
                ],
                "associated_memories": [
                    {
                        "id": "mem_2",
                        "content": "associated memory",
                        "memory_type": "semantic",
                        "importance": 0.6,
                        "score": 0.75,
                    }
                ],
            },
            status=200,
        )
        from dakera import RecallResponse

        result = client.recall("agent-1", "test query", include_associated=True)
        assert isinstance(result, RecallResponse)
        assert len(result.memories) == 1
        assert result.associated_memories is not None
        assert len(result.associated_memories) == 1
        assert result.associated_memories[0].id == "mem_2"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["include_associated"] is True

    def test_recall_without_include_associated_omits_flag(self, client, mock_responses):
        """recall() without include_associated does not send the flag and returns RecallResponse."""
        import responses as resp_lib

        mock_responses.add(
            resp_lib.POST,
            "http://localhost:3000/v1/memory/recall",
            json={
                "memories": [
                    {
                        "id": "mem_1",
                        "content": "primary memory",
                        "memory_type": "episodic",
                        "importance": 0.8,
                        "score": 0.95,
                    }
                ]
            },
            status=200,
        )
        from dakera import RecallResponse

        result = client.recall("agent-1", "test query")
        assert isinstance(result, RecallResponse)
        assert len(result.memories) == 1
        assert result.associated_memories is None
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert "include_associated" not in req_body

    def test_recall_associated_memories_cap(self, client, mock_responses):
        """recall() with associated_memories_cap sends cap in request body."""
        import responses as resp_lib

        mock_responses.add(
            resp_lib.POST,
            "http://localhost:3000/v1/memory/recall",
            json={"memories": [], "associated_memories": []},
            status=200,
        )
        client.recall("agent-1", "test", include_associated=True, associated_memories_cap=3)
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["associated_memories_cap"] == 3


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
            "http://localhost:3000/v1/memory/store",
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
            "http://localhost:3000/v1/memory/store",
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
            "http://localhost:3000/v1/memory/store",
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
        """store_memory() calls POST /v1/memory/store."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, data=None, **kwargs):
            calls.append((method, path))
            return {"id": "mem_1"}

        with patch.object(client, "_request", side_effect=fake_request):
            await client.store_memory("my-agent", "content")

        assert calls == [("POST", "/v1/memory/store")]

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


# ===========================================================================
# Entity Extraction (CE-4 / GLiNER)
# ===========================================================================

EXTRACT_RESPONSE = {
    "entities": [
        {"entity_type": "person", "value": "Alice", "score": 0.97},
        {"entity_type": "location", "value": "Paris", "score": 0.91},
    ],
}

MEMORY_ENTITIES_RESPONSE = {
    "memory_id": "mem-001",
    "entities": [
        {"entity_type": "org", "value": "Dakera", "score": 0.88},
    ],
}


class TestEntityExtractionSyncClient:
    """Tests for CE-4 entity extraction sync client methods."""

    def test_configure_namespace_ner_enable(self, client, mock_responses):
        """configure_namespace_ner() sends PATCH to correct endpoint."""
        mock_responses.add(
            responses.PATCH,
            "http://localhost:3000/v1/namespaces/my-ns/config",
            json={"extract_entities": True, "entity_types": ["person", "org"]},
            status=200,
        )
        result = client.configure_namespace_ner(
            "my-ns", extract_entities=True, entity_types=["person", "org"]
        )
        assert result["extract_entities"] is True
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["extract_entities"] is True
        assert body["entity_types"] == ["person", "org"]

    def test_configure_namespace_ner_disable(self, client, mock_responses):
        """configure_namespace_ner() with extract_entities=False omits entity_types."""
        mock_responses.add(
            responses.PATCH,
            "http://localhost:3000/v1/namespaces/my-ns/config",
            json={"extract_entities": False},
            status=200,
        )
        result = client.configure_namespace_ner("my-ns", extract_entities=False)
        assert result["extract_entities"] is False
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["extract_entities"] is False
        assert "entity_types" not in body

    def test_extract_entities_returns_response(self, client, mock_responses):
        """extract_entities() POSTs text and returns EntityExtractionResponse."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/extract",
            json=EXTRACT_RESPONSE,
            status=200,
        )
        result = client.extract_entities("Alice lives in Paris.")
        assert len(result.entities) == 2
        assert result.entities[0].entity_type == "person"
        assert result.entities[0].value == "Alice"
        assert result.entities[0].score == pytest.approx(0.97)
        assert result.entities[1].entity_type == "location"
        assert result.entities[1].value == "Paris"
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["content"] == "Alice lives in Paris."
        assert "entity_types" not in body

    def test_extract_entities_with_entity_types(self, client, mock_responses):
        """extract_entities() sends entity_types when provided."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/extract",
            json=EXTRACT_RESPONSE,
            status=200,
        )
        client.extract_entities("Alice lives in Paris.", entity_types=["person", "location"])
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["entity_types"] == ["person", "location"]

    def test_memory_entities_returns_response(self, client, mock_responses):
        """memory_entities() GETs entity tags for a stored memory."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memory/entities/mem-001",
            json=MEMORY_ENTITIES_RESPONSE,
            status=200,
        )
        result = client.memory_entities("mem-001")
        assert result.memory_id == "mem-001"
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "org"
        assert result.entities[0].value == "Dakera"
        assert result.entities[0].score == pytest.approx(0.88)


@pytest.mark.asyncio
class TestEntityExtractionAsyncClient:
    """Tests for CE-4 entity extraction async client methods."""

    async def test_configure_namespace_ner(self):
        """configure_namespace_ner() calls PATCH /v1/namespaces/:ns/config."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, data=None, **kwargs):
            calls.append((method, path, data or {}))
            return {"extract_entities": True, "entity_types": ["person"]}

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.configure_namespace_ner(
                "my-ns", extract_entities=True, entity_types=["person"]
            )

        assert calls[0][0] == "PATCH"
        assert calls[0][1] == "/v1/namespaces/my-ns/config"
        assert calls[0][2]["extract_entities"] is True
        assert calls[0][2]["entity_types"] == ["person"]
        assert result["extract_entities"] is True

    async def test_configure_namespace_ner_no_entity_types(self):
        """configure_namespace_ner() omits entity_types when None."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, data=None, **kwargs):
            calls.append((method, path, data or {}))
            return {"extract_entities": False}

        with patch.object(client, "_request", side_effect=fake_request):
            await client.configure_namespace_ner("my-ns", extract_entities=False)

        assert "entity_types" not in calls[0][2]

    async def test_extract_entities(self):
        """extract_entities() calls POST /v1/memories/extract."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, data=None, **kwargs):
            calls.append((method, path, data or {}))
            return EXTRACT_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.extract_entities("Alice lives in Paris.")

        assert calls[0][0] == "POST"
        assert calls[0][1] == "/v1/memories/extract"
        assert calls[0][2]["text"] == "Alice lives in Paris."
        assert len(result.entities) == 2
        assert result.entities[0].value == "Alice"

    async def test_extract_entities_with_types(self):
        """extract_entities() forwards entity_types to the server."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, data=None, **kwargs):
            calls.append((method, path, data or {}))
            return EXTRACT_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            await client.extract_entities("Alice.", entity_types=["person"])

        assert calls[0][2]["entity_types"] == ["person"]

    async def test_memory_entities(self):
        """memory_entities() calls GET /v1/memory/entities/:id."""
        client = AsyncDakeraClient("http://localhost:3000")
        calls: list = []

        async def fake_request(method, path, **kwargs):
            calls.append((method, path))
            return MEMORY_ENTITIES_RESPONSE

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.memory_entities("mem-001")

        assert calls[0][0] == "GET"
        assert calls[0][1] == "/v1/memory/entities/mem-001"
        assert result.memory_id == "mem-001"
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "org"


class TestEntityExtractionModels:
    """Unit tests for CE-4 entity extraction model dataclasses."""

    def test_extracted_entity_from_dict(self):
        """ExtractedEntity.from_dict() parses all fields."""
        from dakera import ExtractedEntity
        entity = ExtractedEntity.from_dict(
            {"entity_type": "person", "value": "Bob", "score": 0.95}
        )
        assert entity.entity_type == "person"
        assert entity.value == "Bob"
        assert entity.score == pytest.approx(0.95)

    def test_extracted_entity_score_defaults_to_zero(self):
        """ExtractedEntity.from_dict() defaults score to 0.0 when missing."""
        from dakera import ExtractedEntity
        entity = ExtractedEntity.from_dict({"entity_type": "org", "value": "Acme"})
        assert entity.score == 0.0

    def test_entity_extraction_response_from_dict(self):
        """EntityExtractionResponse.from_dict() builds entity list."""
        from dakera import EntityExtractionResponse
        resp = EntityExtractionResponse.from_dict(EXTRACT_RESPONSE)
        assert len(resp.entities) == 2
        assert resp.entities[0].entity_type == "person"

    def test_entity_extraction_response_empty(self):
        """EntityExtractionResponse.from_dict() handles empty entities list."""
        from dakera import EntityExtractionResponse
        resp = EntityExtractionResponse.from_dict({"entities": []})
        assert resp.entities == []

    def test_memory_entities_response_from_dict(self):
        """MemoryEntitiesResponse.from_dict() parses memory_id and entities."""
        from dakera import MemoryEntitiesResponse
        resp = MemoryEntitiesResponse.from_dict(MEMORY_ENTITIES_RESPONSE)
        assert resp.memory_id == "mem-001"
        assert len(resp.entities) == 1
        assert resp.entities[0].value == "Dakera"

    def test_namespace_ner_config_to_dict_without_types(self):
        """NamespaceNerConfig.to_dict() omits entity_types when None."""
        from dakera import NamespaceNerConfig
        cfg = NamespaceNerConfig(extract_entities=True)
        d = cfg.to_dict()
        assert d["extract_entities"] is True
        assert "entity_types" not in d

    def test_namespace_ner_config_to_dict_with_types(self):
        """NamespaceNerConfig.to_dict() includes entity_types when set."""
        from dakera import NamespaceNerConfig
        cfg = NamespaceNerConfig(extract_entities=True, entity_types=["person", "date"])
        d = cfg.to_dict()
        assert d["entity_types"] == ["person", "date"]


# ===========================================================================
# INT-1 Memory Feedback Loop test fixtures
# ===========================================================================

FEEDBACK_RESPONSE = {
    "memory_id": "mem-abc",
    "new_importance": 0.92,
    "signal": "upvote",
}

# SEC-1 fixtures
NAMESPACE_KEY_INFO = {
    "key_id": "key-abc",
    "name": "ci-runner",
    "namespace": "prod-ns",
    "created_at": 1774000000,
    "active": True,
    "expires_at": None,
}

CREATE_NAMESPACE_KEY_RESPONSE = {
    "key_id": "key-abc",
    "key": "dak_live_xxxxxxxxxxxx",
    "name": "ci-runner",
    "namespace": "prod-ns",
    "created_at": 1774000000,
    "expires_at": None,
    "warning": "Save this key — it will not be shown again.",
}

LIST_NAMESPACE_KEYS_RESPONSE = {
    "namespace": "prod-ns",
    "keys": [NAMESPACE_KEY_INFO],
    "total": 1,
}

NAMESPACE_KEY_USAGE_RESPONSE = {
    "key_id": "key-abc",
    "namespace": "prod-ns",
    "total_requests": 1000,
    "successful_requests": 980,
    "failed_requests": 20,
    "bytes_transferred": 512000,
    "avg_latency_ms": 12.4,
}

FEEDBACK_HISTORY_RESPONSE = {
    "memory_id": "mem-abc",
    "entries": [
        {
            "signal": "upvote",
            "timestamp": 1774000000,
            "old_importance": 0.5,
            "new_importance": 0.575,
        },
        {
            "signal": "downvote",
            "timestamp": 1774001000,
            "old_importance": 0.575,
            "new_importance": 0.489,
        },
    ],
}

AGENT_FEEDBACK_SUMMARY = {
    "agent_id": "agent-1",
    "upvotes": 42,
    "downvotes": 7,
    "flags": 2,
    "total_feedback": 51,
    "health_score": 0.78,
}

FEEDBACK_HEALTH_RESPONSE = {
    "agent_id": "agent-1",
    "health_score": 0.78,
    "memory_count": 120,
    "avg_importance": 0.72,
}


class TestFeedbackLoopSyncClient:
    """Tests for INT-1 feedback loop sync client methods."""

    def test_feedback_memory_upvote(self, client, mock_responses):
        """feedback_memory() POSTs to /v1/memories/:id/feedback and returns FeedbackResponse."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/mem-abc/feedback",
            json=FEEDBACK_RESPONSE,
            status=200,
        )
        from dakera import FeedbackSignal
        result = client.feedback_memory("mem-abc", "agent-1", FeedbackSignal.UPVOTE)
        assert result.memory_id == "mem-abc"
        assert result.new_importance == pytest.approx(0.92)
        assert result.signal == FeedbackSignal.UPVOTE
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["agent_id"] == "agent-1"
        assert body["signal"] == "upvote"

    def test_feedback_memory_accepts_string_signal(self, client, mock_responses):
        """feedback_memory() accepts a raw string for signal."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/mem-abc/feedback",
            json=FEEDBACK_RESPONSE,
            status=200,
        )
        result = client.feedback_memory("mem-abc", "agent-1", "upvote")
        assert result.memory_id == "mem-abc"

    def test_get_memory_feedback_history(self, client, mock_responses):
        """get_memory_feedback_history() GETs /v1/memories/:id/feedback."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-abc/feedback",
            json=FEEDBACK_HISTORY_RESPONSE,
            status=200,
        )
        result = client.get_memory_feedback_history("mem-abc")
        assert result.memory_id == "mem-abc"
        assert len(result.entries) == 2
        from dakera import FeedbackSignal
        assert result.entries[0].signal == FeedbackSignal.UPVOTE
        assert result.entries[1].signal == FeedbackSignal.DOWNVOTE

    def test_get_agent_feedback_summary(self, client, mock_responses):
        """get_agent_feedback_summary() GETs /v1/agents/:id/feedback/summary."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/agent-1/feedback/summary",
            json=AGENT_FEEDBACK_SUMMARY,
            status=200,
        )
        result = client.get_agent_feedback_summary("agent-1")
        assert result.agent_id == "agent-1"
        assert result.upvotes == 42
        assert result.downvotes == 7
        assert result.flags == 2
        assert result.total_feedback == 51
        assert result.health_score == pytest.approx(0.78)

    def test_patch_memory_importance(self, client, mock_responses):
        """patch_memory_importance() PATCHes /v1/memories/:id/importance."""
        mock_responses.add(
            responses.PATCH,
            "http://localhost:3000/v1/memories/mem-abc/importance",
            json=FEEDBACK_RESPONSE,
            status=200,
        )
        result = client.patch_memory_importance("mem-abc", "agent-1", 0.92)
        assert result.memory_id == "mem-abc"
        assert result.new_importance == pytest.approx(0.92)
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["agent_id"] == "agent-1"
        assert body["importance"] == pytest.approx(0.92)

    def test_get_feedback_health(self, client, mock_responses):
        """get_feedback_health() GETs /v1/feedback/health?agent_id=..."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/feedback/health",
            json=FEEDBACK_HEALTH_RESPONSE,
            status=200,
        )
        result = client.get_feedback_health("agent-1")
        assert result.agent_id == "agent-1"
        assert result.health_score == pytest.approx(0.78)
        assert result.memory_count == 120
        assert result.avg_importance == pytest.approx(0.72)
        assert "agent_id=agent-1" in mock_responses.calls[0].request.url


class TestFeedbackLoopAsyncClient:
    """Tests for INT-1 feedback loop async client methods."""

    @pytest.mark.asyncio
    async def test_feedback_memory_async(self):
        """AsyncDakeraClient.feedback_memory() calls correct endpoint."""
        import httpx

        from dakera import FeedbackSignal
        with patch("httpx.AsyncClient.request") as mock_req:
            mock_req.return_value = httpx.Response(200, json=FEEDBACK_RESPONSE)
            async with AsyncDakeraClient("http://localhost:3000") as client:
                result = await client.feedback_memory("mem-abc", "agent-1", FeedbackSignal.UPVOTE)
        assert result.memory_id == "mem-abc"
        assert result.signal == FeedbackSignal.UPVOTE
        _, kwargs = mock_req.call_args
        assert kwargs["url"].endswith("/v1/memories/mem-abc/feedback")

    @pytest.mark.asyncio
    async def test_get_memory_feedback_history_async(self):
        """AsyncDakeraClient.get_memory_feedback_history() GETs history."""
        import httpx
        with patch("httpx.AsyncClient.request") as mock_req:
            mock_req.return_value = httpx.Response(200, json=FEEDBACK_HISTORY_RESPONSE)
            async with AsyncDakeraClient("http://localhost:3000") as client:
                result = await client.get_memory_feedback_history("mem-abc")
        assert len(result.entries) == 2

    @pytest.mark.asyncio
    async def test_get_agent_feedback_summary_async(self):
        """AsyncDakeraClient.get_agent_feedback_summary() returns summary."""
        import httpx
        with patch("httpx.AsyncClient.request") as mock_req:
            mock_req.return_value = httpx.Response(200, json=AGENT_FEEDBACK_SUMMARY)
            async with AsyncDakeraClient("http://localhost:3000") as client:
                result = await client.get_agent_feedback_summary("agent-1")
        assert result.upvotes == 42
        assert result.health_score == pytest.approx(0.78)

    @pytest.mark.asyncio
    async def test_patch_memory_importance_async(self):
        """AsyncDakeraClient.patch_memory_importance() PATCHes importance."""
        import httpx
        with patch("httpx.AsyncClient.request") as mock_req:
            mock_req.return_value = httpx.Response(200, json=FEEDBACK_RESPONSE)
            async with AsyncDakeraClient("http://localhost:3000") as client:
                result = await client.patch_memory_importance("mem-abc", "agent-1", 0.92)
        assert result.new_importance == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_get_feedback_health_async(self):
        """AsyncDakeraClient.get_feedback_health() returns health response."""
        import httpx
        with patch("httpx.AsyncClient.request") as mock_req:
            mock_req.return_value = httpx.Response(200, json=FEEDBACK_HEALTH_RESPONSE)
            async with AsyncDakeraClient("http://localhost:3000") as client:
                result = await client.get_feedback_health("agent-1")
        assert result.health_score == pytest.approx(0.78)
        assert result.memory_count == 120


class TestFeedbackLoopModels:
    """Unit tests for INT-1 feedback loop model dataclasses."""

    def test_feedback_signal_values(self):
        """FeedbackSignal enum has all INT-1 signal values."""
        from dakera import FeedbackSignal
        assert FeedbackSignal.UPVOTE.value == "upvote"
        assert FeedbackSignal.DOWNVOTE.value == "downvote"
        assert FeedbackSignal.FLAG.value == "flag"
        assert FeedbackSignal.POSITIVE.value == "positive"
        assert FeedbackSignal.NEGATIVE.value == "negative"

    def test_feedback_response_from_dict(self):
        """FeedbackResponse.from_dict() parses all fields."""
        from dakera import FeedbackResponse, FeedbackSignal
        resp = FeedbackResponse.from_dict(FEEDBACK_RESPONSE)
        assert resp.memory_id == "mem-abc"
        assert resp.new_importance == pytest.approx(0.92)
        assert resp.signal == FeedbackSignal.UPVOTE

    def test_feedback_history_entry_from_dict(self):
        """FeedbackHistoryEntry.from_dict() parses entry fields."""
        from dakera import FeedbackHistoryEntry, FeedbackSignal
        entry = FeedbackHistoryEntry.from_dict(
            {
                "signal": "downvote",
                "timestamp": 1774001000,
                "old_importance": 0.5,
                "new_importance": 0.425,
            }
        )
        assert entry.signal == FeedbackSignal.DOWNVOTE
        assert entry.timestamp == 1774001000
        assert entry.old_importance == pytest.approx(0.5)
        assert entry.new_importance == pytest.approx(0.425)

    def test_feedback_history_response_from_dict(self):
        """FeedbackHistoryResponse.from_dict() builds entry list."""
        from dakera import FeedbackHistoryResponse
        resp = FeedbackHistoryResponse.from_dict(FEEDBACK_HISTORY_RESPONSE)
        assert resp.memory_id == "mem-abc"
        assert len(resp.entries) == 2

    def test_feedback_history_response_empty_entries(self):
        """FeedbackHistoryResponse.from_dict() handles empty entries."""
        from dakera import FeedbackHistoryResponse
        resp = FeedbackHistoryResponse.from_dict({"memory_id": "x", "entries": []})
        assert resp.entries == []

    def test_agent_feedback_summary_from_dict(self):
        """AgentFeedbackSummary.from_dict() parses all fields."""
        from dakera import AgentFeedbackSummary
        summary = AgentFeedbackSummary.from_dict(AGENT_FEEDBACK_SUMMARY)
        assert summary.agent_id == "agent-1"
        assert summary.total_feedback == 51
        assert summary.health_score == pytest.approx(0.78)

    def test_feedback_health_response_from_dict(self):
        """FeedbackHealthResponse.from_dict() parses all fields."""
        from dakera import FeedbackHealthResponse
        health = FeedbackHealthResponse.from_dict(FEEDBACK_HEALTH_RESPONSE)
        assert health.agent_id == "agent-1"
        assert health.health_score == pytest.approx(0.78)
        assert health.memory_count == 120
        assert health.avg_importance == pytest.approx(0.72)


class TestNamespaceKeysSEC1:
    """Tests for SEC-1 namespace-scoped API key client methods."""

    def test_create_namespace_key(self, client, mock_responses):
        """create_namespace_key() POSTs to /v1/namespaces/:ns/keys and
        returns CreateNamespaceKeyResponse."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/prod-ns/keys",
            json=CREATE_NAMESPACE_KEY_RESPONSE,
            status=200,
        )
        from dakera import CreateNamespaceKeyResponse
        result = client.create_namespace_key("prod-ns", "ci-runner")
        assert isinstance(result, CreateNamespaceKeyResponse)
        assert result.key_id == "key-abc"
        assert result.key == "dak_live_xxxxxxxxxxxx"
        assert result.namespace == "prod-ns"
        assert result.name == "ci-runner"
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["name"] == "ci-runner"

    def test_create_namespace_key_with_expiry(self, client, mock_responses):
        """create_namespace_key() sends expires_in_days when provided."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/prod-ns/keys",
            json=CREATE_NAMESPACE_KEY_RESPONSE,
            status=200,
        )
        client.create_namespace_key("prod-ns", "ci-runner", expires_in_days=30)
        import json as _json
        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["expires_in_days"] == 30

    def test_list_namespace_keys(self, client, mock_responses):
        """list_namespace_keys() GETs /v1/namespaces/:ns/keys and
        returns ListNamespaceKeysResponse."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/prod-ns/keys",
            json=LIST_NAMESPACE_KEYS_RESPONSE,
            status=200,
        )
        from dakera import ListNamespaceKeysResponse, NamespaceKeyInfo
        result = client.list_namespace_keys("prod-ns")
        assert isinstance(result, ListNamespaceKeysResponse)
        assert result.namespace == "prod-ns"
        assert result.total == 1
        assert len(result.keys) == 1
        assert isinstance(result.keys[0], NamespaceKeyInfo)
        assert result.keys[0].key_id == "key-abc"

    def test_delete_namespace_key(self, client, mock_responses):
        """delete_namespace_key() DELETEs /v1/namespaces/:ns/keys/:key_id."""
        mock_responses.add(
            responses.DELETE,
            "http://localhost:3000/v1/namespaces/prod-ns/keys/key-abc",
            json={"success": True, "message": "Key revoked."},
            status=200,
        )
        result = client.delete_namespace_key("prod-ns", "key-abc")
        assert result["success"] is True

    def test_get_namespace_key_usage(self, client, mock_responses):
        """get_namespace_key_usage() GETs /v1/namespaces/:ns/keys/:key_id/usage."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/prod-ns/keys/key-abc/usage",
            json=NAMESPACE_KEY_USAGE_RESPONSE,
            status=200,
        )
        from dakera import NamespaceKeyUsageResponse
        result = client.get_namespace_key_usage("prod-ns", "key-abc")
        assert isinstance(result, NamespaceKeyUsageResponse)
        assert result.key_id == "key-abc"
        assert result.namespace == "prod-ns"
        assert result.total_requests == 1000
        assert result.successful_requests == 980
        assert result.avg_latency_ms == pytest.approx(12.4)

    async def test_create_namespace_key_async(self):
        """AsyncDakeraClient.create_namespace_key() calls correct endpoint."""
        from unittest.mock import AsyncMock, patch

        import httpx

        from dakera import AsyncDakeraClient, CreateNamespaceKeyResponse
        mock_req = AsyncMock(return_value=httpx.Response(200, json=CREATE_NAMESPACE_KEY_RESPONSE))
        with patch("httpx.AsyncClient.request", mock_req):
            async with AsyncDakeraClient("http://localhost:3000") as client:
                result = await client.create_namespace_key("prod-ns", "ci-runner")
        assert isinstance(result, CreateNamespaceKeyResponse)
        assert result.key == "dak_live_xxxxxxxxxxxx"

    def test_namespace_key_info_from_dict(self):
        """NamespaceKeyInfo.from_dict() parses all fields correctly."""
        from dakera import NamespaceKeyInfo
        info = NamespaceKeyInfo.from_dict(NAMESPACE_KEY_INFO)
        assert info.key_id == "key-abc"
        assert info.namespace == "prod-ns"
        assert info.active is True
        assert info.expires_at is None

    def test_namespace_key_usage_defaults(self):
        """NamespaceKeyUsageResponse.from_dict() uses 0 defaults for missing fields."""
        from dakera import NamespaceKeyUsageResponse
        resp = NamespaceKeyUsageResponse.from_dict({"key_id": "k", "namespace": "n"})
        assert resp.total_requests == 0
        assert resp.avg_latency_ms == 0.0


class TestOpsMetricsINFRA3:
    """Tests for INFRA-3 Prometheus metrics endpoint (v0.9.3)."""

    PROMETHEUS_TEXT = (
        "# HELP dakera_memory_store_total Total memory store operations\n"
        "# TYPE dakera_memory_store_total counter\n"
        "dakera_memory_store_total 42\n"
        "# HELP dakera_memory_count Current stored memory count\n"
        "# TYPE dakera_memory_count gauge\n"
        "dakera_memory_count 1024\n"
    )

    def test_ops_metrics_returns_prometheus_text(self, client, mock_responses):
        """ops_metrics() GETs /v1/ops/metrics and returns Prometheus text body."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/ops/metrics",
            body=self.PROMETHEUS_TEXT,
            content_type="text/plain; version=0.0.4; charset=utf-8",
            status=200,
        )

        result = client.ops_metrics()

        assert isinstance(result, str)
        assert "dakera_memory_store_total" in result
        assert "dakera_memory_count" in result
        assert mock_responses.calls[0].request.method == "GET"
        assert "/v1/ops/metrics" in mock_responses.calls[0].request.url

    def test_ops_metrics_requires_admin_scope_error(self, client, mock_responses):
        """ops_metrics() raises AuthorizationError on 403 (insufficient scope)."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/ops/metrics",
            json={"error": "Admin scope required", "code": "AUTHORIZATION_ERROR"},
            status=403,
        )

        from dakera import AuthorizationError
        with pytest.raises(AuthorizationError):
            client.ops_metrics()


class TestRotateEncryptionKeySEC3:
    """Tests for SEC-3 AES-256-GCM encryption key rotation endpoint."""

    def test_rotate_encryption_key_returns_response(self, client, mock_responses):
        """rotate_encryption_key() POSTs /v1/admin/encryption/rotate-key and returns counts."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/encryption/rotate-key",
            json={"rotated": 42, "skipped": 3, "namespaces": ["ns-a", "ns-b"]},
            status=200,
        )

        from dakera import RotateEncryptionKeyResponse

        result = client.rotate_encryption_key("new-secret-passphrase")

        assert isinstance(result, RotateEncryptionKeyResponse)
        assert result.rotated == 42
        assert result.skipped == 3
        assert result.namespaces == ["ns-a", "ns-b"]
        assert mock_responses.calls[0].request.method == "POST"
        assert "/v1/admin/encryption/rotate-key" in mock_responses.calls[0].request.url

    def test_rotate_encryption_key_with_namespace(self, client, mock_responses):
        """rotate_encryption_key() sends namespace field when provided."""
        import json

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/encryption/rotate-key",
            json={"rotated": 5, "skipped": 0, "namespaces": ["my-ns"]},
            status=200,
        )

        client.rotate_encryption_key("new-key", namespace="my-ns")

        body = json.loads(mock_responses.calls[0].request.body)
        assert body["namespace"] == "my-ns"
        assert body["new_key"] == "new-key"


class TestRotateEncryptionKeyModelSEC3:
    """Unit tests for RotateEncryptionKeyRequest / RotateEncryptionKeyResponse models."""

    def test_response_from_dict(self):
        """RotateEncryptionKeyResponse.from_dict() maps all fields."""
        from dakera import RotateEncryptionKeyResponse

        resp = RotateEncryptionKeyResponse.from_dict(
            {"rotated": 10, "skipped": 2, "namespaces": ["a", "b"]}
        )
        assert resp.rotated == 10
        assert resp.skipped == 2
        assert resp.namespaces == ["a", "b"]

    def test_response_defaults(self):
        """RotateEncryptionKeyResponse.from_dict() uses 0 / [] defaults."""
        from dakera import RotateEncryptionKeyResponse

        resp = RotateEncryptionKeyResponse.from_dict({})
        assert resp.rotated == 0
        assert resp.skipped == 0
        assert resp.namespaces == []


class TestOdeExtractEntitiesODE2:
    """Tests for ODE-2 GLiNER entity extraction via dakera-ode sidecar."""

    ODE_RESPONSE = {
        "entities": [
            {"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.97},
            {"text": "Paris", "label": "location", "start": 16, "end": 21, "score": 0.92},
        ],
        "model": "gliner-multi-v2.1",
        "processing_time_ms": 34,
    }

    @pytest.fixture
    def ode_client(self):
        return DakeraClient("http://localhost:3000", ode_url="http://localhost:8080")

    def test_ode_extract_entities_returns_response(self, ode_client, mock_responses):
        """ode_extract_entities() POSTs /ode/extract and returns ExtractEntitiesResponse."""
        mock_responses.add(
            responses.POST,
            "http://localhost:8080/ode/extract",
            json=self.ODE_RESPONSE,
            status=200,
        )

        from dakera import ExtractEntitiesResponse, OdeEntity

        result = ode_client.ode_extract_entities("Alice lives in Paris.", "agent-1")

        assert isinstance(result, ExtractEntitiesResponse)
        assert len(result.entities) == 2
        assert isinstance(result.entities[0], OdeEntity)
        assert result.entities[0].text == "Alice"
        assert result.entities[0].label == "person"
        assert result.entities[0].start == 0
        assert result.entities[0].end == 5
        assert result.entities[1].text == "Paris"
        assert result.model == "gliner-multi-v2.1"
        assert result.processing_time_ms == 34
        assert mock_responses.calls[0].request.method == "POST"
        assert "/ode/extract" in mock_responses.calls[0].request.url

    def test_ode_extract_entities_sends_optional_fields(self, ode_client, mock_responses):
        """ode_extract_entities() includes memory_id and entity_types when provided."""
        import json as _json

        mock_responses.add(
            responses.POST,
            "http://localhost:8080/ode/extract",
            json=self.ODE_RESPONSE,
            status=200,
        )

        ode_client.ode_extract_entities(
            "Alice works at Dakera.",
            "agent-2",
            memory_id="mem-abc",
            entity_types=["person", "org"],
        )

        body = _json.loads(mock_responses.calls[0].request.body)
        assert body["content"] == "Alice works at Dakera."
        assert body["agent_id"] == "agent-2"
        assert body["memory_id"] == "mem-abc"
        assert body["entity_types"] == ["person", "org"]

    def test_ode_extract_entities_raises_without_ode_url(self, client, mock_responses):
        """ode_extract_entities() raises ValueError when ode_url is not set."""
        with pytest.raises(ValueError, match="ode_url"):
            client.ode_extract_entities("some text", "agent-1")


class TestOdeEntityModelODE2:
    """Unit tests for OdeEntity and ExtractEntitiesResponse models."""

    def test_ode_entity_from_dict(self):
        """OdeEntity.from_dict() maps all fields correctly."""
        from dakera import OdeEntity

        entity = OdeEntity.from_dict(
            {"text": "Berlin", "label": "location", "start": 10, "end": 16, "score": 0.88}
        )
        assert entity.text == "Berlin"
        assert entity.label == "location"
        assert entity.start == 10
        assert entity.end == 16
        assert entity.score == 0.88

    def test_extract_entities_response_from_dict(self):
        """ExtractEntitiesResponse.from_dict() maps entities, model, and processing_time_ms."""
        from dakera import ExtractEntitiesResponse

        resp = ExtractEntitiesResponse.from_dict({
            "entities": [
                {"text": "Bob", "label": "person", "start": 0, "end": 3, "score": 0.95}
            ],
            "model": "gliner-multi-v2.1",
            "processing_time_ms": 12,
        })
        assert len(resp.entities) == 1
        assert resp.entities[0].text == "Bob"
        assert resp.model == "gliner-multi-v2.1"
        assert resp.processing_time_ms == 12

    def test_extract_entities_response_empty(self):
        """ExtractEntitiesResponse.from_dict() handles empty entity list."""
        from dakera import ExtractEntitiesResponse

        resp = ExtractEntitiesResponse.from_dict(
            {"entities": [], "model": "m", "processing_time_ms": 5}
        )
        assert resp.entities == []
        assert resp.model == "m"
