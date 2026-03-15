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
            "http://localhost:3000/v1/namespaces/test-ns/fulltext/hybrid",
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
