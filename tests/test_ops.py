"""Tests for ops methods (diagnostics, jobs, compact, shutdown, stats, metrics)."""

import json

import pytest
import responses

from dakera import DakeraClient, ServerError


@pytest.fixture
def client():
    """Create a test client."""
    return DakeraClient("http://localhost:3000")


@pytest.fixture
def mock_responses():
    """Enable responses mocking."""
    with responses.RequestsMock() as rsps:
        yield rsps


class TestOpsDiagnostics:
    """Tests for ops diagnostics methods."""

    def test_ops_diagnostics(self, client, mock_responses):
        """Test getting system diagnostics."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/ops/diagnostics",
            json={
                "memory_usage_bytes": 512000000,
                "cpu_usage_percent": 25.3,
                "disk_usage_bytes": 10000000000,
                "open_connections": 42,
                "goroutines": 150,
            },
            status=200,
        )
        result = client.ops_diagnostics()
        assert result["memory_usage_bytes"] == 512000000
        assert result["cpu_usage_percent"] == 25.3
        assert len(mock_responses.calls) == 1

    def test_ops_diagnostics_error(self, client, mock_responses):
        """Test diagnostics with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/ops/diagnostics",
            json={"error": "diagnostics unavailable"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.ops_diagnostics()


class TestOpsJobs:
    """Tests for background job methods."""

    def test_ops_list_jobs(self, client, mock_responses):
        """Test listing background jobs."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/ops/jobs",
            json=[
                {"id": "job-1", "type": "compaction", "status": "running", "progress": 0.75},
                {"id": "job-2", "type": "backup", "status": "complete", "progress": 1.0},
            ],
            status=200,
        )
        result = client.ops_list_jobs()
        assert len(result) == 2
        assert result[0]["id"] == "job-1"
        assert result[0]["status"] == "running"

    def test_ops_get_job(self, client, mock_responses):
        """Test getting a specific job."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/ops/jobs/job-1",
            json={
                "id": "job-1",
                "type": "compaction",
                "status": "running",
                "progress": 0.75,
                "started_at": "2026-05-17T10:00:00Z",
            },
            status=200,
        )
        result = client.ops_get_job("job-1")
        assert result["id"] == "job-1"
        assert result["progress"] == 0.75

    def test_ops_get_job_error(self, client, mock_responses):
        """Test getting a non-existent job."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/ops/jobs/nonexistent",
            json={"error": "job not found"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.ops_get_job("nonexistent")


class TestOpsCompact:
    """Tests for compaction methods."""

    def test_ops_compact(self, client, mock_responses):
        """Test triggering compaction."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/ops/compact",
            json={"job_id": "compact-1", "status": "started"},
            status=200,
        )
        result = client.ops_compact(namespace="test-ns", force=True)
        assert result["job_id"] == "compact-1"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["namespace"] == "test-ns"
        assert req_body["force"] is True

    def test_ops_compact_all(self, client, mock_responses):
        """Test triggering compaction for all namespaces."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/ops/compact",
            json={"job_id": "compact-all-1", "status": "started"},
            status=200,
        )
        result = client.ops_compact()
        assert result["status"] == "started"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert "namespace" not in req_body
        assert req_body["force"] is False

    def test_ops_compact_error(self, client, mock_responses):
        """Test compaction with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/ops/compact",
            json={"error": "compaction already in progress"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.ops_compact()


class TestOpsShutdown:
    """Tests for graceful shutdown method."""

    def test_ops_shutdown(self, client, mock_responses):
        """Test requesting graceful shutdown."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/ops/shutdown",
            json={"status": "shutting_down", "draining": True},
            status=200,
        )
        result = client.ops_shutdown()
        assert result["status"] == "shutting_down"
        assert len(mock_responses.calls) == 1


class TestOpsStats:
    """Tests for ops stats and metrics methods."""

    def test_ops_stats(self, client, mock_responses):
        """Test getting server stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/ops/stats",
            json={
                "version": "0.11.54",
                "total_vectors": 150000,
                "namespace_count": 12,
                "uptime_seconds": 86400,
                "timestamp": 1747500000,
                "state": "healthy",
            },
            status=200,
        )
        result = client.ops_stats()
        assert result["version"] == "0.11.54"
        assert result["total_vectors"] == 150000
        assert result["state"] == "healthy"

    def test_ops_stats_error(self, client, mock_responses):
        """Test ops stats with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/ops/stats",
            json={"error": "internal error"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.ops_stats()

    def test_ops_metrics(self, client, mock_responses):
        """Test getting Prometheus metrics."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/ops/metrics",
            json="# HELP dakera_queries_total Total queries\ndakera_queries_total 12345",
            status=200,
        )
        result = client.ops_metrics()
        assert "dakera_queries_total" in result


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health(self, client, mock_responses):
        """Test basic health check."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health",
            json={"status": "ok", "version": "0.11.54"},
            status=200,
        )
        result = client.health()
        assert result["status"] == "ok"

    def test_health_ready(self, client, mock_responses):
        """Test K8s readiness probe."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health/ready",
            json={"ready": True, "storage": "ok", "dependencies": "ok"},
            status=200,
        )
        result = client.health_ready()
        assert result["ready"] is True

    def test_health_live(self, client, mock_responses):
        """Test K8s liveness probe."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/health/live",
            json={"alive": True},
            status=200,
        )
        result = client.health_live()
        assert result["alive"] is True


class TestNamespaceOps:
    """Tests for namespace-scoped ops methods."""

    def test_optimize_namespace(self, client, mock_responses):
        """Test optimizing a namespace."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/namespaces/test-ns/optimize",
            json={"optimized": True, "segments_merged": 3},
            status=200,
        )
        result = client.optimize_namespace("test-ns")
        assert result["optimized"] is True

    def test_index_stats(self, client, mock_responses):
        """Test getting index stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/namespaces/test-ns/index/stats",
            json={"segments": 5, "total_vectors": 10000, "index_size_bytes": 2000000},
            status=200,
        )
        result = client.index_stats("test-ns")
        assert result["segments"] == 5
        assert result["total_vectors"] == 10000

    def test_rebuild_indexes(self, client, mock_responses):
        """Test rebuilding indexes."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/namespaces/test-ns/index/rebuild",
            json={"status": "started", "job_id": "rebuild-1"},
            status=200,
        )
        result = client.rebuild_indexes("test-ns")
        assert result["status"] == "started"

    def test_get_index_stats(self, client, mock_responses):
        """Test getting index stats via v1 namespace path."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/test-ns/stats",
            json={
                "namespace": "test-ns",
                "vector_count": 10000,
                "dimensions": 384,
                "index_type": "hnsw",
                "storage_bytes": 5000000,
            },
            status=200,
        )
        result = client.get_index_stats("test-ns")
        assert result.vector_count == 10000
        assert result.dimensions == 384

    def test_compact_namespace(self, client, mock_responses):
        """Test compacting a specific namespace."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/compact",
            json={"compacted": True, "segments_merged": 2},
            status=200,
        )
        result = client.compact("test-ns")
        assert result["compacted"] is True

    def test_flush_namespace(self, client, mock_responses):
        """Test flushing a namespace."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/flush",
            json={"flushed": True, "pending_writes": 0},
            status=200,
        )
        result = client.flush("test-ns")
        assert result["flushed"] is True


class TestBulkVectorOps:
    """Tests for bulk vector operations."""

    def test_bulk_update_vectors(self, client, mock_responses):
        """Test bulk updating vector metadata."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/vectors/bulk-update",
            json={"updated": 50, "failed": 0, "errors": []},
            status=200,
        )
        result = client.bulk_update_vectors(
            "test-ns",
            filter={"category": {"$eq": "old"}},
            update={"category": "updated"},
        )
        assert result["updated"] == 50
        assert result["failed"] == 0
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["filter"] == {"category": {"$eq": "old"}}
        assert req_body["update"] == {"category": "updated"}

    def test_bulk_update_vectors_error(self, client, mock_responses):
        """Test bulk update with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/vectors/bulk-update",
            json={"error": "namespace locked"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.bulk_update_vectors("test-ns", filter={}, update={})

    def test_bulk_delete_vectors(self, client, mock_responses):
        """Test bulk deleting vectors by filter."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/vectors/bulk-delete",
            json={"deleted": 30, "failed": 0, "errors": []},
            status=200,
        )
        result = client.bulk_delete_vectors(
            "test-ns", filter={"status": {"$eq": "expired"}}
        )
        assert result["deleted"] == 30
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["filter"] == {"status": {"$eq": "expired"}}

    def test_count_vectors(self, client, mock_responses):
        """Test counting vectors."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/vectors/count",
            json={"count": 10000, "namespace": "test-ns"},
            status=200,
        )
        result = client.count_vectors("test-ns")
        assert result["count"] == 10000

    def test_count_vectors_with_filter(self, client, mock_responses):
        """Test counting vectors with filter."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/vectors/count",
            json={"count": 500, "namespace": "test-ns"},
            status=200,
        )
        result = client.count_vectors("test-ns", filter={"type": {"$eq": "active"}})
        assert result["count"] == 500
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert "filter" in req_body


class TestFulltextExtended:
    """Tests for fulltext stats and delete methods."""

    def test_fulltext_stats(self, client, mock_responses):
        """Test getting fulltext index stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/test-ns/fulltext/stats",
            json={
                "document_count": 5000,
                "term_count": 25000,
                "index_size_bytes": 1000000,
            },
            status=200,
        )
        result = client.fulltext_stats("test-ns")
        assert result.document_count == 5000
        assert result.term_count == 25000

    def test_fulltext_stats_error(self, client, mock_responses):
        """Test fulltext stats with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/test-ns/fulltext/stats",
            json={"error": "index not found"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.fulltext_stats("test-ns")

    def test_fulltext_delete(self, client, mock_responses):
        """Test deleting documents from fulltext index."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/fulltext/delete",
            json={"deleted": 3},
            status=200,
        )
        result = client.fulltext_delete("test-ns", ids=["doc-1", "doc-2", "doc-3"])
        assert result["deleted"] == 3
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["ids"] == ["doc-1", "doc-2", "doc-3"]


class TestRouteQuery:
    """Tests for semantic query routing."""

    def test_route_query(self, client, mock_responses):
        """Test routing a query."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/route",
            json={
                "routes": [
                    {"namespace": "ns-1", "similarity": 0.92},
                    {"namespace": "ns-2", "similarity": 0.78},
                ],
                "model": "minilm",
            },
            status=200,
        )
        result = client.route_query("what is the weather?", top_k=3, min_similarity=0.3)
        assert len(result.routes) == 2
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["query"] == "what is the weather?"
        assert req_body["top_k"] == 3

    def test_route_query_error(self, client, mock_responses):
        """Test route query with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/route",
            json={"error": "routing unavailable"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.route_query("test")


class TestImportJobStatus:
    """Tests for import job status."""

    def test_import_job_status(self, client, mock_responses):
        """Test checking import job status."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/import/job-123/status",
            json={
                "job_id": "job-123",
                "status": "complete",
                "imported": 100,
                "failed": 2,
                "total": 102,
            },
            status=200,
        )
        result = client.import_job_status("job-123")
        assert result.job_id == "job-123"
        assert result.status == "complete"
        assert result.imported == 100
