"""Tests for analytics and knowledge graph methods."""

import json

import pytest
import responses

from dakera import DakeraClient, NotFoundError, ServerError


@pytest.fixture
def client():
    """Create a test client."""
    return DakeraClient("http://localhost:3000")


@pytest.fixture
def mock_responses():
    """Enable responses mocking."""
    with responses.RequestsMock() as rsps:
        yield rsps


class TestAnalyticsOverview:
    """Tests for analytics overview methods."""

    def test_analytics_overview(self, client, mock_responses):
        """Test getting analytics overview."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/overview",
            json={
                "total_queries": 50000,
                "total_upserts": 12000,
                "avg_latency_ms": 5.2,
                "period": "24h",
            },
            status=200,
        )
        result = client.analytics_overview(period="24h")
        assert result["total_queries"] == 50000
        assert result["avg_latency_ms"] == 5.2
        assert len(mock_responses.calls) == 1

    def test_analytics_overview_with_namespace(self, client, mock_responses):
        """Test analytics overview filtered by namespace."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/overview",
            json={"total_queries": 1000, "period": "7d"},
            status=200,
        )
        result = client.analytics_overview(period="7d", namespace="test-ns")
        assert result["total_queries"] == 1000

    def test_analytics_overview_error(self, client, mock_responses):
        """Test analytics overview with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/overview",
            json={"error": "analytics unavailable"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.analytics_overview()


class TestAnalyticsLatency:
    """Tests for latency analytics."""

    def test_analytics_latency(self, client, mock_responses):
        """Test getting latency analytics."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/latency",
            json={
                "p50_ms": 2.1,
                "p95_ms": 8.5,
                "p99_ms": 15.0,
                "max_ms": 120.0,
                "period": "1h",
            },
            status=200,
        )
        result = client.analytics_latency(period="1h")
        assert result["p50_ms"] == 2.1
        assert result["p99_ms"] == 15.0

    def test_analytics_latency_with_namespace(self, client, mock_responses):
        """Test latency analytics filtered by namespace."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/latency",
            json={"p50_ms": 3.0, "p99_ms": 20.0},
            status=200,
        )
        result = client.analytics_latency(namespace="test-ns")
        assert result["p50_ms"] == 3.0


class TestAnalyticsThroughput:
    """Tests for throughput analytics."""

    def test_analytics_throughput(self, client, mock_responses):
        """Test getting throughput analytics."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/throughput",
            json={
                "queries_per_second": 150.0,
                "upserts_per_second": 45.0,
                "bytes_per_second": 2000000,
                "period": "1h",
            },
            status=200,
        )
        result = client.analytics_throughput(period="1h")
        assert result["queries_per_second"] == 150.0
        assert result["upserts_per_second"] == 45.0

    def test_analytics_throughput_error(self, client, mock_responses):
        """Test throughput analytics with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/throughput",
            json={"error": "internal error"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.analytics_throughput()


class TestAnalyticsStorage:
    """Tests for storage analytics."""

    def test_analytics_storage(self, client, mock_responses):
        """Test getting storage analytics."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/storage",
            json={
                "total_bytes": 5000000000,
                "vector_bytes": 3000000000,
                "metadata_bytes": 1000000000,
                "index_bytes": 1000000000,
            },
            status=200,
        )
        result = client.analytics_storage()
        assert result["total_bytes"] == 5000000000

    def test_analytics_storage_with_namespace(self, client, mock_responses):
        """Test storage analytics filtered by namespace."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/analytics/storage",
            json={"total_bytes": 500000000},
            status=200,
        )
        result = client.analytics_storage(namespace="test-ns")
        assert result["total_bytes"] == 500000000


class TestKnowledgeGraph:
    """Tests for knowledge graph methods."""

    def test_knowledge_graph(self, client, mock_responses):
        """Test building a knowledge graph from seed memory."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/graph",
            json={
                "nodes": [
                    {"id": "mem-1", "label": "primary"},
                    {"id": "mem-2", "label": "related"},
                ],
                "edges": [
                    {"source": "mem-1", "target": "mem-2", "weight": 0.85}
                ],
            },
            status=200,
        )
        result = client.knowledge_graph("agent-1", memory_id="mem-1", depth=2)
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["agent_id"] == "agent-1"
        assert req_body["memory_id"] == "mem-1"
        assert req_body["depth"] == 2

    def test_full_knowledge_graph(self, client, mock_responses):
        """Test building full knowledge graph for an agent."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/graph/full",
            json={
                "nodes": [{"id": f"mem-{i}"} for i in range(5)],
                "edges": [{"source": "mem-0", "target": "mem-1", "weight": 0.9}],
                "clusters": [{"id": "cluster-0", "members": ["mem-0", "mem-1"]}],
            },
            status=200,
        )
        result = client.full_knowledge_graph(
            "agent-1", max_nodes=50, min_similarity=0.5, cluster_threshold=0.7
        )
        assert len(result["nodes"]) == 5
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["max_nodes"] == 50
        assert req_body["min_similarity"] == 0.5
        assert req_body["cluster_threshold"] == 0.7

    def test_knowledge_graph_error(self, client, mock_responses):
        """Test knowledge graph with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/graph",
            json={"error": "graph construction failed"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.knowledge_graph("agent-1")


class TestKnowledgeSummarize:
    """Tests for knowledge summarization."""

    def test_summarize(self, client, mock_responses):
        """Test summarizing memories."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/summarize",
            json={
                "summary": "Agent prefers concise responses.",
                "source_count": 5,
                "confidence": 0.92,
            },
            status=200,
        )
        result = client.summarize("agent-1", memory_ids=["m1", "m2"], target_type="semantic")
        assert result["summary"] == "Agent prefers concise responses."
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["agent_id"] == "agent-1"
        assert req_body["memory_ids"] == ["m1", "m2"]
        assert req_body["dry_run"] is False

    def test_summarize_dry_run(self, client, mock_responses):
        """Test summarize in dry run mode."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/summarize",
            json={"summary": "preview", "source_count": 3, "dry_run": True},
            status=200,
        )
        result = client.summarize("agent-1", dry_run=True)
        assert result["dry_run"] is True


class TestKnowledgeDeduplicate:
    """Tests for knowledge deduplication."""

    def test_deduplicate(self, client, mock_responses):
        """Test deduplicating memories."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/deduplicate",
            json={"removed": 8, "kept": 42, "threshold": 0.95},
            status=200,
        )
        result = client.deduplicate("agent-1", threshold=0.95, memory_type="episodic")
        assert result["removed"] == 8
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["threshold"] == 0.95
        assert req_body["memory_type"] == "episodic"

    def test_deduplicate_dry_run(self, client, mock_responses):
        """Test deduplication in dry run mode."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/deduplicate",
            json={"removed": 0, "would_remove": 5, "dry_run": True},
            status=200,
        )
        result = client.deduplicate("agent-1", dry_run=True)
        assert result["would_remove"] == 5


class TestKG2QueryExport:
    """Tests for KG-2 graph query and export."""

    def test_knowledge_query(self, client, mock_responses):
        """Test querying knowledge graph."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/knowledge/query",
            json={
                "agent_id": "agent-1",
                "node_count": 2,
                "edge_count": 1,
                "edges": [
                    {
                        "id": "edge-1",
                        "source_id": "mem-1",
                        "target_id": "mem-2",
                        "edge_type": "related_to",
                        "weight": 0.85,
                        "created_at": 1700000000,
                    }
                ],
            },
            status=200,
        )
        result = client.knowledge_query(
            "agent-1", root_id="mem-1", edge_type="related_to", min_weight=0.5, max_depth=2
        )
        assert result.edge_count == 1
        assert len(result.edges) == 1

    def test_knowledge_query_error(self, client, mock_responses):
        """Test knowledge query with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/knowledge/query",
            json={"error": "graph unavailable"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.knowledge_query("agent-1")

    def test_knowledge_path(self, client, mock_responses):
        """Test finding shortest path between memories."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/knowledge/path",
            json={
                "agent_id": "agent-1",
                "from_id": "mem-1",
                "to_id": "mem-5",
                "hop_count": 2,
                "path": ["mem-1", "mem-3", "mem-5"],
            },
            status=200,
        )
        result = client.knowledge_path("agent-1", from_id="mem-1", to_id="mem-5")
        assert result.hop_count == 2
        assert len(result.path) == 3

    def test_knowledge_path_not_found(self, client, mock_responses):
        """Test knowledge path when no path exists."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/knowledge/path",
            json={"error": "no path found"},
            status=404,
        )
        with pytest.raises(NotFoundError):
            client.knowledge_path("agent-1", from_id="mem-1", to_id="mem-99")

    def test_knowledge_export(self, client, mock_responses):
        """Test exporting knowledge graph."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/knowledge/export",
            json={
                "agent_id": "agent-1",
                "format": "json",
                "node_count": 2,
                "edge_count": 1,
                "edges": [
                    {
                        "id": "edge-1",
                        "source_id": "mem-1",
                        "target_id": "mem-2",
                        "edge_type": "related_to",
                        "weight": 0.8,
                        "created_at": 1700000000,
                    }
                ],
            },
            status=200,
        )
        result = client.knowledge_export("agent-1", format="json")
        assert result.node_count == 2
        assert len(result.edges) == 1


class TestCrossAgentNetwork:
    """Tests for cross-agent network methods."""

    def test_cross_agent_network(self, client, mock_responses):
        """Test building cross-agent network."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/network/cross-agent",
            json={
                "agents": [
                    {"agent_id": "agent-1", "memory_count": 100, "avg_importance": 0.7},
                    {"agent_id": "agent-2", "memory_count": 50, "avg_importance": 0.6},
                ],
                "nodes": [
                    {
                        "id": "mem-1",
                        "agent_id": "agent-1",
                        "content": "memory one",
                        "importance": 0.8,
                        "tags": ["tag1"],
                        "memory_type": "episodic",
                        "created_at": 1700000000,
                    },
                    {
                        "id": "mem-2",
                        "agent_id": "agent-2",
                        "content": "memory two",
                        "importance": 0.6,
                        "tags": [],
                        "memory_type": "semantic",
                        "created_at": 1700000001,
                    },
                ],
                "edges": [
                    {
                        "source": "mem-1",
                        "target": "mem-2",
                        "source_agent": "agent-1",
                        "target_agent": "agent-2",
                        "similarity": 0.72,
                    }
                ],
                "stats": {
                    "total_agents": 2,
                    "total_nodes": 2,
                    "total_cross_edges": 1,
                    "density": 0.5,
                },
                "node_count": 2,
            },
            status=200,
        )
        result = client.cross_agent_network(
            agent_ids=["agent-1", "agent-2"],
            min_similarity=0.5,
            max_nodes_per_agent=50,
        )
        assert result.stats.total_agents == 2
        assert result.stats.total_cross_edges == 1
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["agent_ids"] == ["agent-1", "agent-2"]
        assert req_body["min_similarity"] == 0.5

    def test_cross_agent_network_error(self, client, mock_responses):
        """Test cross-agent network with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/knowledge/network/cross-agent",
            json={"error": "insufficient agents"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.cross_agent_network()


class TestMemoryKnowledgeGraph:
    """Tests for CE-5 memory knowledge graph operations."""

    def test_memory_graph(self, client, mock_responses):
        """Test traversing memory graph."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-1/graph",
            json={
                "root_id": "mem-1",
                "depth": 2,
                "nodes": [
                    {"memory_id": "mem-1", "content_preview": "first",
                     "importance": 0.9, "depth": 0},
                    {"memory_id": "mem-2", "content_preview": "second",
                     "importance": 0.7, "depth": 1},
                ],
                "edges": [
                    {
                        "id": "edge-1",
                        "source_id": "mem-1",
                        "target_id": "mem-2",
                        "edge_type": "related_to",
                        "weight": 0.8,
                        "created_at": 1700000000,
                    }
                ],
            },
            status=200,
        )
        result = client.memory_graph("mem-1", depth=2, types=["related_to"])
        assert result.root_id == "mem-1"
        assert len(result.nodes) == 2

    def test_memory_path(self, client, mock_responses):
        """Test finding path between memories."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-1/path",
            json={
                "source_id": "mem-1",
                "target_id": "mem-5",
                "path": ["mem-1", "mem-3", "mem-5"],
                "hops": 2,
                "edges": [
                    {
                        "id": "edge-1",
                        "source_id": "mem-1",
                        "target_id": "mem-3",
                        "edge_type": "related_to",
                        "weight": 0.9,
                        "created_at": 1700000000,
                    },
                    {
                        "id": "edge-2",
                        "source_id": "mem-3",
                        "target_id": "mem-5",
                        "edge_type": "related_to",
                        "weight": 0.7,
                        "created_at": 1700000001,
                    },
                ],
            },
            status=200,
        )
        result = client.memory_path("mem-1", "mem-5")
        assert result.hops == 2

    def test_memory_link(self, client, mock_responses):
        """Test creating explicit link between memories."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/mem-1/links",
            json={
                "edge": {
                    "id": "edge-new",
                    "source_id": "mem-1",
                    "target_id": "mem-2",
                    "edge_type": "linked_by",
                    "weight": 1.0,
                    "created_at": 1700000000,
                }
            },
            status=200,
        )
        from dakera.models import EdgeType

        result = client.memory_link("mem-1", "mem-2", edge_type=EdgeType.LINKED_BY)
        assert result.edge.source_id == "mem-1"
        assert result.edge.target_id == "mem-2"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["target_id"] == "mem-2"
        assert req_body["edge_type"] == "linked_by"

    def test_agent_graph_export(self, client, mock_responses):
        """Test exporting agent graph."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/agent-1/graph/export",
            json={
                "agent_id": "agent-1",
                "format": "json",
                "data": '{"nodes":["mem-1","mem-2"],"edges":[]}',
                "node_count": 2,
                "edge_count": 1,
            },
            status=200,
        )
        result = client.agent_graph_export("agent-1", format="json")
        assert result.node_count == 2
        assert result.agent_id == "agent-1"
