"""Tests for untested memory, session, agent, import/export, audit, and extraction methods."""

import json

import pytest
import responses

from dakera import DakeraClient, NotFoundError, ServerError, ValidationError


@pytest.fixture
def client():
    """Create a test client."""
    return DakeraClient("http://localhost:3000")


@pytest.fixture
def mock_responses():
    """Enable responses mocking."""
    with responses.RequestsMock() as rsps:
        yield rsps


class TestStoreMemory:
    """Tests for store_memory method."""

    def test_store_memory_basic(self, client, mock_responses):
        """Test storing a basic memory."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/store",
            json={
                "memory": {
                    "id": "mem-1",
                    "content": "User prefers dark mode",
                    "memory_type": "semantic",
                    "importance": 0.8,
                    "agent_id": "agent-1",
                },
                "embedding_time_ms": 12,
            },
            status=200,
        )
        result = client.store_memory(
            agent_id="agent-1",
            content="User prefers dark mode",
            memory_type="semantic",
            importance=0.8,
        )
        assert result["id"] == "mem-1"
        assert result["content"] == "User prefers dark mode"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["agent_id"] == "agent-1"
        assert req_body["importance"] == 0.8

    def test_store_memory_with_ttl(self, client, mock_responses):
        """Test storing a memory with TTL."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/store",
            json={"memory": {"id": "mem-2", "content": "temp", "memory_type": "working"}},
            status=200,
        )
        result = client.store_memory(
            agent_id="agent-1",
            content="temp",
            memory_type="working",
            ttl_seconds=3600,
            tags=["temporary"],
            session_id="sess-1",
        )
        assert result["id"] == "mem-2"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["ttl_seconds"] == 3600
        assert req_body["tags"] == ["temporary"]
        assert req_body["session_id"] == "sess-1"

    def test_store_memory_with_expires_at(self, client, mock_responses):
        """Test storing a memory with explicit expiry timestamp."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/store",
            json={"memory": {"id": "mem-3", "content": "expires", "memory_type": "episodic"}},
            status=200,
        )
        client.store_memory(
            agent_id="agent-1", content="expires", expires_at=1750000000
        )
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["expires_at"] == 1750000000

    def test_store_memory_error(self, client, mock_responses):
        """Test store memory with validation error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/store",
            json={"error": "content is required"},
            status=400,
        )
        with pytest.raises(ValidationError):
            client.store_memory(agent_id="agent-1", content="")


class TestGetUpdateMemory:
    """Tests for get_memory and update_memory methods."""

    def test_get_memory(self, client, mock_responses):
        """Test getting a specific memory."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memory/get/mem-1",
            json={
                "id": "mem-1",
                "content": "important fact",
                "memory_type": "semantic",
                "importance": 0.9,
            },
            status=200,
        )
        result = client.get_memory("agent-1", "mem-1")
        assert result["id"] == "mem-1"
        assert result["importance"] == 0.9

    def test_get_memory_not_found(self, client, mock_responses):
        """Test getting non-existent memory."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memory/get/nonexistent",
            json={"error": "memory not found"},
            status=404,
        )
        with pytest.raises(NotFoundError):
            client.get_memory("agent-1", "nonexistent")

    def test_update_memory(self, client, mock_responses):
        """Test updating a memory."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/memory/update/mem-1",
            json={"id": "mem-1", "content": "updated content", "memory_type": "semantic"},
            status=200,
        )
        result = client.update_memory(
            "agent-1", "mem-1", content="updated content", memory_type="semantic"
        )
        assert result["content"] == "updated content"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["content"] == "updated content"
        assert req_body["memory_type"] == "semantic"


class TestForgetMemory:
    """Tests for forget method."""

    def test_forget(self, client, mock_responses):
        """Test forgetting a memory."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/forget",
            json={"deleted": 1},
            status=200,
        )
        result = client.forget("agent-1", "mem-1")
        assert result["deleted"] == 1
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["agent_id"] == "agent-1"
        assert req_body["memory_ids"] == ["mem-1"]


class TestSearchMemories:
    """Tests for search_memories method."""

    def test_search_memories(self, client, mock_responses):
        """Test searching memories."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/search",
            json={
                "memories": [
                    {"id": "mem-1", "content": "hello", "score": 0.92},
                    {"id": "mem-2", "content": "world", "score": 0.85},
                ]
            },
            status=200,
        )
        result = client.search_memories("agent-1", "hello world", top_k=5)
        assert len(result) == 2
        assert result[0]["score"] == 0.92
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["agent_id"] == "agent-1"
        assert req_body["query"] == "hello world"
        assert req_body["top_k"] == 5

    def test_search_memories_with_filters(self, client, mock_responses):
        """Test searching memories with type and importance filters."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/search",
            json={"memories": []},
            status=200,
        )
        client.search_memories(
            "agent-1", "test", memory_type="semantic", min_importance=0.5
        )
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["memory_type"] == "semantic"
        assert req_body["min_importance"] == 0.5


class TestCompressAgent:
    """Tests for compress_agent method."""

    def test_compress_agent(self, client, mock_responses):
        """Test compressing agent memory namespace."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/agents/agent-1/compress",
            json={
                "before_count": 500,
                "after_count": 380,
                "removed_count": 120,
                "elapsed_ms": 450,
            },
            status=200,
        )
        result = client.compress_agent("agent-1")
        assert result.before_count == 500
        assert result.after_count == 380
        assert result.removed_count == 120


class TestConsolidation:
    """Tests for consolidation methods."""

    def test_consolidate(self, client, mock_responses):
        """Test consolidating memories."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/consolidate",
            json={
                "consolidated_count": 5,
                "removed_count": 12,
                "new_memories": ["mem-new-1", "mem-new-2"],
            },
            status=200,
        )
        result = client.consolidate(
            "agent-1", memory_type="episodic", threshold=0.85, dry_run=False
        )
        assert result["consolidated_count"] == 5
        assert result["removed_count"] == 12
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["agent_id"] == "agent-1"
        assert req_body["threshold"] == 0.85

    def test_consolidate_agent(self, client, mock_responses):
        """Test agent-scoped consolidation."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/agents/agent-1/consolidate",
            json={
                "agent_id": "agent-1",
                "memories_scanned": 100,
                "clusters_found": 8,
                "memories_deprecated": 15,
                "anchor_ids": ["mem-1", "mem-2"],
                "deprecated_ids": ["mem-10", "mem-11"],
            },
            status=200,
        )
        result = client.consolidate_agent("agent-1")
        assert result["clusters_found"] == 8
        assert result["memories_deprecated"] == 15

    def test_get_consolidation_log(self, client, mock_responses):
        """Test getting consolidation log."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/agent-1/consolidation/log",
            json=[
                {
                    "timestamp": "2026-05-17T10:00:00Z",
                    "clusters_found": 5,
                    "memories_deprecated": 10,
                }
            ],
            status=200,
        )
        result = client.get_consolidation_log("agent-1")
        assert len(result) == 1
        assert result[0]["clusters_found"] == 5

    def test_patch_consolidation_config(self, client, mock_responses):
        """Test updating consolidation config."""
        mock_responses.add(
            responses.PATCH,
            "http://localhost:3000/v1/agents/agent-1/consolidation/config",
            json={"enabled": True, "epsilon": 0.3, "min_samples": 3},
            status=200,
        )
        result = client.patch_consolidation_config(
            "agent-1", enabled=True, epsilon=0.3, min_samples=3
        )
        assert result["enabled"] is True
        assert result["epsilon"] == 0.3


class TestMemoryFeedback:
    """Tests for memory feedback methods (INT-1)."""

    def test_memory_feedback(self, client, mock_responses):
        """Test submitting feedback on a memory."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/agents/agent-1/memories/feedback",
            json={"success": True, "new_importance": 0.85},
            status=200,
        )
        result = client.memory_feedback("agent-1", "mem-1", "helpful", relevance_score=0.9)
        assert result["success"] is True

    def test_feedback_memory(self, client, mock_responses):
        """Test INT-1 feedback signal on memory."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/mem-1/feedback",
            json={"memory_id": "mem-1", "signal": "upvote", "new_importance": 0.9},
            status=200,
        )
        from dakera.models import FeedbackSignal

        result = client.feedback_memory("mem-1", "agent-1", FeedbackSignal.UPVOTE)
        assert result.new_importance == 0.9
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["signal"] == "upvote"

    def test_get_memory_feedback_history(self, client, mock_responses):
        """Test getting feedback history for a memory."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memories/mem-1/feedback",
            json={
                "memory_id": "mem-1",
                "events": [
                    {"signal": "upvote", "timestamp": "2026-05-17T10:00:00Z"},
                    {"signal": "downvote", "timestamp": "2026-05-17T11:00:00Z"},
                ],
            },
            status=200,
        )
        result = client.get_memory_feedback_history("mem-1")
        assert len(result.events) == 2

    def test_get_agent_feedback_summary(self, client, mock_responses):
        """Test getting agent feedback summary."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/agent-1/feedback/summary",
            json={
                "agent_id": "agent-1",
                "upvotes": 50,
                "downvotes": 5,
                "flags": 1,
                "health_score": 0.89,
            },
            status=200,
        )
        result = client.get_agent_feedback_summary("agent-1")
        assert result.upvotes == 50
        assert result.health_score == 0.89

    def test_patch_memory_importance(self, client, mock_responses):
        """Test directly overriding importance."""
        mock_responses.add(
            responses.PATCH,
            "http://localhost:3000/v1/memories/mem-1/importance",
            json={"memory_id": "mem-1", "new_importance": 0.95},
            status=200,
        )
        result = client.patch_memory_importance("mem-1", "agent-1", 0.95)
        assert result.new_importance == 0.95

    def test_get_feedback_health(self, client, mock_responses):
        """Test getting feedback health score."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/feedback/health",
            json={
                "health_score": 0.85,
                "memory_count": 500,
                "avg_importance": 0.72,
            },
            status=200,
        )
        result = client.get_feedback_health("agent-1")
        assert result.health_score == 0.85
        assert result.memory_count == 500


class TestSessionOperations:
    """Tests for session management methods."""

    def test_start_session(self, client, mock_responses):
        """Test starting a new session."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/sessions/start",
            json={
                "session": {
                    "id": "sess-1",
                    "agent_id": "agent-1",
                    "started_at": "2026-05-17T10:00:00Z",
                    "active": True,
                }
            },
            status=200,
        )
        result = client.start_session("agent-1", metadata={"channel": "web"})
        assert result["id"] == "sess-1"
        assert result["active"] is True
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["agent_id"] == "agent-1"
        assert req_body["metadata"] == {"channel": "web"}

    def test_end_session(self, client, mock_responses):
        """Test ending a session."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/sessions/sess-1/end",
            json={"id": "sess-1", "active": False, "summary": "User asked 3 questions"},
            status=200,
        )
        result = client.end_session("sess-1", summary="User asked 3 questions")
        assert result["active"] is False
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["summary"] == "User asked 3 questions"

    def test_get_session(self, client, mock_responses):
        """Test getting session details."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/sessions/sess-1",
            json={
                "id": "sess-1",
                "agent_id": "agent-1",
                "active": True,
                "memory_count": 5,
            },
            status=200,
        )
        result = client.get_session("sess-1")
        assert result["memory_count"] == 5

    def test_list_sessions(self, client, mock_responses):
        """Test listing sessions."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/sessions",
            json=[
                {"id": "sess-1", "agent_id": "agent-1", "active": True},
                {"id": "sess-2", "agent_id": "agent-1", "active": False},
            ],
            status=200,
        )
        result = client.list_sessions(agent_id="agent-1", active_only=True, limit=10)
        assert len(result) == 2

    def test_session_memories(self, client, mock_responses):
        """Test getting memories for a session."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/sessions/sess-1/memories",
            json={
                "memories": [
                    {"id": "mem-1", "content": "first interaction"},
                    {"id": "mem-2", "content": "second interaction"},
                ]
            },
            status=200,
        )
        result = client.session_memories("sess-1")
        assert len(result) == 2
        assert result[0]["content"] == "first interaction"


class TestAgentOperations:
    """Tests for agent management methods."""

    def test_list_agents(self, client, mock_responses):
        """Test listing all agents."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents",
            json=[
                {"agent_id": "agent-1", "memory_count": 100},
                {"agent_id": "agent-2", "memory_count": 50},
            ],
            status=200,
        )
        result = client.list_agents()
        assert len(result) == 2

    def test_agent_memories(self, client, mock_responses):
        """Test getting memories for an agent."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/agent-1/memories",
            json=[
                {"id": "mem-1", "content": "fact 1", "memory_type": "semantic"},
                {"id": "mem-2", "content": "fact 2", "memory_type": "semantic"},
            ],
            status=200,
        )
        result = client.agent_memories("agent-1", memory_type="semantic", limit=10)
        assert len(result) == 2

    def test_agent_stats(self, client, mock_responses):
        """Test getting agent stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/agent-1/stats",
            json={
                "agent_id": "agent-1",
                "total_memories": 150,
                "total_sessions": 20,
                "avg_importance": 0.65,
            },
            status=200,
        )
        result = client.agent_stats("agent-1")
        assert result["total_memories"] == 150
        assert result["avg_importance"] == 0.65

    def test_agent_sessions(self, client, mock_responses):
        """Test getting sessions for an agent."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/agent-1/sessions",
            json=[
                {"id": "sess-1", "active": True},
                {"id": "sess-2", "active": False},
            ],
            status=200,
        )
        result = client.agent_sessions("agent-1", active_only=True, limit=5)
        assert len(result) == 2

    def test_wake_up(self, client, mock_responses):
        """Test getting wake-up context for an agent."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/agents/agent-1/wake-up",
            json={
                "memories": [
                    {"id": "mem-1", "content": "critical fact", "importance": 0.95},
                    {"id": "mem-2", "content": "important preference", "importance": 0.88},
                ],
                "total_available": 150,
            },
            status=200,
        )
        result = client.wake_up("agent-1", top_n=20, min_importance=0.5)
        assert result.total_available == 150
        assert len(result.memories) == 2


class TestImportExport:
    """Tests for memory import/export methods."""

    def test_import_memories(self, client, mock_responses):
        """Test importing memories."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/import",
            json={"imported": 10, "failed": 1, "errors": ["row 5: invalid format"]},
            status=200,
        )
        result = client.import_memories(
            data=[{"content": "test", "memory_type": "episodic"}],
            format="jsonl",
            agent_id="agent-1",
            namespace="test-ns",
        )
        assert result.imported == 10
        assert result.failed == 1
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["format"] == "jsonl"
        assert req_body["agent_id"] == "agent-1"

    def test_import_memories_error(self, client, mock_responses):
        """Test import with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/import",
            json={"error": "unsupported format"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.import_memories(data=[], format="invalid")

    def test_export_memories(self, client, mock_responses):
        """Test exporting memories."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/export",
            json={
                "data": [
                    {"id": "mem-1", "content": "fact", "memory_type": "semantic"}
                ],
                "count": 1,
                "format": "jsonl",
            },
            status=200,
        )
        result = client.export_memories(format="jsonl", agent_id="agent-1", limit=100)
        assert result.count == 1
        assert result.format == "jsonl"


class TestAuditLog:
    """Tests for OBS-1 audit log methods."""

    def test_list_audit_events(self, client, mock_responses):
        """Test listing audit events."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/audit",
            json={
                "events": [
                    {
                        "id": "evt-1",
                        "event_type": "memory.stored",
                        "agent_id": "agent-1",
                        "timestamp": 1747500000,
                    },
                    {
                        "id": "evt-2",
                        "event_type": "memory.recalled",
                        "agent_id": "agent-1",
                        "timestamp": 1747500100,
                    },
                ],
                "next_cursor": "cursor-abc",
            },
            status=200,
        )
        result = client.list_audit_events(
            agent_id="agent-1", event_type="memory.stored", limit=50
        )
        assert len(result.events) == 2
        assert result.next_cursor == "cursor-abc"

    def test_list_audit_events_with_time_range(self, client, mock_responses):
        """Test listing audit events with time range."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/audit",
            json={"events": [], "next_cursor": None},
            status=200,
        )
        result = client.list_audit_events(from_ts=1747000000, to_ts=1747500000)
        assert len(result.events) == 0

    def test_export_audit(self, client, mock_responses):
        """Test exporting audit log."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/audit/export",
            json={
                "data": '{"id":"evt-1"}\n{"id":"evt-2"}',
                "count": 2,
                "format": "jsonl",
            },
            status=200,
        )
        result = client.export_audit(
            format="jsonl", agent_id="agent-1", from_ts=1747000000
        )
        assert result.count == 2
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["format"] == "jsonl"
        assert req_body["agent_id"] == "agent-1"


class TestEntityExtraction:
    """Tests for entity extraction methods (CE-4)."""

    def test_extract_entities(self, client, mock_responses):
        """Test extracting entities from text."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/extract",
            json={
                "entities": [
                    {"text": "John", "type": "person", "start": 0, "end": 4},
                    {"text": "Acme Corp", "type": "org", "start": 18, "end": 27},
                ],
                "count": 2,
            },
            status=200,
        )
        result = client.extract_entities("John works at Acme Corp")
        assert len(result.entities) == 2
        assert result.entities[0].text == "John"
        assert result.entities[0].type == "person"

    def test_extract_entities_with_types(self, client, mock_responses):
        """Test extracting specific entity types."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memories/extract",
            json={"entities": [{"text": "Paris", "type": "location"}], "count": 1},
            status=200,
        )
        result = client.extract_entities(
            "Meeting in Paris", entity_types=["location"]
        )
        assert len(result.entities) == 1
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["entity_types"] == ["location"]

    def test_memory_entities(self, client, mock_responses):
        """Test getting entities for a stored memory."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/memory/entities/mem-1",
            json={
                "memory_id": "mem-1",
                "entities": [
                    {"text": "Alice", "type": "person"},
                    {"text": "2026-05-17", "type": "date"},
                ],
            },
            status=200,
        )
        result = client.memory_entities("mem-1")
        assert len(result.entities) == 2

    def test_configure_namespace_ner(self, client, mock_responses):
        """Test configuring NER for a namespace."""
        mock_responses.add(
            responses.PATCH,
            "http://localhost:3000/v1/namespaces/test-ns/config",
            json={
                "namespace": "test-ns",
                "extract_entities": True,
                "entity_types": ["person", "org", "location"],
            },
            status=200,
        )
        result = client.configure_namespace_ner(
            "test-ns",
            extract_entities=True,
            entity_types=["person", "org", "location"],
        )
        assert result["extract_entities"] is True

    def test_get_namespace_entity_config(self, client, mock_responses):
        """Test getting entity config for a namespace."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/test-ns/config",
            json={
                "namespace": "test-ns",
                "extract_entities": True,
                "entity_types": ["person", "org"],
            },
            status=200,
        )
        result = client.get_namespace_entity_config("test-ns")
        assert result["extract_entities"] is True


class TestExtractionProviders:
    """Tests for EXT-1 extraction provider methods."""

    def test_extract_text(self, client, mock_responses):
        """Test extracting entities via provider."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/extract",
            json={
                "entities": [{"text": "Bob", "type": "person"}],
                "provider": "gliner",
                "model": "urchade/gliner_multi-v2.1",
                "processing_time_ms": 45,
            },
            status=200,
        )
        result = client.extract_text("Bob is here", provider="gliner")
        assert result.provider == "gliner"
        assert len(result.entities) == 1

    def test_list_extract_providers(self, client, mock_responses):
        """Test listing extraction providers."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/extract/providers",
            json={
                "providers": [
                    {"name": "gliner", "models": ["urchade/gliner_multi-v2.1"]},
                    {"name": "openai", "models": ["gpt-4o"]},
                ]
            },
            status=200,
        )
        result = client.list_extract_providers()
        assert len(result) == 2
        assert result[0].name == "gliner"

    def test_configure_namespace_extractor(self, client, mock_responses):
        """Test setting namespace extractor provider."""
        mock_responses.add(
            responses.PATCH,
            "http://localhost:3000/v1/namespaces/test-ns/extractor",
            json={"provider": "openai", "model": "gpt-4o"},
            status=200,
        )
        result = client.configure_namespace_extractor("test-ns", "openai", model="gpt-4o")
        assert result["provider"] == "openai"


class TestMemoryPolicy:
    """Tests for COG-1 memory policy methods."""

    def test_get_memory_policy(self, client, mock_responses):
        """Test getting memory policy."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/test-ns/memory_policy",
            json={
                "working_ttl_hours": 4,
                "episodic_ttl_days": 30,
                "semantic_ttl_days": 365,
                "procedural_ttl_days": 730,
                "decay_strategy": "exponential",
            },
            status=200,
        )
        result = client.get_memory_policy("test-ns")
        assert result.working_ttl_hours == 4
        assert result.decay_strategy == "exponential"

    def test_set_memory_policy(self, client, mock_responses):
        """Test setting memory policy."""
        from dakera.models import MemoryPolicy

        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/namespaces/test-ns/memory_policy",
            json={
                "working_ttl_hours": 8,
                "episodic_ttl_days": 60,
                "semantic_ttl_days": 365,
                "procedural_ttl_days": 730,
                "decay_strategy": "linear",
            },
            status=200,
        )
        policy = MemoryPolicy(
            working_ttl_hours=8,
            episodic_ttl_days=60,
            decay_strategy="linear",
        )
        result = client.set_memory_policy("test-ns", policy)
        assert result.working_ttl_hours == 8
        assert result.decay_strategy == "linear"


class TestNamespaceKeys:
    """Tests for SEC-1 namespace key methods."""

    def test_create_namespace_key(self, client, mock_responses):
        """Test creating namespace-scoped API key."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/keys",
            json={
                "key_id": "key-1",
                "key": "dk_ns_abc123...",
                "name": "ci-reader",
                "namespace": "test-ns",
            },
            status=200,
        )
        result = client.create_namespace_key("test-ns", "ci-reader", expires_in_days=90)
        assert result.key_id == "key-1"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["name"] == "ci-reader"
        assert req_body["expires_in_days"] == 90

    def test_list_namespace_keys(self, client, mock_responses):
        """Test listing namespace keys."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/test-ns/keys",
            json={
                "keys": [
                    {"key_id": "key-1", "name": "ci-reader", "active": True},
                    {"key_id": "key-2", "name": "admin", "active": True},
                ]
            },
            status=200,
        )
        result = client.list_namespace_keys("test-ns")
        assert len(result.keys) == 2

    def test_delete_namespace_key(self, client, mock_responses):
        """Test revoking a namespace key."""
        mock_responses.add(
            responses.DELETE,
            "http://localhost:3000/v1/namespaces/test-ns/keys/key-1",
            json={"success": True, "message": "Key revoked"},
            status=200,
        )
        result = client.delete_namespace_key("test-ns", "key-1")
        assert result["success"] is True

    def test_get_namespace_key_usage(self, client, mock_responses):
        """Test getting namespace key usage."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/namespaces/test-ns/keys/key-1/usage",
            json={
                "key_id": "key-1",
                "total_requests": 5000,
                "avg_latency_ms": 3.2,
            },
            status=200,
        )
        result = client.get_namespace_key_usage("test-ns", "key-1")
        assert result.total_requests == 5000


class TestAPIKeys:
    """Tests for global API key methods."""

    def test_create_key(self, client, mock_responses):
        """Test creating an API key."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/keys",
            json={"id": "key-1", "name": "test-key", "key": "dk_abc123"},
            status=200,
        )
        result = client.create_key("test-key", permissions=["read", "write"])
        assert result["name"] == "test-key"

    def test_list_keys(self, client, mock_responses):
        """Test listing API keys."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/keys",
            json=[
                {"id": "key-1", "name": "admin", "active": True},
                {"id": "key-2", "name": "reader", "active": True},
            ],
            status=200,
        )
        result = client.list_keys()
        assert len(result) == 2

    def test_delete_key(self, client, mock_responses):
        """Test deleting an API key."""
        mock_responses.add(
            responses.DELETE,
            "http://localhost:3000/v1/keys/key-1",
            json={"deleted": True},
            status=200,
        )
        result = client.delete_key("key-1")
        assert result["deleted"] is True

    def test_deactivate_key(self, client, mock_responses):
        """Test deactivating an API key."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/keys/key-1/deactivate",
            json={"id": "key-1", "active": False},
            status=200,
        )
        result = client.deactivate_key("key-1")
        assert result["active"] is False

    def test_rotate_key(self, client, mock_responses):
        """Test rotating an API key."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/keys/key-1/rotate",
            json={"id": "key-1", "new_key": "dk_new_xyz"},
            status=200,
        )
        result = client.rotate_key("key-1")
        assert "new_key" in result

    def test_key_usage(self, client, mock_responses):
        """Test getting key usage."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/keys/key-1/usage",
            json={"key_id": "key-1", "total_requests": 10000, "last_used": "2026-05-17"},
            status=200,
        )
        result = client.key_usage("key-1")
        assert result["total_requests"] == 10000


class TestUpdateImportance:
    """Tests for update_importance method."""

    def test_update_importance_single(self, client, mock_responses):
        """Test updating importance for a single memory."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/importance",
            json={"memory_id": "mem-1", "importance": 0.95},
            status=200,
        )
        result = client.update_importance("agent-1", ["mem-1"], 0.95)
        assert result["importance"] == 0.95

    def test_update_importance_multiple(self, client, mock_responses):
        """Test updating importance for multiple memories."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/importance",
            json={"memory_id": "mem-1", "importance": 0.8},
            status=200,
        )
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/memory/importance",
            json={"memory_id": "mem-2", "importance": 0.8},
            status=200,
        )
        result = client.update_importance("agent-1", ["mem-1", "mem-2"], 0.8)
        assert len(result) == 2
        assert len(mock_responses.calls) == 2


class TestAdvancedSearch:
    """Tests for advanced search methods."""

    def test_multi_vector_search(self, client, mock_responses):
        """Test multi-vector search."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/multi-vector",
            json={
                "results": [
                    {"id": "vec-1", "score": 0.92},
                    {"id": "vec-2", "score": 0.88},
                ],
            },
            status=200,
        )
        result = client.multi_vector_search(
            "test-ns",
            positive=[[0.1, 0.2, 0.3]],
            negative=[[0.9, 0.8, 0.7]],
            top_k=5,
            mmr_lambda=0.7,
        )
        assert len(result["results"]) == 2
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["positive"] == [[0.1, 0.2, 0.3]]
        assert req_body["negative"] == [[0.9, 0.8, 0.7]]
        assert req_body["mmr_lambda"] == 0.7

    def test_unified_query(self, client, mock_responses):
        """Test unified query."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/unified-query",
            json={
                "results": [{"id": "r-1", "score": 0.9}],
                "fusion_method": "rrf",
            },
            status=200,
        )
        result = client.unified_query(
            "test-ns",
            vector=[0.1, 0.2, 0.3],
            text="hello",
            fusion_method="rrf",
            rerank=True,
        )
        assert len(result["results"]) == 1
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["fusion_method"] == "rrf"
        assert req_body["rerank"] is True

    def test_aggregate(self, client, mock_responses):
        """Test vector aggregation."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/aggregate",
            json={
                "groups": [
                    {"key": "category_a", "count": 100, "avg_score": 0.85},
                    {"key": "category_b", "count": 50, "avg_score": 0.72},
                ]
            },
            status=200,
        )
        result = client.aggregate(
            "test-ns",
            group_by="category",
            metrics=["count", "avg_score"],
            top_groups=5,
        )
        assert len(result["groups"]) == 2

    def test_export_vectors(self, client, mock_responses):
        """Test exporting vectors."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/export",
            json={
                "vectors": [
                    {"id": "v-1", "values": [0.1, 0.2]},
                    {"id": "v-2", "values": [0.3, 0.4]},
                ],
                "next_cursor": "cursor-xyz",
            },
            status=200,
        )
        result = client.export_vectors("test-ns", limit=100)
        assert len(result["vectors"]) == 2
        assert result["next_cursor"] == "cursor-xyz"

    def test_explain_query(self, client, mock_responses):
        """Test explain query."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/explain",
            json={
                "plan": {"type": "hnsw_scan", "ef": 100},
                "results": [{"id": "v-1", "score": 0.95}],
                "timing": {"total_ms": 2.5, "search_ms": 1.8},
            },
            status=200,
        )
        result = client.explain_query("test-ns", vector=[0.1, 0.2, 0.3], top_k=5)
        assert "plan" in result
        assert result["timing"]["total_ms"] == 2.5

    def test_upsert_columns(self, client, mock_responses):
        """Test column-format vector upsert."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/upsert-columns",
            json={"upserted_count": 3},
            status=200,
        )
        result = client.upsert_columns(
            "test-ns",
            ids=["v1", "v2", "v3"],
            vectors=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            attributes={"label": ["a", "b", "c"]},
            ttl_seconds=3600,
        )
        assert result["upserted_count"] == 3
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["ids"] == ["v1", "v2", "v3"]
        assert req_body["ttl_seconds"] == 3600


class TestCacheWarming:
    """Tests for cache warming methods."""

    def test_warm_cache(self, client, mock_responses):
        """Test warming cache."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/namespaces/test-ns/cache/warm",
            json={
                "entries_warmed": 100,
                "already_cached": 50,
                "elapsed_ms": 200,
            },
            status=200,
        )
        from dakera.models import WarmingPriority, WarmingTargetTier

        result = client.warm_cache(
            "test-ns",
            vector_ids=["v1", "v2"],
            priority=WarmingPriority.HIGH,
            target_tier=WarmingTargetTier.BOTH,
        )
        assert result.entries_warmed == 100
        assert result.already_cached == 50
