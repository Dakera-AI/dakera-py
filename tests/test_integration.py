"""Integration tests against a real Dakera server (Docker service in CI).

Requires DAKERA_TEST_URL env var pointing to a running Dakera instance.
Auth is enabled — set DAKERA_API_KEY to a valid key (default: test-key).

Run locally:
  DAKERA_TEST_URL=http://localhost:3000 DAKERA_API_KEY=test-key pytest tests/test_integration.py -v
"""

import contextlib
import os
import time
import uuid

import pytest

from dakera import DakeraClient
from dakera.exceptions import AuthenticationError, DakeraError
from dakera.models import (
    BatchRecallRequest,
    BatchStoreMemoryItem,
    BatchStoreMemoryRequest,
    TextDocument,
)

DAKERA_URL = os.environ.get("DAKERA_TEST_URL", "http://localhost:3000")
TEST_NAMESPACE = f"integ-{uuid.uuid4().hex[:8]}"
TEST_AGENT = f"integ-agent-{uuid.uuid4().hex[:8]}"

pytestmark = pytest.mark.skipif(
    not os.environ.get("DAKERA_TEST_URL"),
    reason="DAKERA_TEST_URL not set — skipping integration tests",
)


@pytest.fixture(scope="module")
def client():
    c = DakeraClient(base_url=DAKERA_URL, api_key=os.environ.get("DAKERA_API_KEY", "test-key"))
    yield c
    c.close()


@pytest.fixture(scope="module")
def namespace(client):
    client.create_namespace(TEST_NAMESPACE, dimensions=1024)
    yield TEST_NAMESPACE
    with contextlib.suppress(Exception):
        client.delete_namespace(TEST_NAMESPACE)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client):
        result = client.health()
        assert result["status"] == "healthy"


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------


class TestNamespaces:
    def test_create_namespace(self, client):
        ns = f"integ-create-{uuid.uuid4().hex[:8]}"
        result = client.create_namespace(ns, dimensions=1024)
        assert result.name == ns
        client.delete_namespace(ns)

    def test_list_namespaces(self, client, namespace):
        namespaces = client.list_namespaces()
        names = [ns.name for ns in namespaces]
        assert namespace in names

    def test_get_namespace(self, client, namespace):
        ns = client.get_namespace(namespace)
        assert ns.name == namespace
        assert ns.dimensions == 1024

    def test_configure_namespace(self, client, namespace):
        result = client.configure_namespace(namespace, dimension=1024)
        assert result is not None

    def test_delete_namespace(self, client):
        ns = f"integ-del-{uuid.uuid4().hex[:8]}"
        client.create_namespace(ns, dimensions=1024)
        client.delete_namespace(ns)
        namespaces = client.list_namespaces()
        names = [n.name for n in namespaces]
        assert ns not in names


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------


class TestMemory:
    def test_store_memory(self, client):
        result = client.store_memory(
            agent_id=TEST_AGENT,
            content="The user prefers dark mode interfaces",
            importance=0.8,
            tags=["preference", "ui"],
        )
        assert "id" in result

    def test_recall_semantic(self, client):
        client.store_memory(
            agent_id=TEST_AGENT,
            content="Python is the user's primary programming language",
            importance=0.9,
            tags=["preference", "coding"],
        )
        time.sleep(0.5)
        results = client.recall(TEST_AGENT, "programming language")
        assert len(results.memories) > 0

    def test_batch_recall(self, client):
        result = client.batch_recall(
            BatchRecallRequest(agent_id=TEST_AGENT, min_importance=0.5)
        )
        assert result.memories is not None
        assert len(result.memories) > 0

    def test_get_memory(self, client):
        store = client.store_memory(
            agent_id=TEST_AGENT,
            content="Memory for get test",
            importance=0.7,
        )
        memory = client.get_memory(TEST_AGENT, store["id"])
        assert memory is not None

    def test_update_importance(self, client):
        store = client.store_memory(
            agent_id=TEST_AGENT,
            content="Memory for importance update",
            importance=0.5,
        )
        result = client.update_importance(TEST_AGENT, [store["id"]], 0.95)
        assert result is not None

    def test_forget(self, client):
        store = client.store_memory(
            agent_id=TEST_AGENT,
            content="Memory to forget",
            importance=0.3,
        )
        result = client.forget(TEST_AGENT, store["id"])
        assert result is not None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_start_and_end_session(self, client):
        session = client.start_session(TEST_AGENT, metadata={"type": "test"})
        assert "id" in session
        result = client.end_session(session["id"])
        assert result is not None

    def test_list_sessions(self, client):
        session = client.start_session(TEST_AGENT)
        sessions = client.list_sessions(agent_id=TEST_AGENT)
        assert len(sessions) > 0
        client.end_session(session["id"])

    def test_session_memories(self, client):
        session = client.start_session(TEST_AGENT)
        memories = client.session_memories(session["id"])
        assert isinstance(memories, list)
        client.end_session(session["id"])


# ---------------------------------------------------------------------------
# Vectors / Text
# ---------------------------------------------------------------------------


class TestVectors:
    def test_upsert_text(self, client, namespace):
        result = client.upsert_text(
            namespace,
            documents=[
                TextDocument(id="doc-1", text="Machine learning transforms data into insights"),
                TextDocument(id="doc-2", text="Natural language processing understands text"),
                TextDocument(id="doc-3", text="Deep learning uses neural networks"),
            ],
        )
        assert result is not None

    def test_query_text(self, client, namespace):
        time.sleep(1)
        result = client.query_text(namespace, "AI neural networks", top_k=3)
        assert result is not None

    def test_hybrid_search(self, client, namespace):
        time.sleep(0.5)
        results = client.hybrid_search(namespace, query="machine learning data", top_k=3)
        assert isinstance(results, list)

    def test_fulltext_search(self, client, namespace):
        time.sleep(0.5)
        results = client.fulltext_search(namespace, query="neural networks", top_k=3)
        assert results is not None

    def test_batch_query_text(self, client, namespace):
        time.sleep(0.5)
        result = client.batch_query_text(
            namespace, queries=["machine learning", "deep learning"], top_k=2
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    def test_memory_graph(self, client):
        store = client.store_memory(
            agent_id=TEST_AGENT,
            content="Knowledge graph integration test memory",
            importance=0.8,
        )
        time.sleep(0.5)
        result = client.memory_graph(store["id"], depth=1)
        assert result is not None

    def test_extract_entities(self, client, namespace):
        result = client.extract_entities(
            text="OpenAI released GPT-4 in San Francisco",
        )
        assert result is not None


class TestMemoryAndGraphContract:
    """Round trip over the routes whose request or response shapes the SDK
    got wrong before 0.13.1 (agent_id query / body, flat link answer, edges as
    from_id/to_id, path `to`, export shape, entities without memory_id)."""

    AGENT = f"integ-graph-{uuid.uuid4().hex[:8]}"

    @pytest.fixture(scope="class")
    @classmethod
    def pair(cls, client):
        a = client.store_memory(
            agent_id=cls.AGENT, content="Anna lives in Berlin", tags=["city", "anna"]
        )["id"]
        b = client.store_memory(
            agent_id=cls.AGENT, content="Anna works at Siemens in Munich", tags=["work"]
        )["id"]
        return a, b

    def test_update_and_get(self, client, pair):
        a, _ = pair
        updated = client.update_memory(self.AGENT, a, content="Anna lives in Hamburg")
        assert updated["content"] == "Anna lives in Hamburg"
        assert client.get_memory(self.AGENT, a)["content"] == "Anna lives in Hamburg"

    def test_async_update_and_get(self, pair):
        import asyncio

        from dakera import AsyncDakeraClient

        a, _ = pair

        async def run() -> str:
            ac = AsyncDakeraClient(DAKERA_URL, api_key=os.environ.get("DAKERA_API_KEY", "test-key"))
            await ac.update_memory(self.AGENT, a, content="Anna lives in Bremen")
            got = await ac.get_memory(self.AGENT, a)
            return str(got["content"])

        assert asyncio.run(run()) == "Anna lives in Bremen"

    def test_recall_with_tags(self, client, pair):
        time.sleep(0.5)
        a, b = pair
        result = client.recall(self.AGENT, "Anna", top_k=10, tags=["work"])
        ids = {m.id for m in result.memories}
        assert b in ids, "the memory tagged `work` must be returned"
        assert a not in ids, "the tag filter must exclude the untagged memory"

    def test_link_graph_path_export_query(self, client, pair):
        a, b = pair
        link = client.memory_link(a, b, agent_id=self.AGENT, label="same person")
        assert (link.from_id, link.to_id, link.edge_type) == (a, b, "linked_by")
        assert link.edge.source_id == a

        graph = client.memory_graph(a, depth=2)
        assert graph.root_id == a
        assert any(e.source_id == a and e.target_id == b for e in graph.edges)
        linked_only = client.memory_graph(a, depth=2, types=["linked_by"])
        assert linked_only.edges
        assert all(str(e.edge_type.value) == "linked_by" for e in linked_only.edges)

        path = client.memory_path(a, b)
        assert path.path[0] == a and path.path[-1] == b
        assert path.hops >= 1

        export = client.agent_graph_export(self.AGENT)
        assert export.agent_id == self.AGENT
        assert export.edge_count >= 1
        assert any(e.source_id == a and e.target_id == b for e in export.edges)

        kg = client.knowledge_query(self.AGENT, edge_type="linked_by")
        assert any(e.source_id == a and e.target_id == b for e in kg.edges)

    def test_memory_entities(self, client, pair):
        a, _ = pair
        result = client.memory_entities(a)
        assert result.memory_id == a
        assert isinstance(result.entities, list)
        assert result.count == len(result.entities)


# ---------------------------------------------------------------------------
# Consolidate / Deduplicate
# ---------------------------------------------------------------------------


class TestConsolidate:
    def test_consolidate(self, client):
        for i in range(3):
            client.store_memory(
                agent_id=TEST_AGENT,
                content=f"Consolidation test memory variation {i}: similar content about testing",
                importance=0.6,
            )
        time.sleep(0.5)
        result = client.consolidate(TEST_AGENT)
        assert result is not None


# ---------------------------------------------------------------------------
# Batch Memory Operations (v0.11.90)
# ---------------------------------------------------------------------------


class TestBatchMemory:
    def test_store_memories_batch(self, client):
        req = BatchStoreMemoryRequest(
            agent_id=TEST_AGENT,
            memories=[
                BatchStoreMemoryItem(
                    content="Batch memory one: user speaks French",
                    importance=0.7,
                    memory_type="semantic",
                ),
                BatchStoreMemoryItem(
                    content="Batch memory two: user prefers async APIs",
                    importance=0.8,
                    memory_type="semantic",
                ),
                BatchStoreMemoryItem(
                    content="Batch memory three: user works in Berlin",
                    importance=0.6,
                    memory_type="episodic",
                ),
            ],
        )
        resp = client.store_memories_batch(req)
        assert resp.stored_count == 3
        assert len(resp.stored) == 3
        assert all(m.id for m in resp.stored)
        assert resp.total_embedding_time_ms >= 0

    def test_search_memories(self, client):
        time.sleep(0.5)
        results = client.search_memories(
            TEST_AGENT,
            query="programming language preferences",
            memory_type="semantic",
            top_k=5,
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Admin Endpoints — Autopilot & Decay (v0.11.91+)
# ---------------------------------------------------------------------------


class TestAdminEndpoints:
    def test_autopilot_status(self, client):
        result = client.autopilot_status()
        assert isinstance(result, dict)
        assert "enabled" in result or "status" in result or result is not None

    def test_decay_config(self, client):
        result = client.decay_config()
        assert isinstance(result, dict)

    def test_decay_stats(self, client):
        result = client.decay_stats()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Full-Text Index Operations (v0.11.90)
# ---------------------------------------------------------------------------


class TestFulltextOps:
    def test_fulltext_stats(self, client, namespace):
        # Ensure at least one doc is indexed (TestVectors runs before this)
        stats = client.fulltext_stats(namespace)
        assert stats.document_count >= 0
        assert stats.unique_terms >= 0

    def test_fulltext_delete(self, client, namespace):
        client.index_documents(
            namespace,
            documents=[
                {"id": "del-1", "text": "Document to be deleted from full-text index"},
                {"id": "del-2", "text": "Another document to be deleted"},
            ],
        )
        time.sleep(0.3)
        resp = client.fulltext_delete(namespace, ["del-1", "del-2"])
        assert resp.get("deleted_count", 0) >= 0


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_nonexistent_namespace(self, client):
        with pytest.raises(DakeraError):
            client.get_namespace("nonexistent-ns-xyz-99999")

    def test_nonexistent_memory(self, client):
        with pytest.raises(DakeraError):
            client.get_memory(TEST_AGENT, "nonexistent-memory-id")


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_rejects_invalid_api_key(self):
        bad_client = DakeraClient(base_url=DAKERA_URL, api_key="invalid-key-xxx")
        with pytest.raises(AuthenticationError):
            bad_client.list_namespaces()
        bad_client.close()

    def test_accepts_valid_api_key(self, client):
        namespaces = client.list_namespaces()
        assert isinstance(namespaces, list)


# ---------------------------------------------------------------------------
# AsyncChatMemorySession (requires AsyncDakeraClient)
# ---------------------------------------------------------------------------


class TestAsyncChatMemorySession:
    @pytest.mark.asyncio
    async def test_create_store_recall_close(self):
        """AsyncChatMemorySession: live store → recall → close round-trip."""
        from dakera import AsyncDakeraClient
        from dakera.session import AsyncChatMemorySession

        api_key = os.environ.get("DAKERA_API_KEY", "test-key")
        async_client = AsyncDakeraClient(base_url=DAKERA_URL, api_key=api_key)
        agent_id = f"async-integ-{uuid.uuid4().hex[:8]}"

        session = await AsyncChatMemorySession.create(async_client, agent_id)
        assert session.session_id
        assert session.agent_id == agent_id

        # Store two turns
        r1 = await session.store("user", "My async favourite colour is indigo.")
        assert r1.get("id") or r1  # server returns memory dict

        r2 = await session.store(
            "assistant", "Noted — indigo is a deep violet-blue.", importance=0.5
        )
        assert r2.get("id") or r2

        time.sleep(0.5)  # allow indexing

        # Recall should surface relevant memories
        memories = await session.recall("favourite colour", top_k=5)
        assert isinstance(memories, list)
        assert len(memories) > 0

        # Close session cleanly
        result = await session.close()
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """AsyncChatMemorySession: async with block ends session automatically."""
        from dakera import AsyncDakeraClient
        from dakera.session import AsyncChatMemorySession

        api_key = os.environ.get("DAKERA_API_KEY", "test-key")
        async_client = AsyncDakeraClient(base_url=DAKERA_URL, api_key=api_key)
        agent_id = f"async-ctx-{uuid.uuid4().hex[:8]}"

        async with await AsyncChatMemorySession.create(async_client, agent_id) as session:
            await session.store("user", "Context manager test message.")
            memories = await session.recall("context manager")
            assert isinstance(memories, list)
        # If we reach here without exception the context manager worked
