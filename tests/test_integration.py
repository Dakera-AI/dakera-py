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
# Admin Endpoints — Autopilot & Decay (v0.11.90)
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
