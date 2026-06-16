"""
Playground scenario integration tests for the Dakera Python SDK.

Validates the core store→recall→search→KG-link workflow that the playground
quickstart demonstrates. Tests are skipped unless DAKERA_TEST_URL is set.

Run:
    DAKERA_TEST_URL=http://localhost:3000 DAKERA_API_KEY=test-key \
        pytest tests/test_playground_integration.py -v
"""

import os
import uuid

import pytest

from dakera import DakeraClient

DAKERA_URL = os.environ.get("DAKERA_TEST_URL", "http://localhost:3000")
AGENT_ID = f"playground-integ-{uuid.uuid4().hex[:8]}"

pytestmark = pytest.mark.skipif(
    not os.environ.get("DAKERA_TEST_URL"),
    reason="DAKERA_TEST_URL not set — skipping playground integration tests",
)


@pytest.fixture(scope="module")
def client():
    c = DakeraClient(
        base_url=DAKERA_URL,
        api_key=os.environ.get("DAKERA_API_KEY", "test-key"),
    )
    yield c
    c.close()


class TestPlaygroundWorkflow:
    """End-to-end playground scenario: store → recall → search → KG link."""

    def test_step1_store_memories_with_tags(self, client):
        """Store two memories with tags and verify IDs are returned."""
        mem1 = client.store_memory(
            AGENT_ID,
            content="Dakera provides persistent, decay-weighted memory for AI agents.",
            memory_type="semantic",
            importance=0.9,
            tags=["dakera", "memory", "overview"],
        )
        mem1_id = mem1.get("id") or mem1.get("memory_id")
        assert mem1_id, "store_memory must return a non-empty memory ID"

        mem2 = client.store_memory(
            AGENT_ID,
            content="The recall API returns semantically similar memories ranked by relevance.",
            memory_type="semantic",
            importance=0.8,
            tags=["dakera", "recall", "api"],
        )
        mem2_id = mem2.get("id") or mem2.get("memory_id")
        assert mem2_id, "second store_memory must return a non-empty memory ID"

        # Stash on the module so later steps can use the IDs.
        TestPlaygroundWorkflow._mem1_id = mem1_id
        TestPlaygroundWorkflow._mem2_id = mem2_id

    def test_step2_recall_by_semantic_query(self, client):
        """Recall memories by semantic query and get at least one result."""
        recalled = client.recall(AGENT_ID, "How does Dakera memory work?", top_k=5)
        assert hasattr(recalled, "memories"), "recall response must have .memories"
        assert len(recalled.memories) >= 1, "expected at least one recalled memory"
        # Each memory must carry a score and content.
        for m in recalled.memories:
            assert m.content, "recalled memory must have content"

    def test_step3_search_with_memory_type_filter(self, client):
        """Search memories filtered by memory_type=semantic."""
        results = client.search_memories(
            AGENT_ID,
            query="memory API",
            memory_type="semantic",
            top_k=5,
        )
        assert isinstance(results, list), "search_memories must return a list"
        assert len(results) >= 1, "expected at least one filtered result"
        for m in results:
            assert m.get("content"), "each search result must have content"

    def test_step4_knowledge_graph_link(self, client):
        """Link two stored memories with a related_to edge."""
        mem1_id = getattr(TestPlaygroundWorkflow, "_mem1_id", None)
        mem2_id = getattr(TestPlaygroundWorkflow, "_mem2_id", None)
        if not mem1_id or not mem2_id:
            pytest.skip("memory IDs unavailable — step1 must run first")

        link = client.memory_link(mem1_id, mem2_id, edge_type="related_to")
        assert link.edge is not None, "memory_link must return an edge object"
        assert link.edge.edge_type == "related_to", (
            f"expected edge_type=related_to, got {link.edge.edge_type}"
        )
