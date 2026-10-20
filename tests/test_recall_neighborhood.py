"""Recall API after server v0.12.2 (Dakera-AI/dakera#917): ``neighborhood_mode``,
``include_summaries``, ``context_memories`` and the result fields ``matched_text``,
``matched_ids``, ``retrieval``, ``context_of`` — and the same calls against a server
that predates them (no new fields in the answer)."""

import json

import httpx
import pytest
import responses

from dakera import AsyncDakeraClient, DakeraClient, RecallResponse
from dakera.models import RETRIEVAL_NEIGHBORHOOD, RecalledMemory, RetryConfig

BASE = "http://localhost:3000"


def _mem(mid: str, content: str = "c") -> dict:
    return {"id": mid, "content": content, "memory_type": "episodic", "importance": 0.5}


# What the engine answers (`RecallResponse`, result envelope `RecallResult`).
INLINE_ANSWER = {
    "memories": [
        {
            "memory": _mem("m1", "port 9000"),
            "score": 0.9,
            "smart_score": 0.8,
            "matched_ids": ["m1_s0", "m1_s2"],
            "matched_text": "The port is 9000.",
        },
        {"memory": _mem("m2"), "score": 0.7, "smart_score": 0.6},
        {"memory": _mem("n1"), "score": 0.42, "retrieval": "neighborhood"},
    ],
    "query_embedding_time_ms": 3,
    "search_time_ms": 9,
    "rerank": {"applied": True},
    "rerank_report": {"applied": True},
}

SEPARATE_ANSWER = {
    "memories": [
        {"memory": _mem("m1"), "score": 0.9, "smart_score": 0.8},
        {"memory": _mem("m2"), "score": 0.7, "smart_score": 0.6},
    ],
    "context_memories": [
        {
            "memory": _mem("n1"),
            "score": 0.42,
            "retrieval": "neighborhood",
            "context_of": "m1",
        },
    ],
    "query_embedding_time_ms": 3,
    "search_time_ms": 9,
    "rerank": {"applied": False},
    "rerank_report": {"applied": False},
}

# A v0.12.2 server: none of the new fields.
OLD_ANSWER = {
    "memories": [
        {"memory": _mem("m1"), "score": 0.9, "smart_score": 0.8},
        {"memory": _mem("n1"), "score": 0.42},
    ],
    "associated_memories": [{"memory": _mem("a1"), "score": 0.3, "depth": 2}],
    "query_embedding_time_ms": 3,
    "search_time_ms": 9,
}


class TestParsing:
    def test_inline_items_are_marked(self):
        r = RecallResponse.from_dict(INLINE_ANSWER)
        assert [m.id for m in r.memories] == ["m1", "m2", "n1"]
        first = r.memories[0]
        assert first.matched_ids == ["m1_s0", "m1_s2"]
        assert first.matched_text == "The port is 9000."
        assert first.retrieval is None and first.context_of is None
        assert first.score == 0.8  # smart_score ranks
        neighbour = r.memories[2]
        assert neighbour.retrieval == RETRIEVAL_NEIGHBORHOOD == "neighborhood"
        assert neighbour.is_neighborhood
        assert [m.id for m in r.ranked] == ["m1", "m2"]
        assert r.context_memories is None

    def test_separate_context_memories(self):
        r = RecallResponse.from_dict(SEPARATE_ANSWER)
        assert [m.id for m in r.memories] == ["m1", "m2"]
        assert r.ranked == r.memories
        assert r.context_memories is not None
        (ctx,) = r.context_memories
        assert ctx.id == "n1"
        assert ctx.context_of == "m1"
        assert ctx.retrieval == "neighborhood"
        assert ctx.is_neighborhood

    def test_rank_score_is_the_ranking_score(self):
        # #940: score inverts down the list; rank_score does not.
        answer = {
            "memories": [
                {"memory": _mem("m1"), "score": 0.0001, "smart_score": 0.3, "rank_score": 0.62},
                {"memory": _mem("m2"), "score": 0.0023, "smart_score": 0.5, "rank_score": 0.41},
                {
                    "memory": _mem("n1"),
                    "score": 0.9,
                    "retrieval": "neighborhood",
                    "rank_score": 0.0,
                },
            ],
            "query_embedding_time_ms": 1,
            "search_time_ms": 1,
        }
        r = RecallResponse.from_dict(answer)
        assert [m.rank_score for m in r.memories] == [0.62, 0.41, 0.0]
        assert [m.score for m in r.memories] == [0.62, 0.41, 0.0]
        scores = [m.score for m in r.memories]
        assert scores == sorted(scores, reverse=True)
        # Older servers send none: score falls back to smart_score.
        assert RecallResponse.from_dict(INLINE_ANSWER).memories[0].rank_score is None

    def test_old_server_answer(self):
        r = RecallResponse.from_dict(OLD_ANSWER)
        assert [m.id for m in r.memories] == ["m1", "n1"]
        for m in r.memories:
            assert m.matched_text is None and m.matched_ids is None
            assert m.retrieval is None and m.context_of is None
            assert not m.is_neighborhood
        assert r.context_memories is None
        assert r.ranked == r.memories
        # The envelope's depth (KG-3) is kept on associated memories.
        assert r.associated_memories is not None
        assert r.associated_memories[0].depth == 2

    def test_flat_result_still_parses(self):
        m = RecalledMemory.from_dict({**_mem("x"), "score": 0.5, "context_of": "y"})
        assert m.context_of == "y" and m.is_neighborhood


@pytest.fixture
def rsps():
    with responses.RequestsMock() as r:
        yield r


@pytest.fixture
def client():
    return DakeraClient(BASE, retry_config=RetryConfig(max_retries=1))


class TestSyncRecall:
    def test_new_fields_sent_when_set(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", json=SEPARATE_ANSWER)
        r = client.recall(
            "agent-1",
            "which port?",
            top_k=2,
            neighborhood_mode="separate",
            include_summaries=False,
            session_id="s1",
        )
        body = json.loads(rsps.calls[0].request.body)
        assert body["neighborhood_mode"] == "separate"
        assert body["include_summaries"] is False
        assert body["session_id"] == "s1"
        assert body["top_k"] == 2
        assert len(r.memories) <= 2
        assert r.context_memories is not None and r.context_memories[0].context_of == "m1"

    def test_new_fields_omitted_by_default(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", json=OLD_ANSWER)
        r = client.recall("agent-1", "q")
        body = json.loads(rsps.calls[0].request.body)
        for key in ("neighborhood_mode", "include_summaries", "session_id", "neighborhood"):
            assert key not in body
        assert r.context_memories is None
        assert r.memories[0].matched_ids is None

    def test_neighborhood_false_still_sent(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", json=OLD_ANSWER)
        client.recall("agent-1", "q", neighborhood=False, neighborhood_mode="inline")
        body = json.loads(rsps.calls[0].request.body)
        assert body["neighborhood"] is False and body["neighborhood_mode"] == "inline"

    def test_search_results_carry_matched_fields(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/search",
            json={
                "memories": [
                    {
                        "memory": _mem("m1"),
                        "score": 0.9,
                        "smart_score": 0.7,
                        "matched_ids": ["m1_s1"],
                        "matched_text": "best sentence",
                    }
                ],
                "total_count": 1,
                "rerank_report": {"applied": False},
                "effective_top_k": 10,
            },
        )
        (item,) = client.search_memories("agent-1", "q", session_id="s1")
        assert item["id"] == "m1" and item["score"] == 0.9
        assert item["smart_score"] == 0.7
        assert item["matched_ids"] == ["m1_s1"]
        assert item["matched_text"] == "best sentence"
        assert json.loads(rsps.calls[0].request.body)["session_id"] == "s1"

    def test_search_old_server(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/search",
            json={"memories": [{"memory": _mem("m1"), "score": 0.9}], "total_count": 1},
        )
        (item,) = client.search_memories("agent-1", "q")
        assert item == {**_mem("m1"), "score": 0.9}


def _async(handler):
    c = AsyncDakeraClient(BASE, retry_config=RetryConfig(max_retries=1))
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


class TestAsyncRecall:
    async def test_new_fields_sent_and_parsed(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=INLINE_ANSWER)

        c = _async(handler)
        r = await c.recall("agent-1", "q", neighborhood_mode="inline", include_summaries=True)
        assert seen[0]["neighborhood_mode"] == "inline"
        assert seen[0]["include_summaries"] is True
        assert r.memories[2].retrieval == "neighborhood"
        assert r.memories[0].matched_ids == ["m1_s0", "m1_s2"]
        assert [m.id for m in r.ranked] == ["m1", "m2"]

    async def test_separate_and_old_server(self):
        answers = [SEPARATE_ANSWER, OLD_ANSWER]
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=answers[len(seen) - 1])

        c = _async(handler)
        r = await c.recall("agent-1", "q", top_k=2, neighborhood_mode="separate")
        assert r.context_memories is not None and r.context_memories[0].context_of == "m1"
        old = await c.recall("agent-1", "q")
        assert "neighborhood_mode" not in seen[1] and "include_summaries" not in seen[1]
        assert old.context_memories is None
        assert all(m.retrieval is None for m in old.memories)

    async def test_search_results_carry_matched_fields(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "memories": [
                        {
                            "memory": _mem("m1"),
                            "score": 0.5,
                            "matched_text": "t",
                            "matched_ids": ["s"],
                        }
                    ],
                    "total_count": 1,
                },
            )

        (item,) = await _async(handler).search_memories("agent-1", "q")
        assert item["matched_text"] == "t" and item["matched_ids"] == ["s"]
