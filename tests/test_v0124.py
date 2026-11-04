"""Server v0.12.4 wire changes: forget by description or by a user's request and
``/v1/memory/forget/detect``, restorable forgets, the store's ``forget_request``,
batch ``forget_requests`` / ``duplicates``, recall ``rerank_head`` /
``importance_weighted`` / ``instructions`` / ``degraded``, the memory's strength
projection, async imports, the entity config's ``source``, the memory policy's
``stored``, link labels, extraction ``extractor`` / ``warnings``, the failure
lists of dedup, the cross-agent network, compression and TTL cleanup, the
autopilot ``consolidation_mode``, cache-warm hints, and ``202`` / ``503`` +
``Retry-After`` answers — each also against an older server that sends none of
the new fields."""

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import responses

from dakera import (
    AsyncDakeraClient,
    DakeraClient,
    ForgetDetectResponse,
    ForgetResponse,
    RecallResponse,
)
from dakera.exceptions import ConflictError, ServiceUnavailableError
from dakera.models import (
    BatchStoreMemoryResponse,
    CompressResponse,
    CrossAgentNetworkResponse,
    DeduplicatedListResponse,
    DeduplicateResponse,
    DerivationStatus,
    EntityExtractionResponse,
    GraphEdge,
    GraphLinkResponse,
    MemoryPolicy,
    RetryConfig,
    TtlCleanupResponse,
    WarmCacheRequest,
)

BASE = "http://localhost:3000"


def _mem(mid: str = "m1", content: str = "c") -> dict:
    return {
        "id": mid,
        "agent_id": "agent-1",
        "content": content,
        "memory_type": "episodic",
        "importance": 0.5,
        "tags": [],
        "created_at": 1_790_000_000,
    }


@pytest.fixture
def rsps():
    with responses.RequestsMock() as r:
        yield r


@pytest.fixture
def client():
    return DakeraClient(BASE, retry_config=RetryConfig(max_retries=1))


def _async(handler, retries: int = 1):
    c = AsyncDakeraClient(BASE, retry_config=RetryConfig(max_retries=retries))
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


INTENT = {"probability": 0.97, "threshold": 0.5, "is_request": True}


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    def test_forget_about_sends_only_what_is_set(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/forget",
            json={"deleted_count": 2, "forgotten_ids": ["m2", "m1"]},
        )
        resp = client.forget_memories(
            "agent-1", about="I love sushi", min_score=0.6, limit=5, until=1_790_000_000
        )
        body = json.loads(rsps.calls[0].request.body)
        assert body == {
            "agent_id": "agent-1",
            "about": "I love sushi",
            "min_score": 0.6,
            "limit": 5,
            "until": 1_790_000_000,
        }
        assert isinstance(resp, ForgetResponse)
        assert resp.deleted_count == 2
        assert resp.forgotten_ids == ["m2", "m1"]
        assert resp.intent is None

    def test_forget_request_parses_intent(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/forget",
            json={
                "deleted_count": 0,
                "forgotten_ids": ["m3"],
                "intent": {**INTENT, "sentence": "forget my address"},
            },
        )
        resp = client.forget_memories(
            "agent-1", request="... and forget my address", dry_run=True, restorable=True
        )
        body = json.loads(rsps.calls[0].request.body)
        assert body["request"] == "... and forget my address"
        assert body["dry_run"] is True and body["restorable"] is True
        assert "about" not in body
        assert resp.intent is not None
        assert resp.intent.is_request and resp.intent.probability == pytest.approx(0.97)
        assert resp.intent.sentence == "forget my address"

    def test_forget_by_filters(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/forget", json={"deleted_count": 4})
        resp = client.forget_memories(
            "agent-1", session_id="s1", tags=["t"], memory_type="working", below_importance=0.2
        )
        body = json.loads(rsps.calls[0].request.body)
        assert body == {
            "agent_id": "agent-1",
            "session_id": "s1",
            "tags": ["t"],
            "memory_type": "working",
            "below_importance": 0.2,
        }
        # An older server answers deleted_count alone.
        assert resp.deleted_count == 4 and resp.forgotten_ids is None and resp.intent is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"about": "a", "request": "b"},
            {"about": "a", "min_score": 1.5},
            {"about": "a", "limit": 0},
            {"about": "a", "limit": 51},
            {"about": "x" * 8001},
            {"request": "x" * 8001},
        ],
    )
    def test_invalid_selection_refused_before_sending(self, client, kwargs):
        with pytest.raises(ValueError):
            client.forget_memories("agent-1", **kwargs)

    def test_forget_restorable(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/forget", json={"deleted_count": 1})
        client.forget("agent-1", "m1", restorable=True)
        assert json.loads(rsps.calls[0].request.body) == {
            "agent_id": "agent-1",
            "memory_ids": ["m1"],
            "restorable": True,
        }

    def test_forget_default_body_unchanged(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/forget", json={"deleted_count": 1})
        client.forget("agent-1", "m1")
        assert json.loads(rsps.calls[0].request.body) == {
            "agent_id": "agent-1",
            "memory_ids": ["m1"],
        }

    def test_forget_without_reranker_is_503(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/forget",
            status=503,
            json={"error": "reranker unavailable", "code": "SERVICE_UNAVAILABLE"},
            headers={"Retry-After": "5"},
        )
        with pytest.raises(ServiceUnavailableError) as exc:
            client.forget_memories("agent-1", about="sushi")
        assert exc.value.retry_after == 5

    def test_detect(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/forget/detect",
            json={
                "model": "multilingual-minilm",
                "threshold": 0.5,
                "sentence_threshold": 0.7,
                "results": [
                    {"probability": 0.02, "is_request": False},
                    {"probability": 0.91, "is_request": True, "sentence": "Oublie ça."},
                ],
            },
        )
        resp = client.detect_forget_requests(["hello", "Bonjour. Oublie ça."])
        assert json.loads(rsps.calls[0].request.body) == {"texts": ["hello", "Bonjour. Oublie ça."]}
        assert isinstance(resp, ForgetDetectResponse)
        assert resp.model == "multilingual-minilm"
        assert resp.sentence_threshold == pytest.approx(0.7)
        assert [r.is_request for r in resp.results] == [False, True]
        assert resp.results[1].sentence == "Oublie ça."

    @pytest.mark.parametrize("texts", [[], ["x"] * 65])
    def test_detect_bounds(self, client, texts):
        with pytest.raises(ValueError):
            client.detect_forget_requests(texts)

    def test_detect_refuses_a_string(self, client):
        with pytest.raises(TypeError):
            client.detect_forget_requests("forget that")  # type: ignore[arg-type]

    async def test_async_forget_and_detect(self):
        seen = []

        def handler(request):
            seen.append((request.url.path, json.loads(request.content)))
            if request.url.path.endswith("/detect"):
                return httpx.Response(
                    200,
                    json={
                        "model": "m",
                        "threshold": 0.5,
                        "sentence_threshold": 0.7,
                        "results": [{"probability": 0.9, "is_request": True}],
                    },
                )
            return httpx.Response(200, json={"deleted_count": 1, "intent": INTENT})

        c = _async(handler)
        det = await c.detect_forget_requests(["forget my job"])
        resp = await c.forget_memories("agent-1", request="forget my job", limit=3)
        await c.forget("agent-1", "m1", restorable=True)
        assert det.results[0].is_request
        assert resp.intent is not None and resp.intent.is_request
        assert seen[1] == (
            "/v1/memory/forget",
            {"agent_id": "agent-1", "request": "forget my job", "limit": 3},
        )
        assert seen[2][1]["restorable"] is True
        with pytest.raises(ValueError):
            await c.forget_memories("agent-1", about="a", request="b")


# ---------------------------------------------------------------------------
# store / batch store
# ---------------------------------------------------------------------------


class TestStoreReadsForgetRequests:
    def test_store_forget_request(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/store",
            json={"memory": _mem(), "embedding_time_ms": 3, "forget_request": INTENT},
        )
        stored = client.store_memory("agent-1", "please forget my address")
        assert stored["forget_request"] == INTENT

    def test_store_without_forget_request(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/store",
            json={"memory": _mem(), "embedding_time_ms": 3},
        )
        assert "forget_request" not in client.store_memory("agent-1", "c")

    def test_batch_forget_requests_and_duplicates(self):
        resp = BatchStoreMemoryResponse.from_dict(
            {
                "stored": [_mem("m1"), _mem("m2")],
                "stored_count": 2,
                "total_embedding_time_ms": 4,
                "forget_requests": [{"id": "m2", **INTENT, "sentence": "forget it"}],
                "duplicates": [{"index": 2, "existing_id": "m0"}],
            }
        )
        assert resp.forget_requests[0].id == "m2"
        assert resp.forget_requests[0].intent.is_request
        assert resp.forget_requests[0].intent.sentence == "forget it"
        assert resp.duplicates[0].index == 2 and resp.duplicates[0].existing_id == "m0"

    def test_batch_from_an_older_server(self):
        resp = BatchStoreMemoryResponse.from_dict(
            {"stored": [], "stored_count": 0, "total_embedding_time_ms": 0}
        )
        assert resp.forget_requests == [] and resp.duplicates == []


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


RECALL_ANSWER = {
    "memories": [{"memory": _mem("m1"), "score": 0.9, "rank_score": 0.9}],
    "instructions": [
        {
            "memory": _mem("i1", "From now on, answer in French"),
            "score": 0.8,
            "retrieval": "instruction",
        }
    ],
    "degraded": ["graph: store unreachable"],
}


class TestRecall:
    def test_new_fields_sent_only_when_set(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", json={"memories": []})
        client.recall("agent-1", "q")
        body = json.loads(rsps.calls[0].request.body)
        for key in ("rerank_head", "importance_weighted", "include_associated", "top_k"):
            assert key not in body

    def test_rerank_head_and_importance_weighted(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", json={"memories": []})
        client.recall("agent-1", "q", top_k=25, rerank_head=0, importance_weighted=False)
        body = json.loads(rsps.calls[0].request.body)
        assert body["rerank_head"] == 0
        assert body["importance_weighted"] is False

    @pytest.mark.parametrize("bad", [-1, True])
    def test_bad_rerank_head(self, client, bad):
        with pytest.raises(ValueError):
            client.recall("agent-1", "q", rerank_head=bad)

    def test_instructions_and_degraded(self):
        resp = RecallResponse.from_dict(RECALL_ANSWER)
        assert resp.instructions is not None
        assert resp.instructions[0].id == "i1"
        assert resp.instructions[0].retrieval == "instruction"
        assert resp.degraded == ["graph: store unreachable"]
        assert resp.is_degraded

    def test_older_server(self):
        resp = RecallResponse.from_dict({"memories": []})
        assert resp.instructions is None
        assert resp.degraded == [] and not resp.is_degraded

    async def test_async_recall(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=RECALL_ANSWER)

        resp = await _async(handler).recall("agent-1", "q", rerank_head=5, importance_weighted=True)
        assert seen[0]["rerank_head"] == 5 and seen[0]["importance_weighted"] is True
        assert resp.is_degraded and resp.instructions is not None


# ---------------------------------------------------------------------------
# get_memory projection, deduplicated content
# ---------------------------------------------------------------------------


class TestProjection:
    def test_projection_days(self, rsps, client):
        rsps.add(
            responses.GET,
            f"{BASE}/v1/memory/get/m1",
            json={
                **_mem(),
                "strength_projection": [
                    {"day": 0, "importance": 0.5},
                    {"day": 1, "importance": 0.48},
                ],
            },
        )
        mem = client.get_memory("agent-1", "m1", projection_days=1)
        q = parse_qs(urlparse(rsps.calls[0].request.url).query)
        assert q == {"agent_id": ["agent-1"], "projection_days": ["1"]}
        assert mem["strength_projection"][1]["importance"] == 0.48

    def test_projection_without_agent(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/memory/get/m1", json=_mem())
        client.get_memory(None, "m1", projection_days=30)
        q = parse_qs(urlparse(rsps.calls[0].request.url).query)
        assert q == {"projection_days": ["30"]}

    @pytest.mark.parametrize("bad", [0, 366, True])
    def test_projection_bounds(self, client, bad):
        with pytest.raises(ValueError):
            client.get_memory("agent-1", "m1", projection_days=bad)

    async def test_async_projection(self):
        seen = []

        def handler(request):
            seen.append(dict(request.url.params))
            return httpx.Response(200, json=_mem())

        await _async(handler).get_memory("agent-1", "m1", projection_days=7)
        assert seen[0] == {"agent_id": "agent-1", "projection_days": "7"}

    def test_deduplicated_content_and_deletion_kind(self):
        resp = DeduplicatedListResponse.from_dict(
            {
                "agent_id": "agent-1",
                "memories": [
                    {
                        "id": "m1",
                        "kind": "deletion",
                        "survivor_id": "",
                        "deleted_at": 1,
                        "restorable_until": 2,
                        "content": "I love sushi",
                    },
                    {
                        "id": "m2",
                        "kind": "deduplication",
                        "survivor_id": "m0",
                        "deleted_at": 1,
                        "restorable_until": 2,
                    },
                ],
                "count": 2,
            }
        )
        assert resp.memories[0].kind == "deletion"
        assert resp.memories[0].content == "I love sushi"
        assert resp.memories[1].content is None


# ---------------------------------------------------------------------------
# async import
# ---------------------------------------------------------------------------


RUNNING = {
    "job_id": "imp-1",
    "status": "running",
    "format": "jsonl",
    "total": 2,
    "imported": 0,
    "skipped": 0,
    "errors": [],
    "started_at": 1,
    "finished_at": None,
}


class TestAsyncImport:
    def test_run_async_query_and_202(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/import", status=202, json=RUNNING)
        resp = client.import_memories([{"content": "a"}], agent_id="agent-1", run_async=True)
        q = parse_qs(urlparse(rsps.calls[0].request.url).query)
        assert q == {"agent_id": ["agent-1"], "format": ["jsonl"], "async": ["true"]}
        assert resp.status == "running" and resp.job_id == "imp-1"

    def test_default_import_has_no_async(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/import", json={**RUNNING, "status": "done"})
        client.import_memories([{"content": "a"}], agent_id="agent-1")
        q = parse_qs(urlparse(rsps.calls[0].request.url).query)
        assert "async" not in q

    def test_wait_for_import_job(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/import/imp-1/status", json=RUNNING)
        rsps.add(
            responses.GET,
            f"{BASE}/v1/import/imp-1/status",
            json={**RUNNING, "status": "done", "imported": 2, "finished_at": 2},
        )
        job = client.wait_for_import_job("imp-1", poll_interval=0.1)
        assert job.status == "done" and job.imported == 2
        assert len(rsps.calls) == 2

    def test_wait_for_import_job_times_out(self, rsps, client):
        from dakera.exceptions import TimeoutError as DakeraTimeout

        rsps.add(responses.GET, f"{BASE}/v1/import/imp-1/status", json=RUNNING)
        with pytest.raises(DakeraTimeout):
            client.wait_for_import_job("imp-1", timeout=0.0)

    async def test_async_import_and_wait(self):
        calls = []

        def handler(request):
            calls.append((request.method, request.url.path, dict(request.url.params)))
            if request.method == "POST":
                return httpx.Response(202, json=RUNNING)
            return httpx.Response(200, json={**RUNNING, "status": "failed"})

        c = _async(handler)
        resp = await c.import_memories("{}\n", agent_id="agent-1", run_async=True)
        job = await c.wait_for_import_job("imp-1", poll_interval=0.1)
        assert resp.status == "running" and job.status == "failed"
        assert calls[0][2]["async"] == "true"


# ---------------------------------------------------------------------------
# namespace config, memory policy, links, extraction
# ---------------------------------------------------------------------------


class TestConfigViews:
    def test_get_namespace_ner_config(self, rsps, client):
        answer = {
            "namespace": "ns",
            "extract_entities": True,
            "entity_types": ["person", "organization", "location"],
            "source": "server_default",
        }
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/ns/config", json=answer)
        assert client.get_namespace_ner_config("ns") == answer

    async def test_async_get_namespace_ner_config(self):
        def handler(request):
            assert request.method == "GET" and request.url.path == "/v1/namespaces/ns/config"
            return httpx.Response(200, json={"source": "namespace"})

        assert (await _async(handler).get_namespace_ner_config("ns"))["source"] == "namespace"

    def test_memory_policy_stored_is_read_not_sent(self):
        policy = MemoryPolicy.from_dict({"stored": False, "working_ttl_seconds": None})
        assert policy.stored is False
        assert policy.working_ttl_seconds is None
        assert "stored" not in policy.to_dict()
        assert MemoryPolicy.from_dict({}).stored is None

    def test_link_label(self):
        link = GraphLinkResponse.from_dict(
            {"from_id": "a", "to_id": "b", "edge_type": "linked_by", "label": "caused"}
        )
        assert link.label == "caused" and link.edge.label == "caused"
        edge = GraphEdge.from_dict(
            {"from_id": "a", "to_id": "b", "edge_type": "related_to", "weight": 0.8}
        )
        assert edge.label is None

    def test_extraction_extractor_and_warnings(self):
        resp = EntityExtractionResponse.from_dict(
            {
                "entities": [{"entity_type": "person", "value": "Alice", "score": 0.9}],
                "count": 1,
                "extractor": "rule_based",
                "warnings": ["neural pass failed"],
            }
        )
        assert resp.extractor == "rule_based" and resp.warnings == ["neural pass failed"]
        old = EntityExtractionResponse.from_dict({"entities": []})
        assert old.extractor is None and old.warnings == []


# ---------------------------------------------------------------------------
# failures that used to be silent
# ---------------------------------------------------------------------------


class TestFailuresAreSaid:
    def test_dedup_groups_failed(self):
        resp = DeduplicateResponse.from_dict(
            {
                "duplicates_found": 3,
                "removed_count": 1,
                "groups": [],
                "groups_failed": [
                    {"canonical_id": "c", "duplicate_ids": ["d1", "d2"], "reason": "store"}
                ],
            }
        )
        assert resp.groups_failed[0].duplicate_ids == ["d1", "d2"]
        assert DeduplicateResponse.from_dict({}).groups_failed == []

    def test_cross_agent_unavailable(self):
        stats = {"total_agents": 1, "total_nodes": 0, "total_cross_edges": 0, "density": 0.0}
        resp = CrossAgentNetworkResponse.from_dict(
            {
                "agents": [],
                "nodes": [],
                "edges": [],
                "stats": stats,
                "unavailable": [{"agent_id": "a2", "reason": "read failed"}],
            }
        )
        assert resp.unavailable[0].agent_id == "a2"
        assert CrossAgentNetworkResponse.from_dict({"stats": stats}).unavailable == []

    def test_compress_deprecation_report(self):
        resp = CompressResponse.from_dict(
            {
                "agent_id": "a",
                "deprecated_ids": ["m1"],
                "deprecation_skipped": ["m2"],
                "deprecation_error": None,
            }
        )
        assert resp.deprecation_skipped == ["m2"] and resp.deprecation_error is None
        failed = CompressResponse.from_dict({"agent_id": "a", "deprecation_error": "store"})
        assert failed.deprecation_error == "store"

    def test_ttl_cleanup_failed(self):
        resp = TtlCleanupResponse.from_dict(
            {
                "success": False,
                "vectors_removed": 3,
                "namespaces_cleaned": ["a"],
                "message": "partial",
                "failed": [{"namespace": "b", "reason": "unreadable"}],
            }
        )
        assert not resp.success and resp.failed[0].namespace == "b"

    def test_derivations_entity_tagging(self):
        status = DerivationStatus.from_dict(
            {"settled": False, "entity_tagging": {"mode": "background", "pending": 4}}
        )
        assert status.entity_tagging == {"mode": "background", "pending": 4}
        assert DerivationStatus.from_dict({}).entity_tagging is None


# ---------------------------------------------------------------------------
# admin
# ---------------------------------------------------------------------------


class TestAdmin:
    def test_consolidation_mode(self, rsps, client):
        rsps.add(
            responses.PUT,
            f"{BASE}/v1/admin/autopilot/config",
            json={"success": True, "config": {}, "message": "ok"},
        )
        client.autopilot_update_config(consolidation_mode="report")
        assert json.loads(rsps.calls[0].request.body) == {"consolidation_mode": "report"}

    async def test_async_consolidation_mode(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"success": True})

        await _async(handler).autopilot_update_config(consolidation_mode="merge")
        assert seen == [{"consolidation_mode": "merge"}]

    def test_warm_cache_hints(self):
        body = WarmCacheRequest(
            "ns", query_vector=[0.1, 0.2], min_score_threshold=0.7, recently_accessed_within=60
        ).to_dict()
        assert body["query_vector"] == [0.1, 0.2]
        assert body["min_score_threshold"] == 0.7
        assert body["recently_accessed_within"] == 60
        plain = WarmCacheRequest("ns").to_dict()
        assert "query_vector" not in plain and "min_score_threshold" not in plain

    def test_202_logged_change_is_success(self, rsps, client):
        rsps.add(
            responses.DELETE,
            f"{BASE}/admin/keys/k1",
            status=202,
            json={"success": True, "message": "logged; stored later"},
        )
        assert client.delete_key("k1")["success"] is True

    def test_409_superseded_change(self, rsps, client):
        rsps.add(
            responses.PUT,
            f"{BASE}/v1/admin/quotas/ns",
            status=409,
            json={"error": "superseded", "code": "CONFLICT"},
        )
        with pytest.raises(ConflictError):
            client.admin_set_quota("ns", {"max_vectors": 10})


# ---------------------------------------------------------------------------
# a fenced namespace answers 503 + Retry-After
# ---------------------------------------------------------------------------


class TestRetryAfter:
    def test_fenced_namespace_retries_after_the_header(self, rsps, monkeypatch):
        sleeps = []
        monkeypatch.setattr("dakera.client.time.sleep", sleeps.append)
        rsps.add(
            responses.POST,
            f"{BASE}/v1/namespaces",
            status=503,
            json={"error": "namespace is being deleted", "code": "SERVICE_UNAVAILABLE"},
            headers={"Retry-After": "5"},
        )
        rsps.add(responses.POST, f"{BASE}/v1/namespaces", json={"name": "ns"})
        c = DakeraClient(BASE, retry_config=RetryConfig(max_retries=2))
        c._request("POST", "/v1/namespaces", data={"name": "ns"})
        assert sleeps == [5.0]

    async def test_async_fenced_namespace_raises_with_retry_after(self):
        def handler(request):
            return httpx.Response(
                503,
                json={"error": "fenced", "code": "SERVICE_UNAVAILABLE"},
                headers={"Retry-After": "5"},
            )

        with pytest.raises(ServiceUnavailableError) as exc:
            await _async(handler)._request("POST", "/v1/namespaces", data={"name": "ns"})
        assert exc.value.retry_after == 5
