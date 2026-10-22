"""Server v0.12.3 wire changes beyond documents and the recall neighbourhood:
``event_time`` on store and batch store, the truncation report, ``warnings``,
``rerank_candidates`` / ``rerank_report``, the restorable deleted memories of
deduplication and consolidation, ``get_memory`` without ``agent_id``, the
autopilot dedup settings, ``capabilities.naming`` session-id rules, audit
``reason`` and the new error codes — each also against an older server that
sends none of the new fields."""

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import responses

from dakera import (
    AsyncDakeraClient,
    DakeraClient,
    DeduplicatedListResponse,
    NotFoundError,
    RecallResponse,
)
from dakera.exceptions import DakeraError, ErrorCode, error_from_response
from dakera.models import (
    AuditListResponse,
    BatchRecallResponse,
    BatchStoreMemoryItem,
    BatchStoreMemoryRequest,
    BatchStoreMemoryResponse,
    NamingCapabilities,
    RetryConfig,
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


TRUNCATION = {
    "truncated": True,
    "content_tokens": 2804,
    "embedded_tokens": 512,
    "max_tokens": 512,
    "windows": 0,
}


@pytest.fixture
def rsps():
    with responses.RequestsMock() as r:
        yield r


@pytest.fixture
def client():
    return DakeraClient(BASE, retry_config=RetryConfig(max_retries=1))


def _async(handler):
    c = AsyncDakeraClient(BASE, retry_config=RetryConfig(max_retries=1))
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


# ---------------------------------------------------------------------------
# event_time
# ---------------------------------------------------------------------------


class TestEventTime:
    @pytest.mark.parametrize(
        ("value", "wire"),
        [
            (1_684_549_260, 1_684_549_260),
            (1_684_549_260.75, 1_684_549_260.75),
            ("2023-05-20T02:21:00Z", "2023-05-20T02:21:00Z"),
            (
                datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc),
                "2023-05-20T02:21:00+00:00",
            ),
            (
                datetime(2023, 5, 20, 4, 21, tzinfo=timezone(timedelta(hours=2))),
                "2023-05-20T04:21:00+02:00",
            ),
        ],
    )
    def test_store_sends_event_time(self, rsps, client, value, wire):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/store",
            json={"memory": _mem(), "embedding_time_ms": 3},
        )
        client.store_memory("agent-1", "moved to Paris", event_time=value)
        body = json.loads(rsps.calls[0].request.body)
        assert body["event_time"] == wire
        assert "_dakera_event_time" not in (body.get("metadata") or {})

    def test_store_omits_event_time_by_default(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/store",
            json={"memory": _mem(), "embedding_time_ms": 3},
        )
        client.store_memory("agent-1", "c")
        assert "event_time" not in json.loads(rsps.calls[0].request.body)

    @pytest.mark.parametrize("bad", [datetime(2023, 5, 20), True, [1], {"t": 1}])
    def test_bad_event_time_is_refused_before_sending(self, client, bad):
        with pytest.raises((ValueError, TypeError)):
            client.store_memory("agent-1", "c", event_time=bad)

    def test_batch_item_event_time(self):
        req = BatchStoreMemoryRequest(
            "agent-1",
            [
                BatchStoreMemoryItem("a", event_time="2023-05-20T02:21:00Z"),
                BatchStoreMemoryItem("b", event_time=datetime(2023, 5, 21, tzinfo=timezone.utc)),
                BatchStoreMemoryItem("c"),
            ],
        )
        items = req.to_dict()["memories"]
        assert items[0]["event_time"] == "2023-05-20T02:21:00Z"
        assert items[1]["event_time"] == "2023-05-21T00:00:00+00:00"
        assert "event_time" not in items[2]

    def test_batch_item_naive_datetime_refused(self):
        with pytest.raises(ValueError):
            BatchStoreMemoryItem("a", event_time=datetime(2023, 5, 20)).to_dict()

    async def test_async_store_sends_event_time(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"memory": _mem(), "embedding_time_ms": 1})

        await _async(handler).store_memory("agent-1", "c", event_time=1_684_549_260)
        assert seen[0]["event_time"] == 1_684_549_260


# ---------------------------------------------------------------------------
# Truncation report and warnings on writes
# ---------------------------------------------------------------------------


class TestStoreReports:
    def test_store_copies_truncation_and_warnings(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/store",
            json={
                "memory": _mem(),
                "embedding_time_ms": 3,
                **TRUNCATION,
                "warnings": ["unknown field 'when' ignored"],
            },
        )
        m = client.store_memory("agent-1", "long text")
        assert m["id"] == "m1"
        assert m["truncated"] is True
        assert (m["content_tokens"], m["embedded_tokens"], m["max_tokens"]) == (2804, 512, 512)
        assert m["windows"] == 0
        assert m["warnings"] == ["unknown field 'when' ignored"]

    def test_store_old_server_has_no_report(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/store",
            json={"memory": _mem(), "embedding_time_ms": 3},
        )
        m = client.store_memory("agent-1", "c")
        assert m == _mem()

    def test_batch_store_truncated_and_warnings(self):
        r = BatchStoreMemoryResponse.from_dict(
            {
                "stored": [_mem("m1"), _mem("m2")],
                "stored_count": 2,
                "total_embedding_time_ms": 9,
                "truncated": [{"id": "m2", **TRUNCATION, "windows": 11}],
                "warnings": ["unknown field 'memories[].when' ignored (in 2 items)"],
            }
        )
        (t,) = r.truncated
        assert t.id == "m2" and t.content_tokens == 2804 and t.embedded_tokens == 512
        assert t.max_tokens == 512 and t.windows == 11 and t.tokens_estimated is False
        assert r.warnings == ["unknown field 'memories[].when' ignored (in 2 items)"]

    def test_batch_store_old_server(self):
        r = BatchStoreMemoryResponse.from_dict(
            {"stored": [_mem()], "stored_count": 1, "total_embedding_time_ms": 1}
        )
        assert r.truncated == [] and r.warnings == [] and r.ended_sessions == []

    def test_batch_recall_warnings(self):
        r = BatchRecallResponse.from_dict(
            {
                "memories": [],
                "total": 0,
                "filtered": 0,
                "truncated": False,
                "warnings": ["unknown field 'filter.when' ignored"],
            }
        )
        assert r.warnings == ["unknown field 'filter.when' ignored"]
        assert (
            BatchRecallResponse.from_dict({"memories": [], "total": 0, "filtered": 0}).warnings
            == []
        )


# ---------------------------------------------------------------------------
# Recall: rerank_candidates, rerank_report, warnings, empty agent
# ---------------------------------------------------------------------------


RECALL_ANSWER = {
    "memories": [{"memory": _mem(), "score": 0.9, "rank_score": 0.8}],
    "query_embedding_time_ms": 4,
    "search_time_ms": 7,
    "rerank": {"applied": True, "candidates": 180, "capped": True},
    "rerank_report": {"applied": True, "candidates": 180, "capped": True},
    "warnings": ["unknown field 'k' ignored"],
}


class TestRecall:
    def test_rerank_candidates_sent_and_report_parsed(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", json=RECALL_ANSWER)
        r = client.recall("agent-1", "q", rerank_candidates=500)
        body = json.loads(rsps.calls[0].request.body)
        assert body["rerank_candidates"] == 500
        assert r.rerank_report is not None
        assert r.rerank_report.applied and r.rerank_report.capped
        assert r.rerank_report.candidates == 180 and r.rerank_report.skipped is None
        assert r.warnings == ["unknown field 'k' ignored"]
        assert (r.query_embedding_time_ms, r.search_time_ms) == (4, 7)

    def test_rerank_candidates_omitted_by_default(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", json=RECALL_ANSWER)
        client.recall("agent-1", "q")
        assert "rerank_candidates" not in json.loads(rsps.calls[0].request.body)

    @pytest.mark.parametrize("bad", [0, -3, True])
    def test_rerank_candidates_must_be_positive(self, client, bad):
        with pytest.raises(ValueError):
            client.recall("agent-1", "q", rerank_candidates=bad)

    def test_report_of_an_older_server(self):
        r = RecallResponse.from_dict(
            {
                "memories": [],
                "query_embedding_time_ms": 1,
                "search_time_ms": 0,
                "rerank": {"applied": False, "candidates": 0, "skipped": "bm25_routing"},
            }
        )
        assert r.rerank_report is not None
        assert r.rerank_report.skipped == "bm25_routing" and r.rerank_report.capped is False
        assert r.warnings == []
        assert RecallResponse.from_dict({"memories": []}).rerank_report is None

    def test_agent_without_memories_is_an_empty_answer(self, rsps, client):
        # v0.12.3: 200 with no memories (older servers: 404 NAMESPACE_NOT_FOUND).
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/recall",
            json={
                "memories": [],
                "query_embedding_time_ms": 1,
                "search_time_ms": 0,
                "rerank": {"applied": False, "candidates": 0, "skipped": "no_candidates"},
                "rerank_report": {"applied": False, "candidates": 0, "skipped": "no_candidates"},
            },
        )
        r = client.recall("never-stored", "q")
        assert r.memories == [] and r.rerank_report is not None
        assert r.rerank_report.skipped == "no_candidates"

    async def test_async_rerank_candidates(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=RECALL_ANSWER)

        c = _async(handler)
        r = await c.recall("agent-1", "q", rerank_candidates=40)
        assert seen[0]["rerank_candidates"] == 40
        assert r.rerank_report is not None and r.rerank_report.capped
        with pytest.raises(ValueError):
            await c.recall("agent-1", "q", rerank_candidates=0)


# ---------------------------------------------------------------------------
# get_memory without agent_id
# ---------------------------------------------------------------------------


class TestGetMemory:
    def test_without_agent_id_sends_no_query(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/memory/get/m1", json=_mem())
        assert client.get_memory(None, "m1")["id"] == "m1"
        assert urlparse(rsps.calls[0].request.url).query == ""

    def test_with_agent_id(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/memory/get/m1", json=_mem())
        client.get_memory("agent-1", "m1")
        query = parse_qs(urlparse(rsps.calls[0].request.url).query)
        assert query == {"agent_id": ["agent-1"]}

    def test_id_is_one_path_segment(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/memory/get/a%2Fb%3Fc", json=_mem())
        client.get_memory("agent-1", "a/b?c")
        assert urlparse(rsps.calls[0].request.url).path == "/v1/memory/get/a%2Fb%3Fc"

    async def test_async_without_agent_id(self):
        urls = []

        def handler(request):
            urls.append(request.url)
            return httpx.Response(200, json=_mem())

        await _async(handler).get_memory(None, "m1")
        assert urls[0].path == "/v1/memory/get/m1" and urls[0].query == b""


# ---------------------------------------------------------------------------
# Restorable deleted memories (deduplication / consolidation)
# ---------------------------------------------------------------------------


DEDUP_LIST = {
    "agent_id": "agent-1",
    "memories": [
        {
            "id": "m9",
            "kind": "deduplication",
            "survivor_id": "m1",
            "session_id": "s1",
            "deleted_at": 1_790_000_000,
            "restorable_until": 1_790_604_800,
        },
        {
            "id": "m8",
            "kind": "consolidation",
            "survivor_id": "m_merged",
            "deleted_at": 1_789_000_000,
            "restorable_until": 1_789_604_800,
        },
    ],
    "count": 2,
}


class TestDeduplicated:
    def test_list(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/memory/deduplicated", json=DEDUP_LIST)
        r = client.list_deduplicated("agent-1")
        assert isinstance(r, DeduplicatedListResponse)
        assert parse_qs(urlparse(rsps.calls[0].request.url).query) == {"agent_id": ["agent-1"]}
        assert r.agent_id == "agent-1" and r.count == 2
        first, second = r.memories
        assert (first.id, first.kind, first.survivor_id, first.session_id) == (
            "m9",
            "deduplication",
            "m1",
            "s1",
        )
        assert first.restorable_until - first.deleted_at == 7 * 24 * 3600
        assert second.kind == "consolidation" and second.session_id is None

    def test_restore(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/deduplicated/m9/restore", json=_mem("m9"))
        m = client.restore_deduplicated("agent-1", "m9")
        assert m["id"] == "m9"
        req = rsps.calls[0].request
        assert parse_qs(urlparse(req.url).query) == {"agent_id": ["agent-1"]}
        assert not req.body

    def test_restore_errors(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/deduplicated/gone/restore",
            status=404,
            json={"error": "no restorable copy", "code": "NOT_FOUND"},
        )
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/deduplicated/back/restore",
            status=409,
            json={"error": "the agent holds back again", "code": "CONFLICT"},
        )
        with pytest.raises(NotFoundError):
            client.restore_deduplicated("agent-1", "gone")
        with pytest.raises(DakeraError) as exc:
            client.restore_deduplicated("agent-1", "back")
        assert exc.value.status_code == 409 and exc.value.code == ErrorCode.CONFLICT

    async def test_async_list_and_restore(self):
        calls = []

        def handler(request):
            calls.append((request.method, request.url.path, dict(request.url.params)))
            if request.method == "GET":
                return httpx.Response(200, json=DEDUP_LIST)
            return httpx.Response(200, json=_mem("m9"))

        c = _async(handler)
        r = await c.list_deduplicated("agent-1")
        assert r.count == 2 and r.memories[0].survivor_id == "m1"
        m = await c.restore_deduplicated("agent-1", "m9")
        assert m["id"] == "m9"
        assert calls == [
            ("GET", "/v1/memory/deduplicated", {"agent_id": "agent-1"}),
            ("POST", "/v1/memory/deduplicated/m9/restore", {"agent_id": "agent-1"}),
        ]


# ---------------------------------------------------------------------------
# Autopilot dedup settings
# ---------------------------------------------------------------------------


class TestAutopilot:
    def test_new_settings_sent_when_set(self, rsps, client):
        rsps.add(
            responses.PUT,
            f"{BASE}/v1/admin/autopilot/config",
            json={"success": True, "config": {}, "message": "ok"},
        )
        client.autopilot_update_config(
            dedup_mode="report",
            dedup_cross_session=False,
            consolidate_cross_session=True,
            trash_retention_hours=24,
        )
        assert json.loads(rsps.calls[0].request.body) == {
            "dedup_mode": "report",
            "dedup_cross_session": False,
            "consolidate_cross_session": True,
            "trash_retention_hours": 24,
        }

    def test_new_settings_omitted_by_default(self, rsps, client):
        rsps.add(
            responses.PUT,
            f"{BASE}/v1/admin/autopilot/config",
            json={"success": True, "config": {}, "message": "ok"},
        )
        client.autopilot_update_config(enabled=True)
        assert json.loads(rsps.calls[0].request.body) == {"enabled": True}

    async def test_async_new_settings(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"success": True, "config": {}, "message": "ok"})

        await _async(handler).autopilot_update_config(dedup_mode="merge", trash_retention_hours=0)
        assert seen[0] == {"dedup_mode": "merge", "trash_retention_hours": 0}


# ---------------------------------------------------------------------------
# Capabilities, audit, error codes
# ---------------------------------------------------------------------------


class TestReadModels:
    def test_naming_session_rules(self):
        n = NamingCapabilities.from_dict(
            {
                "agent_id_pattern": "^[a-zA-Z0-9][a-zA-Z0-9_\\-.]*$",
                "session_id_pattern": "^[a-zA-Z0-9][a-zA-Z0-9_\\-:.]*$",
                "session_id_max_bytes": 256,
            }
        )
        assert ":" in n.session_id_pattern and n.session_id_max_bytes == 256
        old = NamingCapabilities.from_dict({"agent_id_pattern": "x"})
        assert old.session_id_pattern == "" and old.session_id_max_bytes == 0

    def test_audit_events_as_the_server_sends_them(self):
        r = AuditListResponse.from_dict(
            {
                "events": [
                    {
                        "id": 41,
                        "event_type": "deduplicated",
                        "agent_id": "agent-1",
                        "memory_id": "m9",
                        "session_id": "s1",
                        "importance": 0.4,
                        "timestamp": 1_790_000_000_123,
                        "reason": "absorbed into m1",
                    },
                    {
                        "id": 40,
                        "event_type": "stored",
                        "agent_id": "agent-1",
                        "timestamp": 1_789_999_999_000,
                    },
                ],
                "count": 2,
            }
        )
        assert r.total == 2
        first, second = r.events
        assert first.id == 41 and first.memory_id == "m9" and first.session_id == "s1"
        assert first.importance == 0.4 and first.reason == "absorbed into m1"
        assert first.timestamp == 1_790_000_000_123
        assert second.reason is None and second.memory_id is None and second.details == {}

    def test_list_audit_events_params(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/audit", json={"events": [], "count": 0})
        r = client.list_audit_events(event_type="deduplicated", from_ts=1, to_ts=2, limit=5)
        query = parse_qs(urlparse(rsps.calls[0].request.url).query)
        assert query == {"event_type": ["deduplicated"], "from": ["1"], "to": ["2"], "limit": ["5"]}
        assert r.total == 0 and r.events == []

    @pytest.mark.parametrize(
        ("status", "code"),
        [
            (421, ErrorCode.HOST_NOT_ALLOWED),
            (422, ErrorCode.UNPROCESSABLE_CONTENT),
            (504, ErrorCode.EXTRACTION_TIMEOUT),
            (429, ErrorCode.RATE_LIMIT_EXCEEDED),
        ],
    )
    def test_error_codes(self, status, code):
        err = error_from_response(status, {"error": "x", "code": code.value})
        assert err.code == code and err.status_code == status
        # The server's message is kept for every status, including 421 / 422 / 429.
        assert err.message == "x"
