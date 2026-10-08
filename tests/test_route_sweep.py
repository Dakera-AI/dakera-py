# ruff: noqa: E501
"""Every route the SDK calls exists on the v0.12.2 server, and the calls fixed by the
v0.12 route sweep send the bodies the server reads.

``tests/v0_12_routes.txt`` is the v0.12.2 router (``crates/api/src/lib.rs``), one
``METHOD /path`` per line with path parameters normalised to ``{}``.
"""

import json
import pathlib
import re

import httpx
import pytest
import responses

from dakera import AsyncDakeraClient, DakeraClient, NotFoundError
from dakera.models import RetryConfig

BASE = "http://localhost:3000"
HERE = pathlib.Path(__file__).parent
SRC = HERE.parent / "src" / "dakera"


def _server_routes() -> set[tuple[str, str]]:
    out = set()
    for line in (HERE / "v0_12_routes.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            method, path = line.split(" ", 1)
            out.add((method, path))
    return out


def _sdk_calls() -> dict[tuple[str, str], set[str]]:
    calls: dict[tuple[str, str], set[str]] = {}
    pat = re.compile(
        r'(?:_request|self\._client\.(?:stream|request))\(\s*"(GET|POST|PUT|PATCH|DELETE)",\s*(?:self\._url\()?f?"([^"]+)"'
    )
    for name in ("client.py", "async_client.py"):
        text = (SRC / name).read_text()
        for m in pat.finditer(text):
            path = re.sub(r"\{[^}]+\}", "{}", m.group(2)).split("?")[0]
            calls.setdefault((m.group(1), path), set()).add(name)
        for m in re.finditer(r'self\._url\(f?"([^"]+)"\)', text):
            path = re.sub(r"\{[^}]+\}", "{}", m.group(1))
            calls.setdefault(("URL", path), set()).add(name)
    return calls


class TestEveryCallHasARoute:
    def test_snapshot_is_populated(self):
        routes = _server_routes()
        assert len(routes) > 150
        assert ("PUT", "/v1/admin/quotas/default") in routes
        assert ("PUT", "/v1/admin/quotas") not in routes

    def test_sdk_calls_exist_on_the_server(self):
        routes = _server_routes()
        calls = _sdk_calls()
        assert len(calls) > 150  # the scan itself works
        paths = {p for _, p in routes}
        missing = sorted(
            f"{m} {p} ({', '.join(sorted(files))})"
            for (m, p), files in calls.items()
            # "URL" entries are streamed / uploaded by raw URL: the path must exist
            if ((m, p) not in routes if m != "URL" else p not in paths)
        )
        assert missing == []


@pytest.fixture
def rsps():
    with responses.RequestsMock() as r:
        yield r


@pytest.fixture
def client():
    return DakeraClient(BASE, retry_config=RetryConfig(max_retries=1))


def body(call):
    return json.loads(call.request.body)


def _async(handler):
    c = AsyncDakeraClient(BASE, retry_config=RetryConfig(max_retries=1))
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE)
    return c


class Recorder:
    def __init__(self, payload=None):
        self.payload = {} if payload is None else payload
        self.seen: list[tuple[str, str, dict | None]] = []

    def __call__(self, request):
        data = json.loads(request.content) if request.content else None
        self.seen.append(
            (
                request.method,
                request.url.path + (f"?{request.url.query.decode()}" if request.url.query else ""),
                data,
            )
        )
        return httpx.Response(200, json=self.payload)


class TestSyncBodies:
    def test_query_sends_include_vectors_and_reads_vector(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/n/query",
                 json={"results": [{"id": "a", "score": 0.9, "vector": [1.0, 2.0]}]})  # fmt: skip
        out = client.query("n", [1.0, 2.0], include_values=True)
        b = body(rsps.calls[0])
        assert b["include_vectors"] is True and "include_values" not in b
        assert out.results[0].values == [1.0, 2.0]

    def test_create_namespace_body(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces",
                 json={"namespace": "n", "dimension": 4, "distance": "euclidean", "created": True})  # fmt: skip
        info = client.create_namespace("n", dimensions=4, distance="euclidean")
        assert body(rsps.calls[0]) == {"name": "n", "dimension": 4, "distance": "euclidean"}
        assert info.name == "n" and info.dimensions == 4

    def test_explain_query_body(self, rsps, client):
        rsps.add(
            responses.POST, f"{BASE}/v1/namespaces/n/explain", json={"query_type": "hybrid_search"}
        )
        client.explain_query(
            "n", vector=[1.0], text_query="hi", query_type="hybrid_search", execute=True
        )
        assert body(rsps.calls[0]) == {"query_type": "hybrid_search", "top_k": 10, "vector": [1.0],
                                       "text_query": "hi", "execute": True}  # fmt: skip

    def test_explain_query_default_type(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/n/explain", json={})
        client.explain_query("n", vector=[1.0])
        assert body(rsps.calls[0])["query_type"] == "vector_search"

    def test_delete_by_ids(self, rsps, client):
        rsps.add(
            responses.POST, f"{BASE}/v1/namespaces/n/vectors/delete", json={"deleted_count": 2}
        )
        assert client.delete("n", ids=["a", "b"])["deleted_count"] == 2
        assert body(rsps.calls[0]) == {"ids": ["a", "b"]}

    def test_restore_backup_and_index_stats_and_rebuild(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/admin/backups/restore", json={"restore_id": "r"})
        client.restore_backup("b1")
        assert body(rsps.calls[0]) == {"backup_id": "b1"}
        rsps.add(responses.POST, f"{BASE}/v1/admin/indexes/rebuild", json={"job_id": "j"})
        client.rebuild_indexes("n")
        assert body(rsps.calls[1]) == {"namespace": "n"}


class TestAsyncParity:
    async def test_update_quotas(self):
        r = Recorder({"success": True})
        c = _async(r)
        await c.update_quotas({"max_vectors": 5})
        await c.update_quotas({"max_vectors": 6}, namespace="ns")
        assert r.seen[0] == ("PUT", "/v1/admin/quotas/default", {"config": {"max_vectors": 5}})
        assert r.seen[1] == ("PUT", "/v1/admin/quotas/ns", {"config": {"max_vectors": 6}})

    async def test_memory_feedback(self):
        r = Recorder({"memory_id": "m", "signal": "flag", "new_importance": 0.5})
        await _async(r).memory_feedback("a", "m", "flag")
        assert r.seen[0] == (
            "POST",
            "/v1/memory/feedback",
            {"agent_id": "a", "memory_id": "m", "signal": "flag"},
        )

    async def test_export_audit(self):
        r = Recorder({"events": [{"id": "e1"}], "count": 1})
        out = await _async(r).export_audit(format="json", agent_id="a", limit=5)
        method, path, _ = r.seen[0]
        assert method == "GET" and path.startswith("/v1/audit/export?")
        assert "format=json" in path and "agent_id=a" in path and "limit=5" in path
        assert out.count == 1 and json.loads(out.data) == [{"id": "e1"}]

    async def test_get_index_stats_compact_flush_fetch(self):
        r = Recorder({"namespace": "n", "vector_count": 7, "dimension": 3, "index_type": "flat"})
        c = _async(r)
        st = await c.get_index_stats("n")
        assert r.seen[0][:2] == ("GET", "/v1/namespaces/n")
        assert (st.total_vectors, st.dimensions, st.index_type) == (7, 3, "flat")
        await c.compact("n", force=True)
        assert r.seen[1] == ("POST", "/ops/compact", {"force": True, "namespace": "n"})
        for gone in ("flush", "fetch", "configure_ttl", "list_extract_providers"):
            assert not hasattr(c, gone)

    async def test_delete_multi_unified_explain(self):
        r = Recorder({"results": []})
        c = _async(r)
        await c.delete("n", ids=["a"])
        await c.delete("n", filter={"k": "v"})
        with pytest.raises(ValueError):
            await c.delete("n", delete_all=True)
        await c.multi_vector_search("n", [[1.0]], negative=[[2.0]], mmr_lambda=0.3)
        await c.unified_query("n", vector=[1.0], text="t", text_weight=0.4)
        await c.explain_query("n", vector=[1.0])
        assert r.seen[0] == ("POST", "/v1/namespaces/n/vectors/delete", {"ids": ["a"]})
        assert r.seen[1] == ("POST", "/v1/namespaces/n/vectors/bulk-delete", {"filter": {"k": "v"}})
        assert r.seen[2][1] == "/v1/namespaces/n/multi-vector"
        assert r.seen[2][2]["positive_vectors"] == [[1.0]] and r.seen[2][2]["enable_mmr"] is True
        assert r.seen[3][1] == "/v1/namespaces/n/unified-query"
        assert r.seen[3][2]["rank_by"] == [
            "Sum",
            [["ANN", [1.0]], ["Product", 0.4, ["text", "BM25", "t"]]],
        ]
        assert (
            r.seen[4][1] == "/v1/namespaces/n/explain"
            and r.seen[4][2]["query_type"] == "vector_search"
        )

    async def test_admin_index_and_backup_routes(self):
        r = Recorder({"namespaces": {"n": {"indexed_vectors": 3}}, "job_id": "j"})
        c = _async(r)
        assert (await c.index_stats("n")) == {"indexed_vectors": 3}
        assert (await c.index_stats())["namespaces"]
        with pytest.raises(NotFoundError):
            await c.index_stats("missing")
        await c.rebuild_indexes("n")
        await c.rebuild_indexes()
        await c.restore_backup("b1")
        assert r.seen[0][:2] == ("GET", "/v1/admin/indexes/stats")
        assert r.seen[3] == ("POST", "/v1/admin/indexes/rebuild", {"namespace": "n"})
        assert r.seen[5] == ("POST", "/v1/admin/backups/restore", {"backup_id": "b1"})

    async def test_query_create_aggregate(self):
        r = Recorder({"results": [{"id": "a", "score": 1.0, "vector": [9.0]}], "namespace": "n",
                      "dimension": 2})  # fmt: skip
        c = _async(r)
        out = await c.query("n", [1.0], include_values=True)
        assert out.results[0].values == [9.0] and r.seen[0][2]["include_vectors"] is True
        await c.create_namespace("n", dimensions=2, distance="cosine")
        assert r.seen[1][2] == {"name": "n", "dimension": 2, "distance": "cosine"}
        await c.aggregate("n", group_by="k", metrics=["count", "max:price"], top_groups=3)
        assert r.seen[2][2] == {"aggregate_by": {"count": ["Count"], "max_price": ["Max", "price"]},
                                "group_by": ["k"], "limit": 3}  # fmt: skip
