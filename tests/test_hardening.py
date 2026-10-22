"""Request safety after server v0.12.2 (Dakera-AI/dakera#920) and SDK robustness:

* the four admin requests the server now checks with ``deny_unknown_fields``
  (``422`` on an unknown field) carry exactly the server's fields;
* ``/ops/shutdown`` (global super_admin);
* retries only where repeating the request is harmless;
* no API key in a ``repr``; SSE decoding.
"""

import json
import warnings

import httpx
import pytest
import requests
import responses

from dakera import (
    AsyncDakeraClient,
    AuthorizationError,
    ConnectionError,
    DakeraClient,
    RotateEncryptionKeyRequest,
    ServerError,
    ServiceUnavailableError,
)
from dakera._util import _retry_safe, _SSEDecoder
from dakera.models import CreateNamespaceKeyResponse, RetryConfig, RotateKeyResponse

BASE = "http://localhost:3000"

# The engine's request structs (crates/api/src/routes/keys.rs `CreateKeyRequest`,
# namespace_keys.rs `CreateNamespaceKeyRequest`, admin/types.rs `CreateBackupRequest`,
# admin/mod.rs `RotateEncryptionKeyRequest`), all `#[serde(deny_unknown_fields)]`.
SERVER_FIELDS = {
    "create_key": ({"name", "scope"}, {"namespaces", "expires_in_days"}),
    "create_namespace_key": ({"name", "scope"}, {"extra_namespaces", "expires_in_days"}),
    "create_backup": ({"name"}, {"backup_type", "namespaces", "encrypt", "compression"}),
    "rotate_encryption_key": (set(), {"new_key", "namespace", "wait_secs"}),
}


@pytest.fixture
def rsps():
    with responses.RequestsMock() as r:
        yield r


@pytest.fixture
def client():
    return DakeraClient(BASE, retry_config=RetryConfig(max_retries=3, base_delay=0, jitter=False))


def body(call):
    return json.loads(call.request.body) if call.request.body else {}


def _allowed(route):
    required, optional = SERVER_FIELDS[route]
    return required, required | optional


class TestStrictAdminBodiesSync:
    def test_create_key_all_fields(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/admin/keys", json={"key_id": "k", "key": "dk"})
        client.create_key("ci", scope="write", namespaces=("a", "b*"), expires_in_days=7)
        required, allowed = _allowed("create_key")
        b = body(rsps.calls[0])
        assert set(b) == allowed
        assert b == {
            "name": "ci",
            "scope": "write",
            "namespaces": ["a", "b*"],
            "expires_in_days": 7,
        }

    def test_create_key_minimal_and_deprecated_args_not_sent(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/admin/keys", json={"key_id": "k", "key": "dk"})
        with pytest.warns(DeprecationWarning) as w:
            client.create_key("ci", ["read"], "2030-01-01")
        messages = " ".join(str(x.message) for x in w)
        assert "permissions" in messages and "expires_at" in messages
        required, _ = _allowed("create_key")
        assert set(body(rsps.calls[0])) == required

    def test_create_namespace_key(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/keys",
            json={"key_id": "k", "key": "dk", "name": "n", "created_at": 1, "warning": "w"},
        )
        client.create_namespace_key("ns", "n", 3, scope="admin", extra_namespaces=["x"])
        required, allowed = _allowed("create_namespace_key")
        assert set(body(rsps.calls[0])) == allowed

    def test_create_namespace_key_minimal(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/namespaces/ns/keys",
            json={"key_id": "k", "key": "dk", "name": "n", "created_at": 1, "warning": "w"},
        )
        client.create_namespace_key("ns", "n")
        assert body(rsps.calls[0]) == {"name": "n", "scope": "read"}

    def test_create_backup(self, rsps, client):
        for _ in range(3):
            rsps.add(responses.POST, f"{BASE}/v1/admin/backups", json={"backup": {}})
        client.create_backup(
            "nightly", backup_type="full", namespaces=["a"], encrypt=True, compression="zstd"
        )
        client.admin_create_backup("n2")
        with pytest.warns(DeprecationWarning, match="include_data"):
            client.create_backup("n3", include_data=True)
        _, allowed = _allowed("create_backup")
        assert set(body(rsps.calls[0])) == allowed
        assert body(rsps.calls[0])["encrypt"] is True
        assert body(rsps.calls[1]) == {"name": "n2"}
        assert body(rsps.calls[2]) == {"name": "n3"}

    def test_create_backup_needs_a_name(self, rsps, client):
        with pytest.raises(ValueError, match="name"):
            client.create_backup()
        # The old positional create_backup(True) never sent a name either.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="name"):
                client.create_backup(True)
        assert len(rsps.calls) == 0

    def test_rotate_encryption_key(self, rsps, client):
        answer = {
            "key_id": "k2",
            "previous_key_id": "k1",
            "scope": "namespace",
            "namespace": "ns",
            "reseal": "completed",
            "rotated": 4,
            "skipped": 0,
            "namespaces": ["ns"],
            "fulltext_indices_rotated": 1,
            "status_url": "/admin/encryption/status",
        }
        for _ in range(2):
            rsps.add(responses.POST, f"{BASE}/v1/admin/encryption/rotate-key", json=answer)
        out = client.rotate_encryption_key("pass phrase", "ns", wait_secs=0)
        assert body(rsps.calls[0]) == {"new_key": "pass phrase", "namespace": "ns", "wait_secs": 0}
        assert out.key_id == "k2" and out.reseal == "completed" and out.scope == "namespace"
        client.rotate_encryption_key()  # random key, global
        assert body(rsps.calls[1]) == {}
        _, allowed = _allowed("rotate_encryption_key")
        assert set(body(rsps.calls[0])) <= allowed

    def test_request_model_to_dict(self):
        assert RotateEncryptionKeyRequest().to_dict() == {}
        assert RotateEncryptionKeyRequest("k", wait_secs=5).to_dict() == {
            "new_key": "k",
            "wait_secs": 5,
        }

    def test_ops_shutdown_needs_super_admin(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/ops/shutdown",
            status=403,
            json={"error": "requires super_admin", "code": "INSUFFICIENT_SCOPE"},
        )
        with pytest.raises(AuthorizationError):
            client.ops_shutdown()
        assert len(rsps.calls) == 1


def _async(handler, **rc):
    c = AsyncDakeraClient(BASE, retry_config=RetryConfig(base_delay=0, jitter=False, **rc))
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


class TestStrictAdminBodiesAsync:
    async def test_bodies(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content) if request.content else {})
            return httpx.Response(
                200,
                json={"key_id": "k", "key": "dk", "name": "n", "created_at": 1, "warning": "w"},
            )

        c = _async(handler)
        with pytest.warns(DeprecationWarning):
            await c.create_key("n", permissions=["x"], scope="read", namespaces=[])
        await c.create_namespace_key("ns", "n", extra_namespaces=["e"])
        await c.create_backup("b", compression="lz4")
        await c.rotate_encryption_key(namespace="ns")
        assert seen[0] == {"name": "n", "scope": "read", "namespaces": []}
        assert seen[1] == {"name": "n", "scope": "read", "extra_namespaces": ["e"]}
        assert seen[2] == {"name": "b", "compression": "lz4"}
        assert seen[3] == {"namespace": "ns"}
        for route, sent in zip(SERVER_FIELDS, seen, strict=True):
            assert set(sent) <= _allowed(route)[1]


class TestParityFixes:
    def test_consolidate_dry_run_refused(self, rsps, client):
        with pytest.raises(ValueError, match="dry run"):
            client.consolidate("a", dry_run=True)
        assert len(rsps.calls) == 0

    def test_consolidate_body(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/consolidate", json={})
        with pytest.warns(DeprecationWarning, match="memory_type"):
            client.consolidate(
                "a", memory_type="episodic", threshold=0.8, memory_ids=["m1", "m2"],
                target_type="semantic",
            )  # fmt: skip
        assert body(rsps.calls[0]) == {
            "agent_id": "a",
            "threshold": 0.8,
            "memory_ids": ["m1", "m2"],
            "target_type": "semantic",
        }

    def test_end_session_and_update_memory(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/sessions/s1/end", json={})
        rsps.add(responses.PUT, f"{BASE}/v1/memory/update/m1", json={})
        client.end_session("s1", auto_summarize=True)
        client.update_memory("a", "m1", importance=0.9, tags=["t"])
        assert body(rsps.calls[0]) == {"auto_summarize": True}
        assert body(rsps.calls[1]) == {"importance": 0.9, "tags": ["t"]}


class TestRetryPolicy:
    def test_retry_safe(self):
        assert _retry_safe("GET", "/v1/agents")
        assert _retry_safe("delete", "/v1/namespaces/x")
        assert _retry_safe("POST", "/v1/memory/recall")
        assert _retry_safe("POST", "/v1/memory/search")
        assert _retry_safe("POST", "/v1/namespaces/n/query")
        assert _retry_safe("POST", "/v1/admin/quotas/n/check")
        assert not _retry_safe("POST", "/v1/memory/store")
        assert not _retry_safe("POST", "/v1/memories/store/batch")
        assert not _retry_safe("POST", "/v1/namespaces/n/vectors")
        assert not _retry_safe("PATCH", "/admin/keys/k")
        assert not _retry_safe("POST", "/v1/v1/memory/recall")

    def test_write_not_retried_after_5xx(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", status=500, json={"error": "x"})
        with pytest.raises(ServerError):
            client.store_memory("a", "c")
        assert len(rsps.calls) == 1

    def test_read_only_post_retried_after_5xx(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", status=502, json={"error": "x"})
        rsps.add(responses.POST, f"{BASE}/v1/memory/recall", json={"memories": []})
        assert client.recall("a", "q").memories == []
        assert len(rsps.calls) == 2

    def test_write_retried_on_429_and_503(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", status=429,
                 headers={"Retry-After": "0"}, json={"error": "slow down"})  # fmt: skip
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", status=503,
                 headers={"Retry-After": "0"}, json={"error": "busy"})  # fmt: skip
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", json={"memory": {"id": "m"}})
        assert client.store_memory("a", "c")["id"] == "m"
        assert len(rsps.calls) == 3

    def test_503_exhausted(self, rsps):
        c = DakeraClient(BASE, retry_config=RetryConfig(max_retries=2, base_delay=0))
        for _ in range(2):
            rsps.add(responses.POST, f"{BASE}/v1/memory/store", status=503,
                     headers={"Retry-After": "0"}, json={"error": "busy"})  # fmt: skip
        with pytest.raises(ServiceUnavailableError):
            c.store_memory("a", "c")
        assert len(rsps.calls) == 2

    def test_opt_in_retries_writes(self, rsps):
        rc = RetryConfig(max_retries=2, base_delay=0, retry_non_idempotent=True)
        c = DakeraClient(BASE, retry_config=rc)
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", status=500, json={"error": "x"})
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", json={"memory": {"id": "m"}})
        assert c.store_memory("a", "c")["id"] == "m"

    def test_failed_connect_retried_dropped_connection_not(self, rsps, client):
        refused = requests.exceptions.ConnectTimeout("connect timed out")
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", body=refused)
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", json={"memory": {"id": "m"}})
        assert client.store_memory("a", "c")["id"] == "m"
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/store",
            body=requests.exceptions.ConnectionError("connection reset by peer"),
        )
        with pytest.raises(ConnectionError):
            client.store_memory("a", "c")
        assert len(rsps.calls) == 3

    async def test_async_write_not_retried_after_5xx_or_reset(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if len(calls) == 2:
                raise httpx.ReadError("reset")
            return httpx.Response(500, json={"error": "x"})

        c = _async(handler, max_retries=3)
        with pytest.raises(ServerError):
            await c.store_memory("a", "c")
        with pytest.raises(ConnectionError):  # wrapped, not a raw httpx error
            await c.store_memory("a", "c")
        assert len(calls) == 2

    async def test_async_read_retried_connect_error_retried(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if len(calls) == 1:
                raise httpx.ConnectError("refused")
            if len(calls) == 2:
                return httpx.Response(500, json={"error": "x"})
            return httpx.Response(200, json={"memories": []})

        c = _async(handler, max_retries=3)
        assert (await c.recall("a", "q")).memories == []
        assert len(calls) == 3


class TestNoSecretsInRepr:
    def test_clients(self):
        c = DakeraClient(BASE, api_key="dk-secret-123")
        assert "dk-secret-123" not in repr(c)
        assert "api_key=<set>" in repr(c)
        a = AsyncDakeraClient(BASE, api_key="dk-secret-123")
        assert "dk-secret-123" not in repr(a)
        assert "dk-secret-123" not in repr(DakeraClient(BASE).__dict__.get("api_key"))

    def test_models(self):
        assert "s3cret" not in repr(RotateEncryptionKeyRequest(new_key="s3cret"))
        assert "dk-new" not in repr(RotateKeyResponse(new_key="dk-new", key_id="k"))
        resp = CreateNamespaceKeyResponse(
            key_id="k", key="dk-raw", name="n", namespace="ns", created_at=1, warning="w"
        )
        assert "dk-raw" not in repr(resp)
        assert resp.key == "dk-raw"

    def test_tls_verification_is_on(self):
        assert DakeraClient(BASE)._session.verify is True


class TestSSEDecoder:
    def test_lf_crlf_cr_and_comments(self):
        d = _SSEDecoder()
        assert d.feed(b': keep-alive\n\ndata: {"a": 1}\n\n') == ['{"a": 1}']
        assert d.feed(b"data: x\r\n\r\n") == ["x"]
        # A trailing CR may be half a CRLF: the event completes with the next byte.
        assert d.feed(b"data: y\r\r") == []
        assert d.feed(b"data: z\r\r:") == ["y", "z"]
        assert d.feed(b"event: e\nid: 1\ndata: l1\ndata:l2\n\n") == ["l1\nl2"]

    def test_split_crlf_and_utf8_across_chunks(self):
        d = _SSEDecoder()
        raw = 'data: {"c": "café ☕"}\r\n\r\n'.encode()
        cut = raw.index("é".encode()) + 1  # inside the two-byte é
        out = d.feed(raw[:cut]) + d.feed(raw[cut:-3]) + d.feed(raw[-3:-1]) + d.feed(raw[-1:])
        assert out == ['{"c": "café ☕"}']

    def test_oversized_event_refused(self):
        d = _SSEDecoder(max_event_bytes=10)
        with pytest.raises(ValueError):
            d.feed(b"data: " + b"x" * 20)

    def test_stream_decodes_utf8_without_charset(self, rsps, client):
        payload = {"event_type": "stored", "agent_id": "a", "memory_id": "m", "content": "naïve ☕",
                   "timestamp": 1}  # fmt: skip
        rsps.add(
            responses.GET,
            f"{BASE}/v1/events/stream",
            body=(
                ": hi\r\n\r\ndata: " + json.dumps(payload, ensure_ascii=False) + "\r\n\r\n"
            ).encode(),
            content_type="text/event-stream",
        )
        events = list(client.stream_memory_events())
        assert len(events) == 1
        assert events[0].content == "naïve ☕"

    async def test_async_stream(self):
        def handler(request):
            return httpx.Response(
                200,
                content=b'data: {"type": "namespace_deleted", "namespace": "n\xc3\xa9"}\r\n\r\n',
                headers={"content-type": "text/event-stream"},
            )

        c = _async(handler)
        events = [e async for e in c.stream_namespace_events("n")]
        assert [(e.type, e.namespace) for e in events] == [("namespace_deleted", "né")]

    def test_parse_sse_block_compat(self, client):
        ev = client._parse_sse_block('data: {"type": "stream_lagged", "dropped": 3}')
        assert ev is not None and ev.dropped == 3
        assert client._parse_sse_block(": comment") is None


class TestAsyncImportExport:
    async def test_import_multipart_and_export_file(self):
        seen = []

        def handler(request):
            seen.append(request)
            if request.url.path == "/v1/import":
                return httpx.Response(
                    200, json={"job_id": "j", "status": "done", "imported": 2, "skipped": 0}
                )
            return httpx.Response(
                200,
                content=b'{"id": "m1", "content": "caf\xc3\xa9"}\n{"id": "m2", "content": "b"}',
                headers={"content-type": "application/x-ndjson", "x-dakera-export-skipped": "0"},
            )

        c = _async(handler)
        out = await c.import_memories([{"content": "a"}], agent_id="agent-1")
        assert out.imported_count == 2 and out.job_id == "j"
        req = seen[0]
        assert req.url.params["agent_id"] == "agent-1"
        assert req.url.params["format"] == "jsonl"
        assert req.headers["content-type"].startswith("multipart/form-data; boundary=")
        assert b'{"content": "a"}' in req.content
        export = await c.export_memories(agent_id="agent-1")
        assert [m["id"] for m in export.data] == ["m1", "m2"]
        assert export.data[0]["content"] == "café"
        with pytest.raises(ValueError, match="agent_id"):
            await c.export_memories()
