# ruff: noqa: E501
"""Server v0.12.0 support: error bodies, Retry-After, health, attachments, records,
per-request ``lang``, namespace config PUT, and the capabilities additions.

Wire shapes are taken from the v0.12.0 server source (``routes/attachments.rs``,
``routes/records.rs``, ``routes/health.rs``, ``routes/capabilities.rs``,
``error.rs``, ``startup.rs``, ``common/types/{attachment,record,mod}.rs``).
"""

import json
from unittest.mock import patch

import httpx
import pytest
import requests
import responses

from dakera import (
    AsyncDakeraClient,
    AttachmentJob,
    AuthorizationError,
    BatchStoreMemoryItem,
    BatchStoreMemoryRequest,
    BlockDType,
    ConflictError,
    DakeraClient,
    ErrorCode,
    FeatureNotAvailableError,
    NotFoundError,
    PayloadTooLargeError,
    RateLimitError,
    Record,
    Representation,
    RepresentationKind,
    ServerCapabilities,
    ServerError,
    ServiceUnavailableError,
    ValidationError,
)
from dakera.exceptions import TimeoutError as DakeraTimeoutError
from dakera.exceptions import error_from_response, parse_retry_after
from dakera.models import RetryConfig

BASE = "http://localhost:3000"
REF = "sha256:" + "ab" * 32


@pytest.fixture
def rsps():
    with responses.RequestsMock() as r:
        yield r


@pytest.fixture
def client():
    return DakeraClient(
        BASE, retry_config=RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
    )


def body_of(call):
    return json.loads(call.request.body)


# ============================================================================
# Error bodies
# ============================================================================


class TestErrorMapping:
    def test_413_quota_vs_oversize(self):
        quota = error_from_response(
            413, {"error": "Quota exceeded", "code": "QUOTA_EXCEEDED", "details": "ns: a"}
        )
        big = error_from_response(413, {"error": "too big", "code": "PAYLOAD_TOO_LARGE"})
        assert isinstance(quota, PayloadTooLargeError) and quota.is_quota
        assert not quota.is_oversize and quota.details == "ns: a"
        assert isinstance(big, PayloadTooLargeError) and big.is_oversize and not big.is_quota
        assert big.code is ErrorCode.PAYLOAD_TOO_LARGE

    def test_501_feature_disabled_and_not_implemented(self):
        off = error_from_response(
            501,
            {"error": "The records API is not enabled on this server",
             "code": "FEATURE_DISABLED", "details": "set DAKERA_RECORDS to enable it"},
        )  # fmt: skip
        assert isinstance(off, FeatureNotAvailableError) and off.is_feature_disabled
        assert "DAKERA_RECORDS" in off.details
        other = error_from_response(501, {"error": "x", "code": "NOT_IMPLEMENTED"})
        assert isinstance(other, FeatureNotAvailableError) and not other.is_feature_disabled
        # a 501 is not a ServerError: it is a configuration answer, never retried
        assert not isinstance(off, ServerError)

    def test_404_resource_and_job_not_found(self):
        e = error_from_response(
            404, {"error": "job unknown", "code": "JOB_NOT_FOUND", "resource": "job"}
        )
        assert isinstance(e, NotFoundError) and e.resource == "job"
        assert e.code is ErrorCode.JOB_NOT_FOUND
        v011 = error_from_response(404, {"error": "nope", "code": "VECTOR_NOT_FOUND"})
        assert v011.resource is None  # v0.11 servers do not send the field

    def test_409_conflict(self):
        assert isinstance(
            error_from_response(409, {"error": "x", "code": "CONFLICT"}), ConflictError
        )

    def test_503_carries_retry_after(self):
        e = error_from_response(
            503, {"error": "Service unavailable", "code": "SERVICE_UNAVAILABLE"}, "7"
        )
        assert isinstance(e, ServiceUnavailableError) and isinstance(e, ServerError)
        assert e.retry_after == 7.0

    def test_other_statuses_and_plain_text_body(self):
        assert isinstance(
            error_from_response(403, {"code": "INSUFFICIENT_SCOPE"}), AuthorizationError
        )
        assert isinstance(error_from_response(400, "bad", None), ValidationError)
        e = error_from_response(504, {"error": "t", "code": "QUERY_TIMEOUT"})
        assert isinstance(e, ServerError) and e.code is ErrorCode.QUERY_TIMEOUT
        assert error_from_response(500, {"code": "BRAND_NEW"}).code is ErrorCode.UNKNOWN
        assert error_from_response(405, {"code": "METHOD_NOT_ALLOWED"}).status_code == 405

    def test_429_retry_after(self):
        e = error_from_response(429, {"code": "RATE_LIMIT_EXCEEDED"}, "12")
        assert isinstance(e, RateLimitError) and e.retry_after == 12

    def test_parse_retry_after(self):
        assert parse_retry_after("5") == 5.0
        assert parse_retry_after(" 30 ") == 30.0
        assert parse_retry_after(None) is None
        assert parse_retry_after("") is None
        assert parse_retry_after("soon") is None
        assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # past date
        assert parse_retry_after("-3") == 0.0


# ============================================================================
# Retry-After honoured
# ============================================================================

OVERLOAD = {"error": "Service unavailable", "code": "SERVICE_UNAVAILABLE", "details": "busy"}


class TestRetryAfter:
    def test_503_waits_retry_after_then_succeeds(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/health", json=OVERLOAD, status=503,
                 headers={"Retry-After": "4"})  # fmt: skip
        rsps.add(responses.GET, f"{BASE}/health", json={"status": "healthy"}, status=200)
        with patch("dakera.client.time.sleep") as sleep:
            assert client.health() == {"status": "healthy"}
        sleep.assert_called_once_with(4.0)

    def test_retry_after_is_capped_at_max_delay(self, rsps):
        c = DakeraClient(BASE, retry_config=RetryConfig(max_retries=2, max_delay=2.0))
        rsps.add(responses.GET, f"{BASE}/health", json=OVERLOAD, status=503,
                 headers={"Retry-After": "600"})  # fmt: skip
        rsps.add(responses.GET, f"{BASE}/health", json={"ok": 1}, status=200)
        with patch("dakera.client.time.sleep") as sleep:
            c.health()
        sleep.assert_called_once_with(2.0)

    def test_503_without_header_uses_backoff(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/health", json=OVERLOAD, status=503)
        rsps.add(responses.GET, f"{BASE}/health", json={"ok": 1}, status=200)
        with patch("dakera.client.time.sleep") as sleep:
            client.health()
        assert sleep.call_args[0][0] == pytest.approx(0.01)

    def test_503_exhausted_raises_with_retry_after(self, rsps, client):
        for _ in range(3):
            rsps.add(responses.GET, f"{BASE}/health", json=OVERLOAD, status=503,
                     headers={"Retry-After": "1"})  # fmt: skip
        with patch("dakera.client.time.sleep"), pytest.raises(ServiceUnavailableError) as ei:
            client.health()
        assert ei.value.retry_after == 1.0 and ei.value.details == "busy"

    def test_429_and_501_and_413(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/health", json={"code": "RATE_LIMIT_EXCEEDED"},
                 status=429, headers={"Retry-After": "3"})  # fmt: skip
        rsps.add(responses.GET, f"{BASE}/health", json={"ok": 1})
        with patch("dakera.client.time.sleep") as sleep:
            client.health()
        sleep.assert_called_once_with(3.0)

    @pytest.mark.parametrize(
        "status, code, exc",
        [
            (501, "FEATURE_DISABLED", FeatureNotAvailableError),
            (413, "PAYLOAD_TOO_LARGE", PayloadTooLargeError),
            (409, "CONFLICT", ConflictError),
        ],
    )
    def test_not_retried(self, rsps, client, status, code, exc):
        rsps.add(responses.GET, f"{BASE}/health", json={"error": "x", "code": code}, status=status)
        with patch("dakera.client.time.sleep") as sleep, pytest.raises(exc):
            client.health()
        sleep.assert_not_called()
        assert len(rsps.calls) == 1

    async def test_async_503_waits_retry_after(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503, json=OVERLOAD, headers={"Retry-After": "5"})
            return httpx.Response(200, json={"status": "healthy"})

        c = _async_client(handler)
        slept = []

        async def fake_sleep(s):
            slept.append(s)

        with patch("dakera.async_client.asyncio.sleep", fake_sleep):
            assert await c.health() == {"status": "healthy"}
        assert slept == [5.0]

    async def test_async_501_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(
                501, json={"error": "off", "code": "FEATURE_DISABLED", "details": "set X"}
            )

        c = _async_client(handler)
        with pytest.raises(FeatureNotAvailableError):
            await c.health()
        assert len(calls) == 1


def _async_client(handler):
    c = AsyncDakeraClient(BASE, retry_config=RetryConfig(max_retries=3, base_delay=0.01))
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE)
    return c


# ============================================================================
# Health: ready / live; 503 is never healthy
# ============================================================================

STARTING = {"ready": False, "version": "0.12.0", "starting": True, "reason": "loading models",
            "downloads": []}  # fmt: skip
READY = {"ready": True, "version": "0.12.0", "checks": {"storage": {"status": "ok"}}}


class TestHealth:
    def test_is_ready_false_on_503_true_on_200(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/health/ready", json=STARTING, status=503,
                 headers={"Retry-After": "5"})  # fmt: skip
        rsps.add(responses.GET, f"{BASE}/health/ready", json=READY)
        assert client.is_ready() is False
        assert client.is_ready() is True
        assert len(rsps.calls) == 2  # one probe each, no retries

    def test_is_ready_false_when_unreachable(self, rsps, client):
        rsps.add(
            responses.GET, f"{BASE}/health/ready", body=requests.exceptions.ConnectionError("down")
        )
        assert client.is_ready() is False

    def test_health_ready_503_is_an_error_not_healthy(self, rsps, client):
        for _ in range(3):
            rsps.add(responses.GET, f"{BASE}/health/ready", json=STARTING, status=503,
                     headers={"Retry-After": "1"})  # fmt: skip
        with patch("dakera.client.time.sleep"), pytest.raises(ServiceUnavailableError):
            client.health_ready()

    def test_health_live(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/health/live",
                 json={"alive": True, "version": "0.12.0", "uptime_seconds": 3, "starting": True})  # fmt: skip
        assert client.health_live()["alive"] is True

    def test_wait_until_ready_polls_with_retry_after(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/health/ready", json=STARTING, status=503,
                 headers={"Retry-After": "2"})  # fmt: skip
        rsps.add(responses.GET, f"{BASE}/health/ready", json=STARTING, status=503)
        rsps.add(responses.GET, f"{BASE}/health/ready", json=READY)
        rsps.add(responses.GET, f"{BASE}/health/ready", json=READY)  # the body fetch
        with patch("dakera.client.time.sleep") as sleep:
            body = client.wait_until_ready(timeout=60, poll_interval=0.5)
        assert body["ready"] is True
        assert [c.args[0] for c in sleep.call_args_list] == [2.0, 0.5]

    def test_wait_until_ready_times_out(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/health/ready", json=STARTING, status=503)
        with pytest.raises(DakeraTimeoutError):
            client.wait_until_ready(timeout=0.05, poll_interval=0.02)

    async def test_async_wait_until_ready(self):
        n = []

        def handler(request):
            n.append(1)
            if len(n) < 3:
                return httpx.Response(503, json=STARTING, headers={"Retry-After": "1"})
            return httpx.Response(200, json=READY)

        c = _async_client(handler)

        async def fake_sleep(s):
            pass

        with patch("dakera.async_client.asyncio.sleep", fake_sleep):
            assert (await c.wait_until_ready(timeout=30))["ready"] is True
        assert await c.is_ready() is True


# ============================================================================
# Capabilities: v0.12 additions
# ============================================================================

CAPS = {
    "capabilities_version": 1,
    "server_version": "0.12.0",
    "scoring": {
        "strategy": "late-interaction",
        "strategies_accepted": "single-vector, late-interaction",
        "late_interaction": {"enabled": True, "model_supported": True, "lane": "text",
                             "token_slot": "colbert", "fde_slot": "colbert.fde", "fde_dim": 4096},
    },
    "attachments": {
        "enabled": True,
        "max_bytes": 26214400,
        "transcription": {"model": "whisper-tiny.en", "models": ["whisper-tiny.en"],
                          "media_types": ["audio/wav"], "languages": ["en"], "sample_rate_hz": 16000},
    },
    "vision": {"enabled": False, "model": "colmodernvbert", "models": ["colmodernvbert"],
               "media_types": ["image/png"], "dimension": 128, "tile_size": 512},
    "unreadable_records": 2,
    "late_interaction_stats": {"searches": 9},
}  # fmt: skip


class TestCapabilitiesV012:
    def test_new_sections_parse(self):
        caps = ServerCapabilities.from_dict(CAPS)
        assert caps.scoring.strategy == "late-interaction"
        assert caps.scoring.strategies_accepted == ["single-vector", "late-interaction"]
        assert (
            caps.scoring.late_interaction.enabled
            and caps.scoring.late_interaction.fde_slot == "colbert.fde"
        )
        assert caps.supports_attachments and caps.attachments.max_bytes == 26214400
        assert caps.attachments.transcription.model == "whisper-tiny.en"
        assert caps.attachments.transcription.media_types == ["audio/wav"]
        assert not caps.supports_vision and caps.vision.media_types == ["image/png"]
        assert caps.unreadable_records == 2 and caps.late_interaction_stats == {"searches": 9}
        assert caps.scoring.late_interaction.raw["fde_dim"] == 4096  # unknown field kept

    def test_v011_style_document_defaults(self):
        caps = ServerCapabilities.from_dict({"server_version": "0.11.108"})
        assert caps.scoring.strategy == "single-vector"
        assert not caps.supports_attachments and not caps.supports_vision
        assert caps.unreadable_records == 0

    def test_wrong_shapes_do_not_raise(self):
        caps = ServerCapabilities.from_dict({"scoring": "x", "attachments": [], "vision": None})
        assert caps.attachments.max_bytes == 0


# ============================================================================
# Attachments
# ============================================================================

NS = "_dakera_agent_a1"


class TestAttachments:
    def test_upload_bytes_raw_body_and_content_type(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/{NS}/attachments",
                 json={"attachment_ref": REF, "content_type": "audio/wav", "size_bytes": 4,
                       "created": True}, status=201)  # fmt: skip
        out = client.upload_attachment(NS, b"RIFF", content_type="audio/wav")
        req = rsps.calls[0].request
        assert req.body == b"RIFF"
        assert req.headers["Content-Type"] == "audio/wav"
        assert out.attachment_ref == REF and out.created and out.size_bytes == 4

    def test_upload_path_guesses_type_and_dedup(self, rsps, client, tmp_path):
        f = tmp_path / "note.wav"
        f.write_bytes(b"RIFFdata")
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/u/attachments",
                 json={"attachment_ref": REF, "content_type": "audio/x-wav", "size_bytes": 8,
                       "created": False}, status=200)  # fmt: skip
        out = client.upload_attachment("u", f)
        assert rsps.calls[0].request.body == b"RIFFdata"
        assert rsps.calls[0].request.headers["Content-Type"].startswith("audio/")
        assert out.created is False
        with open(f, "rb") as fh:
            rsps.add(responses.POST, f"{BASE}/v1/namespaces/u/attachments",
                     json={"attachment_ref": REF, "content_type": "x", "size_bytes": 8, "created": True},
                     status=201)  # fmt: skip
            client.upload_attachment("u", fh)
        assert rsps.calls[1].request.body == b"RIFFdata"

    def test_upload_oversize_is_413_not_retried(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/u/attachments", status=413,
                 json={"error": "body over DAKERA_ATTACHMENT_MAX_BYTES (26214400 bytes)",
                       "code": "PAYLOAD_TOO_LARGE"})  # fmt: skip
        with pytest.raises(PayloadTooLargeError) as ei:
            client.upload_attachment("u", b"x" * 10)
        assert ei.value.is_oversize
        assert len(rsps.calls) == 1

    def test_upload_feature_disabled(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/u/attachments", status=501,
                 json={"error": "The attachments API is not enabled on this server",
                       "code": "FEATURE_DISABLED", "details": "set DAKERA_ATTACHMENTS to enable it"})  # fmt: skip
        with pytest.raises(FeatureNotAvailableError) as ei:
            client.upload_attachment("u", b"x")
        assert "DAKERA_ATTACHMENTS" in ei.value.details

    def test_list_download_delete(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/u/attachments",
                 json={"attachments": [{"attachment_ref": REF, "content_type": "audio/wav",
                                        "size_bytes": 4}]})  # fmt: skip
        items = client.list_attachments("u")
        assert items[0].attachment_ref == REF and items[0].size_bytes == 4

        rsps.add(responses.GET, f"{BASE}/v1/namespaces/u/attachments/{REF}", body=b"\x00\x01\xff",
                 content_type="audio/wav", headers={"ETag": '"abcd"'})  # fmt: skip
        got = client.download_attachment("u", REF)
        assert got.data == b"\x00\x01\xff" and got.content_type == "audio/wav"
        assert got.etag == "abcd" and got.size_bytes == 3

        rsps.add(responses.DELETE, f"{BASE}/v1/namespaces/u/attachments/{REF}", status=204)
        assert client.delete_attachment("u", REF) is None

    def test_delete_conflict_when_referenced(self, rsps, client):
        rsps.add(responses.DELETE, f"{BASE}/v1/namespaces/u/attachments/{REF}", status=409,
                 json={"error": "referenced", "code": "CONFLICT", "details": "forget the memory"})  # fmt: skip
        with pytest.raises(ConflictError):
            client.delete_attachment("u", REF)

    def test_download_missing_is_404(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/u/attachments/{REF}", status=404,
                 json={"error": "Attachment not found", "code": "VECTOR_NOT_FOUND",
                       "resource": "attachment"})  # fmt: skip
        with pytest.raises(NotFoundError) as ei:
            client.download_attachment("u", REF)
        assert ei.value.resource == "attachment"

    def test_transcribe_and_job(self, rsps, client):
        jid = "job_1a2b3c4d_0"
        status_url = f"/v1/namespaces/u/attachments/{REF}/transcribe/{jid}"
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/u/attachments/{REF}/transcribe", status=202,
                 json={"job_id": jid, "attachment_ref": REF, "agent_id": "a1", "memory_id": "mem_1",
                       "model": "whisper-tiny.en", "status_url": status_url})  # fmt: skip
        acc = client.transcribe_attachment(
            "u", REF, "a1", tags=["voice"], importance=0.7, lang="de"
        )
        assert body_of(rsps.calls[0]) == {"agent_id": "a1", "tags": ["voice"], "importance": 0.7,
                                          "lang": "de"}  # fmt: skip
        assert acc.job_id == jid and acc.memory_id == "mem_1" and acc.status_url == status_url

        running = {
            "id": jid,
            "job_type": "transcription",
            "status": "Running",
            "created_at": 1,
            "progress": 40,
            "message": "transcribing",
            "metadata": {"namespace": "u"},
        }
        done = dict(running, status="Completed", progress=100, message="memory mem_1 stored")
        rsps.add(responses.GET, f"{BASE}{status_url}", json=running)
        rsps.add(responses.GET, f"{BASE}{status_url}", json=done)
        with patch("dakera.client.time.sleep") as sleep:
            job = client.wait_for_transcription("u", REF, jid, poll_interval=0.25)
        assert job.succeeded and job.is_done and job.progress == 100
        sleep.assert_called_once_with(0.25)

    def test_failed_job_carries_error(self):
        job = AttachmentJob.from_dict(
            {
                "id": "j",
                "job_type": "transcription",
                "status": "Failed",
                "progress": 80,
                "message": "no speech",
                "error": {"status": 400, "code": "INVALID_REQUEST"},
            }  # fmt: skip
        )
        assert job.is_done and not job.succeeded
        assert (job.error_status, job.error_code) == (400, "INVALID_REQUEST")

    def test_job_unknown_after_restart(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/u/attachments/{REF}/transcribe/job_x_9",
                 status=404, json={"error": "job unknown: the server restarted",
                                   "code": "JOB_NOT_FOUND", "resource": "job"})  # fmt: skip
        with pytest.raises(NotFoundError) as ei:
            client.get_transcription_job("u", REF, "job_x_9")
        assert ei.value.code is ErrorCode.JOB_NOT_FOUND

    def test_wait_times_out(self, rsps, client):
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/u/attachments/{REF}/transcribe/j",
                 json={"id": "j", "job_type": "t", "status": "Pending"})  # fmt: skip
        with pytest.raises(DakeraTimeoutError):
            client.wait_for_transcription("u", REF, "j", timeout=0.05, poll_interval=0.02)

    def test_index_image(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/u/attachments/{REF}/index", status=202,
                 json={"job_id": "job_1_1", "attachment_ref": REF, "agent_id": "a1",
                       "memory_id": "m", "model": "colmodernvbert",
                       "status_url": f"/v1/namespaces/u/attachments/{REF}/index/job_1_1"})  # fmt: skip
        acc = client.index_attachment("u", REF, "a1", content="page 1", id="p1")
        assert body_of(rsps.calls[0]) == {"agent_id": "a1", "content": "page 1", "id": "p1"}
        assert acc.model == "colmodernvbert"
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/u/attachments/{REF}/index/job_1_1",
                 json={"id": "job_1_1", "job_type": "image_index", "status": "Completed",
                       "progress": 100})  # fmt: skip
        assert client.get_index_job("u", REF, "job_1_1").succeeded
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/u/attachments/{REF}/index/job_1_1",
                 json={"id": "job_1_1", "job_type": "image_index", "status": "Completed"})  # fmt: skip
        assert client.wait_for_index("u", REF, "job_1_1").succeeded

    def test_index_without_vision_is_501(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/u/attachments/{REF}/index", status=501,
                 json={"error": "The vision API is not enabled on this server",
                       "code": "FEATURE_DISABLED", "details": "set DAKERA_VISION to enable it"})  # fmt: skip
        with pytest.raises(FeatureNotAvailableError):
            client.index_attachment("u", REF, "a1")

    def test_store_memory_with_attachment_ref_and_lang(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/store",
                 json={"memory": {"id": "m1"}, "embedding_time_ms": 3})  # fmt: skip
        client.store_memory("a1", "hallo", lang="de", attachment_ref=REF)
        b = body_of(rsps.calls[0])
        assert b["lang"] == "de" and b["attachment_ref"] == REF and b["agent_id"] == "a1"

    def test_batch_store_lang_and_item_attachment(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memories/store/batch",
                 json={"stored": [], "stored_count": 0, "total_embedding_time_ms": 0})  # fmt: skip
        req = BatchStoreMemoryRequest(
            "a1",
            [BatchStoreMemoryItem("x", attachment_ref=REF), BatchStoreMemoryItem("y")],
            lang="fr",
        )
        client.store_memories_batch(req)
        b = body_of(rsps.calls[0])
        assert b["lang"] == "fr" and b["memories"][0]["attachment_ref"] == REF
        assert "attachment_ref" not in b["memories"][1]

    async def test_async_attachment_flow(self):
        seen = []

        def handler(request):
            seen.append(request)
            p = request.url.path
            if request.method == "POST" and p.endswith("/attachments"):
                assert request.content == b"RIFF"
                assert request.headers["content-type"] == "audio/wav"
                return httpx.Response(
                    201,
                    json={
                        "attachment_ref": REF,
                        "content_type": "audio/wav",
                        "size_bytes": 4,
                        "created": True,
                    },
                )
            if request.method == "GET" and p.endswith(REF):
                return httpx.Response(
                    200,
                    content=b"RIFF",
                    headers={"Content-Type": "audio/wav", "ETag": f'"{REF[7:]}"'},
                )
            if request.method == "POST" and p.endswith("/transcribe"):
                return httpx.Response(
                    202,
                    json={
                        "job_id": "job_1_0",
                        "attachment_ref": REF,
                        "agent_id": "a1",
                        "memory_id": "m",
                        "model": "w",
                        "status_url": "/x",
                    },
                )
            if request.method == "GET" and p.endswith("/transcribe/job_1_0"):
                return httpx.Response(
                    200,
                    json={"id": "job_1_0", "job_type": "t", "status": "Completed", "progress": 100},
                )
            if request.method == "DELETE":
                return httpx.Response(409, json={"error": "ref", "code": "CONFLICT"})
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "attachments": [
                            {"attachment_ref": REF, "content_type": "a", "size_bytes": 1}
                        ]
                    },
                )
            raise AssertionError(request)

        c = _async_client(handler)
        up = await c.upload_attachment("u", b"RIFF", "audio/wav")
        assert up.attachment_ref == REF
        dl = await c.download_attachment("u", REF)
        assert dl.data == b"RIFF" and dl.etag == REF[7:]
        acc = await c.transcribe_attachment("u", REF, "a1")
        assert (await c.wait_for_transcription("u", REF, acc.job_id)).succeeded
        assert (await c.get_transcription_job("u", REF, acc.job_id)).succeeded
        with pytest.raises(ConflictError):
            await c.delete_attachment("u", REF)
        assert len(await c.list_attachments("u")) == 1

    async def test_async_index_and_wait(self):
        def handler(request):
            if request.method == "POST":
                return httpx.Response(
                    202,
                    json={
                        "job_id": "job_1_2",
                        "attachment_ref": REF,
                        "agent_id": "a",
                        "memory_id": "m",
                        "model": "v",
                        "status_url": "/s",
                    },
                )
            return httpx.Response(
                200, json={"id": "job_1_2", "job_type": "i", "status": "Completed"}
            )

        c = _async_client(handler)
        acc = await c.index_attachment("u", REF, "a", content="cap")
        assert (await c.wait_for_index("u", REF, acc.job_id)).succeeded
        assert (await c.get_index_job("u", REF, acc.job_id)).is_done


# ============================================================================
# Records
# ============================================================================


class TestRecords:
    def test_upsert_with_representations(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/docs/records", json={"upserted_count": 2})
        rec = Record(
            id="r1",
            values=[0.1, 0.2, 0.3, 0.4],
            representations=[
                Representation(
                    "tokens", [[0.1, 0.2], [0.3, 0.4]],
                    kind=RepresentationKind.TOKEN_MULTIVECTOR, store_as=BlockDType.F16,
                ),
                Representation("alt", [[1.0, 2.0]], kind="future_kind", model="bge-m3", store_as="i8"),
            ],
            metadata={"source": "demo"},
            ttl_seconds=60,
        )  # fmt: skip
        out = client.upsert_records("docs", [rec, {"id": "r2", "values": [1, 2, 3, 4]}])
        assert out.upserted_count == 2
        b = body_of(rsps.calls[0])
        assert b["records"][0] == {
            "id": "r1",
            "values": [0.1, 0.2, 0.3, 0.4],
            "representations": [
                {"name": "tokens", "kind": "token_multivector", "vectors": [[0.1, 0.2], [0.3, 0.4]],
                 "store_as": "f16"},
                {"name": "alt", "kind": "future_kind", "vectors": [[1.0, 2.0]], "store_as": "i8",
                 "model": "bge-m3"},
            ],
            "metadata": {"source": "demo"},
            "ttl_seconds": 60,
        }  # fmt: skip
        assert b["records"][1] == {"id": "r2", "values": [1, 2, 3, 4]}

    def test_get_record_manifest_and_vectors(self, rsps, client):
        view = {
            "id": "r1", "dimension": 4,
            "representations": [{"name": "tokens", "kind": "token_multivector", "dim": 2,
                                 "count": 2, "dtype": "f16", "bytes": 8, "model": "bge-m3"},
                                {"name": "x", "kind": "from_the_future", "dim": 1, "count": 1,
                                 "dtype": "e4m3", "bytes": 1}],
            "unsupported_representations": 1,
            "metadata": {"source": "demo"},
        }  # fmt: skip
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/docs/records/r1", json=view)
        rv = client.get_record("docs", "r1")
        assert "include_vectors" not in rsps.calls[0].request.url
        assert rv.values is None and rv.dimension == 4 and rv.unsupported_representations == 1
        rep = rv.representations[0]
        assert rep.kind is RepresentationKind.TOKEN_MULTIVECTOR and rep.dtype is BlockDType.F16
        assert rep.vectors is None and rep.bytes == 8
        assert not rv.representations[1].kind.is_known  # lenient

        full = dict(view, values=[1, 2, 3, 4])
        full["representations"] = [
            dict(view["representations"][0], vectors=[[0.1, 0.2], [0.3, 0.4]])
        ]
        rsps.add(responses.GET, f"{BASE}/v1/namespaces/docs/records/r1", json=full)
        rv = client.get_record("docs", "r1", include_vectors=True)
        assert "include_vectors=true" in rsps.calls[1].request.url
        assert rv.values == [1, 2, 3, 4] and rv.representations[0].vectors[1] == [0.3, 0.4]

    def test_records_disabled_and_too_big(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/docs/records", status=501,
                 json={"error": "The records API is not enabled on this server",
                       "code": "FEATURE_DISABLED", "details": "set DAKERA_RECORDS to enable it"})  # fmt: skip
        with pytest.raises(FeatureNotAvailableError):
            client.upsert_records("docs", [{"id": "a", "values": [1.0]}])
        rsps.add(responses.POST, f"{BASE}/v1/namespaces/docs/records", status=413,
                 json={"error": "records[0] (id 'a'): too many vectors", "code": "PAYLOAD_TOO_LARGE"})  # fmt: skip
        with pytest.raises(PayloadTooLargeError):
            client.upsert_records("docs", [{"id": "a", "values": [1.0]}])

    async def test_async_records(self):
        def handler(request):
            if request.method == "POST":
                assert json.loads(request.content)["records"][0]["id"] == "r1"
                return httpx.Response(200, json={"upserted_count": 1})
            assert request.url.params["include_vectors"] == "true"
            return httpx.Response(200, json={"id": "r1", "dimension": 2, "values": [1, 2]})

        c = _async_client(handler)
        assert (await c.upsert_records("d", [Record("r1", [1.0, 2.0])])).upserted_count == 1
        assert (await c.get_record("d", "r1", include_vectors=True)).values == [1, 2]


# ============================================================================
# Per-request lang, namespace config PUT
# ============================================================================


class TestLangAndConfig:
    def test_lang_on_update_recall_search_extract(self, rsps, client):
        rsps.add(responses.PUT, f"{BASE}/v1/memory/update/m1", json={"id": "m1"})
        rsps.add(
            responses.POST, f"{BASE}/v1/memory/recall", json={"memories": [], "total_found": 0}
        )
        rsps.add(responses.POST, f"{BASE}/v1/memory/search", json={"memories": []})
        rsps.add(responses.POST, f"{BASE}/v1/memories/extract", json={"entities": []})
        client.update_memory("a1", "m1", content="x", lang="es")
        client.recall("a1", "q", lang="it")
        client.search_memories("a1", "q", lang="pt-BR")
        client.extract_entities("Alice", lang="nl")
        langs = [body_of(c).get("lang") for c in rsps.calls]
        assert langs == ["es", "it", "pt-BR", "nl"]
        assert body_of(rsps.calls[3])["content"] == "Alice"

    def test_lang_omitted_by_default(self, rsps, client):
        rsps.add(
            responses.POST, f"{BASE}/v1/memory/recall", json={"memories": [], "total_found": 0}
        )
        client.recall("a1", "q")
        assert "lang" not in body_of(rsps.calls[0])

    def test_unsupported_lang_is_validation_error(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/store", status=400,
                 json={"error": "lang 'xx' unsupported; supported: de, en", "code": "INVALID_REQUEST"})  # fmt: skip
        with pytest.raises(ValidationError):
            client.store_memory("a1", "x", lang="xx")

    def test_put_config_replaces_and_clears(self, rsps, client):
        rsps.add(responses.PUT, f"{BASE}/v1/namespaces/n/config",
                 json={"namespace": "n", "extract_entities": True, "entity_types": []})  # fmt: skip
        out = client.replace_namespace_ner_config("n", True)
        assert body_of(rsps.calls[0]) == {"extract_entities": True, "entity_types": []}
        assert out["entity_types"] == []

    def test_patch_still_sends_explicit_empty_list(self, rsps, client):
        rsps.add(responses.PATCH, f"{BASE}/v1/namespaces/n/config",
                 json={"namespace": "n", "extract_entities": True, "entity_types": []})  # fmt: skip
        client.configure_namespace_ner("n", True, entity_types=[])
        assert body_of(rsps.calls[0])["entity_types"] == []

    async def test_async_lang_extract_and_put(self):
        seen = {}

        def handler(request):
            seen[request.method + request.url.path] = json.loads(request.content)
            return httpx.Response(
                200, json={"entities": [], "id": "m", "memories": [], "total_found": 0}
            )

        c = _async_client(handler)
        await c.extract_entities("Alice", lang="de")
        await c.replace_namespace_ner_config("n", False, ["person"])
        await c.store_memory("a", "x", lang="fr", attachment_ref=REF)
        await c.update_memory("a", "m", content="y", lang="es")
        await c.recall("a", "q", lang="it")
        await c.search_memories("a", "q", lang="nl")
        assert seen["POST/v1/memories/extract"] == {"content": "Alice", "lang": "de"}
        assert seen["PUT/v1/namespaces/n/config"] == {
            "extract_entities": False,
            "entity_types": ["person"],
        }
        assert seen["POST/v1/memory/store"]["attachment_ref"] == REF
        assert seen["PUT/v1/memory/update/m"]["lang"] == "es"
        assert seen["POST/v1/memory/recall"]["lang"] == "it"
        assert seen["POST/v1/memory/search"]["lang"] == "nl"
