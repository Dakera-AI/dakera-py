"""Document import (Dakera-AI/dakera#923, server v0.12.3+): the routes of
``/v1/memory/documents`` and the attachment import, sync and async, with the request
and answer shapes of the engine (``crates/common/src/types/document.rs``,
``crates/api/src/documents/routes.rs``)."""

import json

import httpx
import pytest
import responses

from dakera import (
    AsyncDakeraClient,
    ChunkingOptions,
    ChunkStrategy,
    DakeraClient,
    DocumentChunkMeta,
    FeatureNotAvailableError,
    NotFoundError,
    RecallResponse,
    ServerCapabilities,
    document_chunk_meta,
)
from dakera.exceptions import TimeoutError
from dakera.models import RetryConfig

BASE = "http://localhost:3000"

ACCEPTED = {
    "job_id": "job-1",
    "document_id": "manual-v3",
    "agent_id": "docs-bot",
    "status_url": "/v1/memory/documents/jobs/job-1?agent_id=docs-bot",
}


def _job(status, progress=0, message=None, error=None):
    job = {
        "id": "job-1",
        "job_type": "document_import",
        "status": status,
        "created_at": 1792252596,
        "started_at": 1792252597,
        "completed_at": None,
        "progress": progress,
        "message": message,
        "metadata": {
            "namespace": "_dakera_agent_docs-bot",
            "agent_id": "docs-bot",
            "document_id": "manual-v3",
            "media_type": "application/pdf",
            "source_namespace": "library",
            "attachment_ref": "sha256:ab",
        },
    }
    if error is not None:
        job["error"] = error
    return job


CHUNK_META = {
    "id": "manual-v3",
    "title": "Field Manual",
    "media_type": "application/pdf",
    "chunk_index": 41,
    "chunk_count": 212,
    "page_start": 39,
    "page_end": 40,
    "heading_path": ["Field Manual", "Install"],
    "char_start": 80412,
    "char_end": 81733,
    "sha256": "e3b0",
    "bytes": 1843311,
    "pages": 212,
    "created_at": 1792252596,
    "updated_at": 1792252596,
    "fingerprint": "f1",
}

SUMMARY = {
    "document_id": "manual-v3",
    "title": "Field Manual",
    "media_type": "application/pdf",
    "chunk_count": 212,
    "bytes": 1843311,
    "pages": 212,
    "created_at": 1792252596,
    "updated_at": 1792252600,
    "sha256": "e3b0",
    "attachment_ref": "sha256:ab",
}


@pytest.fixture
def rsps():
    with responses.RequestsMock() as r:
        yield r


@pytest.fixture
def client():
    return DakeraClient(BASE, retry_config=RetryConfig(max_retries=1))


def body(call):
    return json.loads(call.request.body)


class TestImportSync:
    def test_import_text_minimal(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/documents", json=ACCEPTED, status=202)
        out = client.import_document("docs-bot", "# Title\n\nHello")
        assert body(rsps.calls[0]) == {"agent_id": "docs-bot", "content": "# Title\n\nHello"}
        assert out.job_id == "job-1" and out.document_id == "manual-v3"
        assert out.status_url.startswith("/v1/memory/documents/jobs/")

    def test_import_text_all_fields(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/documents", json=ACCEPTED, status=202)
        client.import_document(
            "docs-bot",
            "<h1>A</h1>",
            media_type="text/html",
            title="Manual",
            document_id="manual-v3",
            memory_type="semantic",
            session_id="s1",
            importance=0.8,
            tags=("manual",),
            metadata={"source": "wiki"},
            ttl_seconds=3600,
            expires_at=1900000000,
            lang="de",
            chunking=ChunkingOptions(
                max_tokens=200, overlap_tokens=20, strategy=ChunkStrategy.FIXED
            ),
            decompose=True,
        )
        assert body(rsps.calls[0]) == {
            "agent_id": "docs-bot",
            "content": "<h1>A</h1>",
            "media_type": "text/html",
            "title": "Manual",
            "document_id": "manual-v3",
            "memory_type": "semantic",
            "session_id": "s1",
            "importance": 0.8,
            "tags": ["manual"],
            "metadata": {"source": "wiki"},
            "ttl_seconds": 3600,
            "expires_at": 1900000000,
            "lang": "de",
            "decompose": True,
            "chunking": {"max_tokens": 200, "overlap_tokens": 20, "strategy": "fixed"},
        }

    def test_chunking_dict_and_unknown_key(self, rsps, client):
        rsps.add(responses.POST, f"{BASE}/v1/memory/documents", json=ACCEPTED, status=202)
        client.import_document("a", "x", chunking={"strategy": "structure"})
        assert body(rsps.calls[0])["chunking"] == {"strategy": "structure"}
        # ChunkingOptions is `deny_unknown_fields` on the server: refused here first.
        with pytest.raises(TypeError):
            client.import_document("a", "x", chunking={"size": 3})
        assert len(rsps.calls) == 1

    def test_import_attachment(self, rsps, client):
        url = f"{BASE}/v1/namespaces/library/attachments/sha256:ab/document"
        rsps.add(responses.POST, url, json=ACCEPTED, status=202)
        out = client.import_attachment_document(
            "library", "sha256:ab", "docs-bot", document_id="manual-v3", tags=["manual"]
        )
        assert body(rsps.calls[0]) == {
            "agent_id": "docs-bot",
            "document_id": "manual-v3",
            "tags": ["manual"],
        }
        assert out.agent_id == "docs-bot"

    def test_feature_disabled(self, rsps, client):
        rsps.add(
            responses.POST,
            f"{BASE}/v1/memory/documents",
            status=501,
            json={"error": "document import is off", "code": "FEATURE_DISABLED"},
        )
        with pytest.raises(FeatureNotAvailableError):
            client.import_document("a", "x")

    def test_import_not_retried_after_5xx(self, rsps):
        c = DakeraClient(BASE, retry_config=RetryConfig(max_retries=3, base_delay=0))
        rsps.add(responses.POST, f"{BASE}/v1/memory/documents", status=500, json={"error": "x"})
        with pytest.raises(Exception):  # noqa: B017 - ServerError
            c.import_document("a", "x")
        assert len(rsps.calls) == 1


class TestJobsSync:
    def test_get_job(self, rsps, client):
        rsps.add(
            responses.GET,
            f"{BASE}/v1/memory/documents/jobs/job-1",
            json=_job("Running", 54, "chunks_stored 96/212"),
        )
        job = client.get_document_job("job-1", "docs-bot")
        assert "agent_id=docs-bot" in rsps.calls[0].request.url
        assert job.status == "Running" and job.progress == 54 and not job.is_done
        assert job.message == "chunks_stored 96/212"
        assert job.document_id == "manual-v3"
        assert job.agent_id == "docs-bot"
        assert job.attachment_ref == "sha256:ab"
        assert job.media_type == "application/pdf"

    def test_wait_until_failed(self, rsps, client, monkeypatch):
        sleeps = []
        monkeypatch.setattr("dakera.client.time.sleep", sleeps.append)
        url = f"{BASE}/v1/memory/documents/jobs/job-1"
        rsps.add(responses.GET, url, json=_job("Pending"))
        rsps.add(responses.GET, url, json=_job("Running", 10, "chunks_stored 0/5000"))
        rsps.add(
            responses.GET,
            url,
            json=_job("Failed", error={"status": 504, "code": "EXTRACTION_TIMEOUT"}),
        )
        job = client.wait_for_document_job("job-1", "docs-bot", poll_interval=0)
        assert job.is_done and not job.succeeded
        assert (job.error_status, job.error_code) == (504, "EXTRACTION_TIMEOUT")
        # No tight loop: a zero interval is raised to the minimum.
        assert sleeps == [0.1, 0.1]

    def test_wait_times_out(self, rsps, client, monkeypatch):
        clock = iter(range(0, 1000, 5))
        monkeypatch.setattr("dakera.client.time.monotonic", lambda: next(clock))
        monkeypatch.setattr("dakera.client.time.sleep", lambda s: None)
        rsps.add(responses.GET, f"{BASE}/v1/memory/documents/jobs/job-1", json=_job("Running"))
        with pytest.raises(TimeoutError):
            client.wait_for_document_job("job-1", "docs-bot", timeout=12)


class TestReadDeleteSync:
    def test_list_with_cursor_and_iterate(self, rsps, client):
        url = f"{BASE}/v1/memory/documents"
        rsps.add(responses.GET, url, json={"documents": [SUMMARY], "next_cursor": "manual-v3"})
        page = client.list_documents("docs-bot", limit=1)
        assert page.next_cursor == "manual-v3"
        doc = page.documents[0]
        assert (doc.document_id, doc.title, doc.pages, doc.attachment_ref) == (
            "manual-v3",
            "Field Manual",
            212,
            "sha256:ab",
        )
        assert "limit=1" in rsps.calls[0].request.url
        assert "cursor" not in rsps.calls[0].request.url
        rsps.add(responses.GET, url, json={"documents": [SUMMARY], "next_cursor": "manual-v3"})
        rsps.add(responses.GET, url, json={"documents": [{**SUMMARY, "document_id": "z"}]})
        ids = [d.document_id for d in client.iter_documents("docs-bot", page_size=1)]
        assert ids == ["manual-v3", "z"]
        assert "cursor=manual-v3" in rsps.calls[2].request.url

    def test_get_document(self, rsps, client):
        rsps.add(
            responses.GET,
            f"{BASE}/v1/memory/documents/manual-v3",
            json={
                "document": SUMMARY,
                "chunks": [
                    {
                        "id": "doc_1",
                        "chunk_index": 0,
                        "page_start": 1,
                        "page_end": 1,
                        "heading_path": ["Field Manual"],
                        "char_start": 0,
                        "char_end": 900,
                    },
                    {
                        "id": "doc_2",
                        "chunk_index": 1,
                        "heading_path": [],
                        "char_start": 850,
                        "char_end": 1700,
                    },
                ],
            },
        )
        detail = client.get_document("manual-v3", "docs-bot")
        assert "agent_id=docs-bot" in rsps.calls[0].request.url
        assert detail.document.chunk_count == 212
        assert [c.id for c in detail.chunks] == ["doc_1", "doc_2"]
        assert detail.chunks[1].page_start is None

    def test_delete_and_not_found(self, rsps, client):
        url = f"{BASE}/v1/memory/documents/manual-v3"
        rsps.add(responses.DELETE, url, json={"document_id": "manual-v3", "deleted_chunks": 212})
        rsps.add(
            responses.DELETE,
            url,
            status=404,
            json={"error": "Document not found", "code": "NOT_FOUND", "resource": "document"},
        )
        out = client.delete_document("manual-v3", "docs-bot")
        assert out.deleted_chunks == 212
        with pytest.raises(NotFoundError):
            client.delete_document("manual-v3", "docs-bot")


class TestChunkMetadata:
    def test_recalled_chunk_cites_its_document(self):
        r = RecallResponse.from_dict(
            {
                "memories": [
                    {
                        "memory": {
                            "id": "doc_1",
                            "content": "Field Manual › Install\n\nRun the installer.",
                            "memory_type": "episodic",
                            "importance": 0.5,
                            "metadata": {"_dakera_document": CHUNK_META, "source": "x"},
                        },
                        "score": 0.9,
                    },
                    {"memory": {"id": "m2", "content": "plain"}, "score": 0.5},
                ]
            }
        )
        meta = r.memories[0].document
        assert meta is not None
        assert (meta.id, meta.page_start, meta.page_end) == ("manual-v3", 39, 40)
        assert meta.breadcrumb == "Field Manual › Install"
        assert meta.chunk_index == 41 and meta.chunk_count == 212
        assert r.memories[1].document is None
        assert document_chunk_meta(r.memories[0]) == meta

    def test_helper_on_dicts(self):
        assert DocumentChunkMeta.from_memory({"metadata": {"_dakera_document": CHUNK_META}})
        nested = {"memory": {"metadata": {"_dakera_document": CHUNK_META}}}
        assert document_chunk_meta(nested).title == "Field Manual"
        assert document_chunk_meta({"_dakera_document": CHUNK_META}).sha256 == "e3b0"
        assert document_chunk_meta({"metadata": None}) is None
        assert document_chunk_meta({"metadata": {"_dakera_document": "x"}}) is None


class TestCapabilities:
    def test_documents_block(self):
        caps = ServerCapabilities.from_dict(
            {
                "documents": {
                    "enabled": True,
                    "text_media_types": ["text/plain", "text/markdown", "text/html"],
                    "media_types": ["application/pdf", "text/plain"],
                    "max_bytes": 26214400,
                    "max_text_bytes": 20971520,
                    "max_chunks": 5000,
                    "default_chunk_tokens": 400,
                    "default_overlap_tokens": 50,
                    "strategies": ["structure", "fixed"],
                }
            }
        )
        assert caps.supports_documents
        assert caps.documents.max_chunks == 5000
        assert caps.documents.strategies == ["structure", "fixed"]

    def test_old_server(self):
        caps = ServerCapabilities.from_dict({"capabilities_version": 2})
        assert not caps.supports_documents
        assert caps.documents.media_types == []


def _async(handler):
    c = AsyncDakeraClient(BASE, retry_config=RetryConfig(max_retries=1))
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


class TestAsync:
    async def test_import_poll_list_get_delete(self, monkeypatch):
        seen = []
        polls = iter([_job("Running", 50), _job("Completed", 100, "document imported")])

        def handler(request):
            seen.append((request.method, request.url.path, dict(request.url.params)))
            path = request.url.path
            if request.method == "POST":
                seen[-1] = (*seen[-1], json.loads(request.content))
                return httpx.Response(202, json=ACCEPTED)
            if path.endswith("/jobs/job-1"):
                return httpx.Response(200, json=next(polls))
            if request.method == "DELETE":
                return httpx.Response(200, json={"document_id": "d", "deleted_chunks": 3})
            if path == "/v1/memory/documents":
                return httpx.Response(200, json={"documents": [SUMMARY]})
            return httpx.Response(200, json={"document": SUMMARY, "chunks": []})

        async def no_sleep(_):
            return None

        monkeypatch.setattr("dakera.async_client.asyncio.sleep", no_sleep)
        c = _async(handler)
        acc = await c.import_document(
            "docs-bot", "text", media_type="text/markdown", chunking={"max_tokens": 64}
        )
        await c.import_attachment_document("library", "sha256:ab", "docs-bot", title="T")
        job = await c.wait_for_document_job(acc.job_id, "docs-bot")
        assert job.succeeded and job.document_id == "manual-v3"
        docs = [d async for d in c.iter_documents("docs-bot")]
        assert [d.document_id for d in docs] == ["manual-v3"]
        detail = await c.get_document("manual-v3", "docs-bot")
        assert detail.document.title == "Field Manual"
        deleted = await c.delete_document("manual-v3", "docs-bot")
        assert deleted.deleted_chunks == 3
        assert seen[0][3] == {
            "agent_id": "docs-bot",
            "content": "text",
            "media_type": "text/markdown",
            "chunking": {"max_tokens": 64},
        }
        assert seen[1][1] == "/v1/namespaces/library/attachments/sha256:ab/document"
        assert seen[1][3] == {"agent_id": "docs-bot", "title": "T"}
        assert seen[2][2] == {"agent_id": "docs-bot"}
        assert all(s[2].get("agent_id") == "docs-bot" for s in seen if s[0] != "POST")
