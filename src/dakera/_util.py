"""Internal helpers shared by the sync and async clients."""

import codecs
import csv
import io
import json
import mimetypes
import os
import re
import uuid
import warnings
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from dakera.models import (
        AuditExportResponse,
        DakeraEvent,
        MemoryEvent,
        MemoryExportResponse,
    )

# Methods whose repetition has the effect of one request (RFC 9110 §9.2.2). Only
# these are retried after a failure that may have reached the server (a 5xx, a
# read timeout, a dropped connection); every method is retried on a 429 / 503
# (the server refused the request) and on a failed connect (nothing was sent).
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})

# POST routes that only read: retrying them after a 5xx cannot write twice.
_READ_ONLY_POST = re.compile(
    r"^/(?:"
    r"v1/memory/(?:recall|search)"
    r"|v1/memories/(?:recall/batch|extract)"
    r"|v1/namespaces/[^/]+/(?:query|batch-query|query-text|batch-query-text|fulltext/search"
    r"|hybrid|unified-query|aggregate|vectors/count|multi-vector|explain)"
    r"|v1/knowledge/(?:graph|graph/full|network/cross-agent)"
    r"|v1/(?:route|extract)"
    r"|(?:v1/)?admin/quotas/[^/]+/check"
    r")$"
)


def _retry_safe(method: str, path: str) -> bool:
    """Whether repeating ``method path`` after a failure that may have reached the
    server is harmless: an idempotent method or a read-only ``POST``."""
    method = method.upper()
    if method in _IDEMPOTENT_METHODS:
        return True
    return method == "POST" and bool(_READ_ONLY_POST.match(path.split("?", 1)[0]))


def _dakera_event(data: str) -> "DakeraEvent | None":
    """A :class:`DakeraEvent` from one SSE ``data`` payload; ``None`` if it is not one."""
    from dakera.models import DakeraEvent

    try:
        return DakeraEvent.from_dict(json.loads(data))
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _memory_event(data: str) -> "MemoryEvent | None":
    """A :class:`MemoryEvent` from one SSE ``data`` payload; ``None`` if it is not one."""
    from dakera.models import MemoryEvent

    try:
        return MemoryEvent.from_dict(json.loads(data))
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _redacted_repr(obj: Any) -> str:
    """``repr`` of a client: its class and server URL, never its API key."""
    key = "set" if getattr(obj, "api_key", None) else "none"
    return f"{type(obj).__name__}(base_url={getattr(obj, 'base_url', '?')!r}, api_key=<{key}>)"


def _not_sent(method: str, param: str, instead: str | None = None) -> None:
    """Warn that ``param`` of ``method`` is not a field the server reads and is not sent."""
    hint = f"; use {instead}" if instead else ""
    warnings.warn(
        f"{method}(): `{param}` is not a field the server accepts and is not sent{hint}",
        DeprecationWarning,
        stacklevel=3,
    )


def _create_key_body(
    name: str,
    scope: str,
    namespaces: list[str] | None,
    expires_in_days: int | None,
    permissions: list[str] | None,
    expires_at: str | None,
) -> dict[str, Any]:
    """Body of ``POST /admin/keys``: exactly the server's ``CreateKeyRequest``.

    The server refuses an unknown field with ``422`` (from v0.12.3), so the
    deprecated ``permissions`` / ``expires_at`` arguments are never sent.
    """
    if permissions is not None:
        _not_sent("create_key", "permissions", "scope=")
    if expires_at is not None:
        _not_sent("create_key", "expires_at", "expires_in_days=")
    body: dict[str, Any] = {"name": name, "scope": scope}
    if namespaces is not None:
        body["namespaces"] = list(namespaces)
    if expires_in_days is not None:
        body["expires_in_days"] = expires_in_days
    return body


def _create_namespace_key_body(
    name: str, scope: str, extra_namespaces: list[str] | None, expires_in_days: int | None
) -> dict[str, Any]:
    """Body of ``POST /v1/namespaces/{ns}/keys``: exactly ``CreateNamespaceKeyRequest``."""
    body: dict[str, Any] = {"name": name, "scope": scope}
    if extra_namespaces is not None:
        body["extra_namespaces"] = list(extra_namespaces)
    if expires_in_days is not None:
        body["expires_in_days"] = expires_in_days
    return body


def _create_backup_body(
    name: str | None,
    backup_type: str | None,
    namespaces: list[str] | None,
    encrypt: bool | None,
    compression: str | None,
) -> dict[str, Any]:
    """Body of ``POST /admin/backups``: exactly the server's ``CreateBackupRequest``
    (``name`` required; ``backup_type`` ``full`` / ``incremental`` / ``snapshot``;
    ``compression`` ``none`` / ``zstd`` / ``lz4``)."""
    if not isinstance(name, str) or not name:
        raise ValueError("a backup needs a name: the server requires one")
    body: dict[str, Any] = {"name": name}
    if backup_type is not None:
        body["backup_type"] = getattr(backup_type, "value", backup_type)
    if namespaces is not None:
        body["namespaces"] = list(namespaces)
    if encrypt is not None:
        body["encrypt"] = encrypt
    if compression is not None:
        body["compression"] = getattr(compression, "value", compression)
    return body


def _import_upload(data: Any, fmt: str) -> tuple[bytes, dict[str, str]]:
    """Multipart body and headers of ``POST /v1/import``.

    The server reads the first file field of a ``multipart/form-data`` body.
    ``data`` is sent as is when it is ``bytes`` / ``str``; otherwise it is
    serialised for ``fmt``: ``jsonl`` one JSON object per line, ``mem0``
    ``{"memories": [...]}``, ``zep`` ``{"messages": [...]}`` (a dict is sent as
    given). ``csv`` needs the CSV text.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        payload = bytes(data)
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    elif fmt == "csv":
        raise TypeError("import_memories(format='csv') needs the CSV text (str or bytes)")
    elif fmt == "jsonl":
        items = [data] if isinstance(data, dict) else list(data)
        payload = "\n".join(json.dumps(item) for item in items).encode("utf-8")
    elif isinstance(data, dict):
        payload = json.dumps(data).encode("utf-8")
    else:
        wrapper = "messages" if fmt == "zep" else "memories"
        payload = json.dumps({wrapper: list(data)}).encode("utf-8")
    ext = {"csv": "csv", "jsonl": "jsonl"}.get(fmt, "json")
    ctype = {"csv": "text/csv", "jsonl": "application/x-ndjson"}.get(fmt, "application/json")
    boundary = f"dakera-{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="import.{ext}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    body = head + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def _export_response(text: str, fmt: str, skipped: str | None) -> "MemoryExportResponse":
    """Build a :class:`MemoryExportResponse` from the body of ``GET /v1/export``.

    The server answers the file itself: JSON lines, CSV, or the Mem0 / Zep JSON
    document. ``data`` is the memories as dicts; ``raw`` the text as received.
    """
    from dakera.models import MemoryExportResponse

    items: list[dict[str, Any]]
    if fmt == "csv":
        items = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    elif fmt in ("mem0", "zep"):
        doc = json.loads(text) if text.strip() else {}
        raw_items = doc.get("messages" if fmt == "zep" else "memories", [])
        items = raw_items if isinstance(raw_items, list) else []
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    try:
        skipped_count = int(skipped) if skipped is not None else 0
    except ValueError:
        skipped_count = 0
    return MemoryExportResponse(
        data=items, format=fmt, count=len(items), raw=text, skipped=skipped_count
    )


class _SSEDecoder:
    """Incremental ``text/event-stream`` decoder.

    Feed it the raw bytes as they arrive; it returns the ``data`` payload of
    every complete event. The stream is UTF-8 whatever the ``Content-Type``
    says (a multi-byte character may be split across chunks), lines end with
    LF, CRLF or CR, ``:`` lines are comments (keep-alives), a single space after
    ``data:`` is removed and multi-line data is joined with ``\n``. An event
    larger than ``max_event_bytes`` is an error rather than unbounded memory.
    """

    def __init__(self, max_event_bytes: int = 16 * 1024 * 1024) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self._data: list[str] = []
        self._size = 0
        self._max = max_event_bytes

    def feed(self, chunk: bytes | str) -> list[str]:
        """Add ``chunk``; return the data of the events it completed."""
        text = chunk if isinstance(chunk, str) else self._decoder.decode(chunk)
        buf = self._pending + text
        # A CR at the very end may be the first half of a CRLF: keep it.
        keep_cr = buf.endswith("\r")
        if keep_cr:
            buf = buf[:-1]
        lines = buf.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self._pending = lines.pop() + ("\r" if keep_cr else "")
        if len(self._pending) + self._size > self._max:
            raise ValueError(f"SSE event larger than {self._max} bytes")
        events: list[str] = []
        for line in lines:
            if line == "":
                if self._data:
                    events.append("\n".join(self._data))
                self._data = []
                self._size = 0
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                self._data.append(value)
                self._size += len(value)
                if self._size > self._max:
                    raise ValueError(f"SSE event larger than {self._max} bytes")
        return events


def _attachment_body(data: bytes | str | os.PathLike[str] | BinaryIO) -> tuple[bytes, str]:
    """Bytes and a guessed media type for an attachment upload."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data), "application/octet-stream"
    if isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        with open(path, "rb") as fh:
            body = fh.read()
        return body, mimetypes.guess_type(path)[0] or "application/octet-stream"
    name = getattr(data, "name", None)
    guessed = mimetypes.guess_type(name)[0] if isinstance(name, str) else None
    return data.read(), guessed or "application/octet-stream"


def _memory_job_body(
    agent_id: str,
    content: str | None,
    memory_type: str | None,
    session_id: str | None,
    importance: float | None,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    ttl_seconds: int | None,
    expires_at: int | None,
    id: str | None,
    lang: str | None,
) -> dict[str, Any]:
    """Body of the transcribe / index routes: the memory's fields, minus its text."""
    body: dict[str, Any] = {"agent_id": agent_id}
    for key, value in (
        ("content", content),
        ("memory_type", memory_type),
        ("session_id", session_id),
        ("importance", importance),
        ("tags", tags),
        ("metadata", metadata),
        ("ttl_seconds", ttl_seconds),
        ("expires_at", expires_at),
        ("id", id),
        ("lang", lang),
    ):
        if value is not None:
            body[key] = value
    return body


def _audit_export_response(result: Any, fmt: str) -> "AuditExportResponse":
    """Build an :class:`AuditExportResponse` from ``GET /v1/audit/export``.

    The server answers ``{"events": [...], "count": n}`` for ``format=json`` and the
    CSV text for ``format=csv``; ``jsonl`` is rendered here, one event per line.
    """
    import json

    from dakera.models import AuditExportResponse

    if isinstance(result, dict):
        events = result.get("events", [])
        count = int(result.get("count", len(events)))
        data = "\n".join(json.dumps(e) for e in events) if fmt == "jsonl" else json.dumps(events)
        return AuditExportResponse(data=data, format=fmt, count=count)
    text = result if isinstance(result, str) else ""
    rows = [line for line in text.splitlines() if line]
    return AuditExportResponse(data=text, format="csv", count=max(0, len(rows) - 1))


def _key_update_body(
    name: str | None, namespaces: list[str] | None, all_namespaces: bool
) -> dict[str, Any]:
    """Body of ``PATCH /admin/keys/{id}`` and ``PATCH /v1/namespaces/{ns}/keys/{id}``.

    Only the fields that change are sent: an absent ``namespaces`` leaves the
    key's list unchanged, ``null`` (``all_namespaces=True``) grants every
    namespace, ``[]`` none.
    """
    if all_namespaces and namespaces is not None:
        raise ValueError("pass either namespaces or all_namespaces=True, not both")
    body: dict[str, Any] = {}
    if name is not None:
        body["name"] = name
    if all_namespaces:
        body["namespaces"] = None
    elif namespaces is not None:
        body["namespaces"] = list(namespaces)
    if not body:
        raise ValueError("a key update needs name, namespaces or all_namespaces=True")
    return body


def _store_memory_result(response: Any) -> Any:
    """The memory of a ``POST /v1/memory/store`` answer.

    The server wraps it in ``{"memory": {...}, "embedding_time_ms": ...}``;
    v0.12.2 adds ``session_state`` (``"active"`` / ``"ended"``) when the memory
    went into a started session, which is copied onto the returned dict.
    """
    if isinstance(response, dict) and isinstance(response.get("memory"), dict):
        memory = dict(response["memory"])
        if response.get("session_state") is not None:
            memory["session_state"] = response["session_state"]
        return memory
    return response


def _listing_params(
    params: dict[str, Any],
    *,
    limit: int | None = None,
    offset: int | None = None,
    include_derived: bool | None = None,
    content_preview_chars: int | None = None,
) -> dict[str, Any]:
    """Add the optional memory-listing query parameters that are set."""
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    if include_derived is not None:
        params["include_derived"] = str(include_derived).lower()
    if content_preview_chars is not None:
        params["content_preview_chars"] = content_preview_chars
    return params


# Shortest wait between two polls of a background job.
_MIN_POLL_INTERVAL = 0.1


def _path_segment(value: str) -> str:
    """``value`` percent-encoded for one URL path segment (``:`` kept)."""
    from urllib.parse import quote

    return quote(value, safe=":")


def _document_fields(
    memory_type: str | None,
    session_id: str | None,
    importance: float | None,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    ttl_seconds: int | None,
    expires_at: int | None,
    lang: str | None,
    chunking: Any,
    decompose: bool | None,
) -> dict[str, Any]:
    """The shared fields of the two document import requests.

    ``chunking`` may be a :class:`~dakera.models.ChunkingOptions` or a dict of
    its fields; any other key is a ``TypeError`` here (the server refuses it
    with ``422``).
    """
    from dakera.models import ChunkingOptions

    if isinstance(chunking, dict):
        chunking = ChunkingOptions(**chunking)
    return {
        "memory_type": memory_type,
        "session_id": session_id,
        "importance": importance,
        "tags": tags,
        "metadata": metadata,
        "ttl_seconds": ttl_seconds,
        "expires_at": expires_at,
        "lang": lang,
        "chunking": chunking,
        "decompose": decompose,
    }
