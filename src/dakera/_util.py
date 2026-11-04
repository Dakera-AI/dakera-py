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


def _import_params(agent_id: str, fmt: str, run_async: bool) -> dict[str, Any]:
    """Query of ``POST /v1/import``; ``async=true`` (server v0.12.4+) only when asked."""
    params: dict[str, Any] = {"agent_id": agent_id, "format": fmt}
    if run_async:
        params["async"] = "true"
    return params


def _get_memory_params(agent_id: str | None, projection_days: int | None) -> dict[str, Any] | None:
    """Query of ``GET /v1/memory/get/{id}`` (``projection_days``: server v0.12.4+)."""
    params: dict[str, Any] = {}
    if agent_id is not None:
        params["agent_id"] = agent_id
    if projection_days is not None:
        if isinstance(projection_days, bool) or not 1 <= projection_days <= 365:
            raise ValueError("projection_days must be between 1 and 365")
        params["projection_days"] = projection_days
    return params or None


_FORGET_TEXT_MAX_CHARS = 8000


def _forget_body(
    agent_id: str,
    *,
    memory_ids: list[str] | None,
    memory_type: str | None,
    session_id: str | None,
    tags: list[str] | None,
    below_importance: float | None,
    about: str | None,
    request: str | None,
    until: int | None,
    min_score: float | None,
    limit: int | None,
    restorable: bool | None,
    dry_run: bool | None,
) -> dict[str, Any]:
    """Body of ``POST /v1/memory/forget``; only the fields given are sent.

    Checks what the server would refuse with ``400`` before sending: ``about``
    with ``request``, ``min_score`` outside ``[0, 1]``, ``limit`` outside
    ``1..50`` and an ``about`` / ``request`` above 8000 characters.
    """
    if about is not None and request is not None:
        raise ValueError("pass about or request, not both")
    for name, text in (("about", about), ("request", request)):
        if text is not None and len(text) > _FORGET_TEXT_MAX_CHARS:
            raise ValueError(f"{name} is at most {_FORGET_TEXT_MAX_CHARS} characters")
    if min_score is not None and not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    if limit is not None and (isinstance(limit, bool) or not 1 <= limit <= 50):
        raise ValueError("limit must be between 1 and 50")
    data: dict[str, Any] = {"agent_id": agent_id}
    optional: dict[str, Any] = {
        "memory_ids": list(memory_ids) if memory_ids is not None else None,
        "memory_type": memory_type,
        "session_id": session_id,
        "tags": list(tags) if tags is not None else None,
        "below_importance": below_importance,
        "about": about,
        "request": request,
        "until": until,
        "min_score": min_score,
        "limit": limit,
        "restorable": restorable,
        "dry_run": dry_run,
    }
    data.update({k: v for k, v in optional.items() if v is not None})
    return data


# Forget fields a server before v0.12.4 ignores, and the selectors every server applies.
_FORGET_V0124_FIELDS = ("about", "request", "until", "min_score", "limit", "restorable", "dry_run")
_FORGET_FILTER_FIELDS = ("memory_ids", "memory_type", "session_id", "tags", "below_importance")
_FORGET_V0124_MIN = (0, 12, 4)


def _forget_new_fields_with_filter(body: dict[str, Any]) -> list[str]:
    """The v0.12.4-only fields of a forget body that is also selected by
    ``memory_ids`` or a filter; ``[]`` when there are none (or no filter).

    A server before v0.12.4 ignores those fields and forgets every memory the
    filters select (a ``dry_run`` would delete), so such a body may only go to a
    v0.12.4+ server. Without a filter an older server refuses the body (``400``).
    """
    new = [k for k in _FORGET_V0124_FIELDS if k in body]
    if new and any(k in body for k in _FORGET_FILTER_FIELDS):
        return new
    return []


def _parse_server_version(version: str | None) -> tuple[int, int, int] | None:
    """``(major, minor, patch)`` of a server version such as ``0.12.4``; ``None``
    when it is missing or unreadable."""
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", (version or "").strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _server_too_old(
    version: str | None, minimum: tuple[int, int, int], what: str, why: str
) -> str | None:
    """``None`` when ``version`` is at least ``minimum``; otherwise the message of
    the error to raise. An unknown version counts as too old (the safe side)."""
    parsed = _parse_server_version(version)
    if parsed is not None and parsed >= minimum:
        return None
    shown = f"v{version}" if version else "unknown (no server version advertised)"
    need = ".".join(str(n) for n in minimum)
    return f"{what} needs Dakera server v{need}+ (this server: {shown}); {why}"


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


# Fields of a ``POST /v1/memory/store`` answer, next to ``memory``, that are
# copied onto the returned memory dict when the server sends them:
# ``session_state`` (v0.12.2), the truncation report and ``warnings`` (v0.12.3).
_STORE_RESULT_KEYS = (
    "session_state",
    "truncated",
    "content_tokens",
    "embedded_tokens",
    "max_tokens",
    "tokens_estimated",
    "windows",
    "warnings",
    "forget_request",
)


def _store_memory_result(response: Any) -> Any:
    """The memory of a ``POST /v1/memory/store`` answer.

    The server wraps it in ``{"memory": {...}, "embedding_time_ms": ...}``.
    The fields it adds next to ``memory`` are copied onto the returned dict
    when present (:data:`_STORE_RESULT_KEYS`): ``session_state`` (v0.12.2,
    ``"active"`` / ``"ended"``) when the memory went into a started session;
    from v0.12.3 the truncation report (``truncated``, ``content_tokens``,
    ``embedded_tokens``, ``max_tokens``, ``tokens_estimated``, ``windows``)
    when the text is longer than the embedding model reads, and ``warnings``
    when the server ignored request fields it does not know; from v0.12.4
    ``forget_request`` (``{probability, threshold, is_request, sentence?}``,
    see :class:`~dakera.models.ForgetIntent`) when the text reads as a request
    to forget something said.
    """
    if isinstance(response, dict) and isinstance(response.get("memory"), dict):
        memory = dict(response["memory"])
        for key in _STORE_RESULT_KEYS:
            if response.get(key) is not None:
                memory[key] = response[key]
        return memory
    return response


def _event_time(value: Any) -> int | float | str:
    """The wire form of a memory's ``event_time`` (server v0.12.3+).

    Unix seconds (``int`` / ``float``) and RFC 3339 strings are sent as given; a
    timezone-aware :class:`datetime.datetime` is sent as RFC 3339. A naive
    datetime is refused: the server needs an offset, and guessing one would
    move the event.
    """
    from datetime import datetime

    if isinstance(value, bool):
        raise TypeError("event_time must be Unix seconds, an RFC 3339 string or a datetime")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "event_time datetime must be timezone-aware "
                "(e.g. datetime(..., tzinfo=timezone.utc))"
            )
        return value.isoformat()
    if isinstance(value, (int, float, str)):
        return value
    raise TypeError("event_time must be Unix seconds, an RFC 3339 string or a datetime")


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
