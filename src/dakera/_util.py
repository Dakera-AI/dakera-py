"""Internal helpers shared by the sync and async clients."""

import mimetypes
import os
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from dakera.models import AuditExportResponse


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
