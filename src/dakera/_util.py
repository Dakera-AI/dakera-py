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
