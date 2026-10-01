"""
Dakera SDK Exceptions

Custom exception hierarchy for Dakera operations.
"""

import enum
from typing import Any


class ErrorCode(enum.Enum):
    """Server error codes returned in structured error responses."""

    # 404 Not Found
    NAMESPACE_NOT_FOUND = "NAMESPACE_NOT_FOUND"
    VECTOR_NOT_FOUND = "VECTOR_NOT_FOUND"
    # 400 Bad Request
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    EMPTY_VECTOR = "EMPTY_VECTOR"
    INVALID_REQUEST = "INVALID_REQUEST"
    # 500 Internal Server Error
    STORAGE_ERROR = "STORAGE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # 413 Content Too Large
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"  # v0.12: the request is over a limit
    # 503 Service Unavailable
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    # 401 Unauthorized
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    INVALID_API_KEY = "INVALID_API_KEY"
    API_KEY_EXPIRED = "API_KEY_EXPIRED"
    # 403 Forbidden
    INSUFFICIENT_SCOPE = "INSUFFICIENT_SCOPE"
    NAMESPACE_ACCESS_DENIED = "NAMESPACE_ACCESS_DENIED"
    # v0.12 additions (additive; v0.11 servers never send them)
    CONFLICT = "CONFLICT"  # 409
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"  # 501: the configured backend cannot do it
    FEATURE_DISABLED = "FEATURE_DISABLED"  # 501: the feature flag is off
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"  # 429
    QUERY_TIMEOUT = "QUERY_TIMEOUT"  # 504
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"  # 408
    ROUTE_NOT_FOUND = "ROUTE_NOT_FOUND"  # 404: no such path
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"  # 405
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"  # 415
    API_KEY_NOT_FOUND = "API_KEY_NOT_FOUND"  # 404
    JOB_NOT_FOUND = "JOB_NOT_FOUND"  # 404
    CROSS_ORIGIN_REQUEST_REFUSED = "CROSS_ORIGIN_REQUEST_REFUSED"  # 403
    # Fallback for unrecognised codes
    UNKNOWN = "UNKNOWN"


class DakeraError(Exception):
    """Base exception for all Dakera errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: Any | None = None,
        code: ErrorCode | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.code = code

    @property
    def details(self) -> str | None:
        """The error body's ``details`` string, when the server sent one."""
        if isinstance(self.response_body, dict):
            value = self.response_body.get("details")
            return value if isinstance(value, str) else None
        return None

    @property
    def resource(self) -> str | None:
        """On a 404, what was not found (``namespace``, ``vector``, ``memory``,
        ``job``, ``attachment``, ...) — server v0.12+; ``None`` on older servers."""
        if isinstance(self.response_body, dict):
            value = self.response_body.get("resource")
            return value if isinstance(value, str) else None
        return None

    def __str__(self) -> str:
        if self.status_code and self.code:
            return f"[{self.status_code}] {self.code.value}: {self.message}"
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class ConnectionError(DakeraError):
    """Raised when unable to connect to Dakera server."""

    pass


class NotFoundError(DakeraError):
    """Raised when a requested resource (namespace, vector) is not found."""

    pass


class ValidationError(DakeraError):
    """Raised when request validation fails."""

    pass


class UnsupportedCapabilityError(ValidationError):
    """Raised *before* a request is sent when the server's advertised
    capabilities (``GET /v1/capabilities``) do not include what was asked for.

    ``kind`` names the registry (``"model"``, ``"index_kind"``,
    ``"distance_metric"``, ``"search_mode"``, ``"query_language"``),
    ``requested`` is the wire string that was rejected and ``supported`` is what
    the server does accept, so the message is actionable on its own.
    """

    def __init__(
        self,
        kind: str,
        requested: str,
        supported: list[str],
        server_version: str | None = None,
    ) -> None:
        self.kind = kind
        self.requested = requested
        self.supported = list(supported)
        self.server_version = server_version
        server = f"Dakera server v{server_version}" if server_version else "this Dakera server"
        accepted = ", ".join(self.supported) if self.supported else "(none advertised)"
        super().__init__(
            f"{kind} '{requested}' is not supported by {server}; "
            f"supported {kind} values: {accepted}",
            code=ErrorCode.INVALID_REQUEST,
        )


class ConflictError(DakeraError):
    """Raised on 409: the request conflicts with the server's state (for example
    deleting an attachment a memory still references — forget the memory instead)."""

    pass


class PayloadTooLargeError(DakeraError):
    """Raised on 413.

    Two causes share the status: a namespace **quota** (``code ==
    ErrorCode.QUOTA_EXCEEDED``, :attr:`is_quota`) and a request **over a size
    limit** such as ``DAKERA_ATTACHMENT_MAX_BYTES`` or a record's
    ``DAKERA_RECORD_MAX_BYTES`` (``code == ErrorCode.PAYLOAD_TOO_LARGE``,
    :attr:`is_oversize`). Neither is retried: resending the same request fails
    the same way.
    """

    @property
    def is_quota(self) -> bool:
        return self.code == ErrorCode.QUOTA_EXCEEDED

    @property
    def is_oversize(self) -> bool:
        return self.code == ErrorCode.PAYLOAD_TOO_LARGE


class FeatureNotAvailableError(DakeraError):
    """Raised on 501: the server cannot do this with its current configuration.

    ``code == FEATURE_DISABLED`` means the route exists but its opt-in switch is
    off (attachments, records, vision — ``details`` names the ``DAKERA_*``
    variable that enables it); ``NOT_IMPLEMENTED`` means the configured backend
    cannot perform it. A configuration error is never transient, so it is not
    retried.
    """

    @property
    def is_feature_disabled(self) -> bool:
        return self.code == ErrorCode.FEATURE_DISABLED


class RateLimitError(DakeraError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: Any | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message, status_code, response_body)
        self.retry_after = retry_after


class ServerError(DakeraError):
    """Raised when the server returns a 5xx error."""

    pass


class ServiceUnavailableError(ServerError):
    """Raised on 503. Every v0.12 ``503`` carries ``Retry-After`` (whole seconds),
    exposed as :attr:`retry_after`; the retry logic waits that long. A
    :class:`ServerError` subclass, so existing ``except ServerError`` code keeps
    working."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: Any | None = None,
        code: ErrorCode | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code, response_body, code)
        self.retry_after = retry_after


class AuthenticationError(DakeraError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(DakeraError):
    """Raised when the server returns a 403 Forbidden response.

    Covers INSUFFICIENT_SCOPE and NAMESPACE_ACCESS_DENIED error codes.
    """

    pass


class TimeoutError(DakeraError):
    """Raised when a request times out."""

    pass


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header: whole seconds (what Dakera sends), or an
    HTTP date. Returns seconds to wait, or ``None`` when absent / unparseable."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        import email.utils
        import time

        try:
            when = email.utils.parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError):
            return None
        return max(0.0, when - time.time())
    return max(0.0, seconds)


def error_from_response(
    status_code: int, body: Any, retry_after_header: str | None = None
) -> DakeraError:
    """Map an error response (status + parsed body) onto the exception hierarchy.

    Every v0.12 error body is JSON ``{"error", "code", "details"?, "resource"?}``;
    a v0.11 server or a proxy may answer with plain text, which is carried as the
    message.
    """
    raw_code = body.get("code") if isinstance(body, dict) else None
    try:
        code = ErrorCode(raw_code) if raw_code is not None else ErrorCode.UNKNOWN
    except ValueError:
        code = ErrorCode.UNKNOWN

    def message(default: str) -> str:
        if isinstance(body, dict):
            return str(body.get("error", default))
        return str(body) if body else default

    kw: dict[str, Any] = {"status_code": status_code, "response_body": body, "code": code}
    retry_after = parse_retry_after(retry_after_header)
    if status_code == 400:
        return ValidationError(message("Validation error"), **kw)
    if status_code == 401:
        return AuthenticationError(message("Authentication failed"), **kw)
    if status_code == 403:
        return AuthorizationError(message("Forbidden"), **kw)
    if status_code == 404:
        return NotFoundError(message("Resource not found"), **kw)
    if status_code == 409:
        return ConflictError(message("Conflict"), **kw)
    if status_code == 413:
        return PayloadTooLargeError(message("Payload too large"), **kw)
    if status_code == 429:
        return RateLimitError(
            "Rate limit exceeded",
            status_code=status_code,
            response_body=body,
            retry_after=int(retry_after) if retry_after is not None else None,
        )
    if status_code == 501:
        return FeatureNotAvailableError(message("Not implemented"), **kw)
    if status_code == 503:
        return ServiceUnavailableError(
            message("Service unavailable"), retry_after=retry_after, **kw
        )
    if status_code >= 500:
        return ServerError(message("Server error"), **kw)
    return DakeraError(f"Unexpected status code: {status_code}", **kw)
