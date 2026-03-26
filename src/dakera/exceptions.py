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
    # 503 Service Unavailable
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    # 401 Unauthorized
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    INVALID_API_KEY = "INVALID_API_KEY"
    API_KEY_EXPIRED = "API_KEY_EXPIRED"
    # 403 Forbidden
    INSUFFICIENT_SCOPE = "INSUFFICIENT_SCOPE"
    NAMESPACE_ACCESS_DENIED = "NAMESPACE_ACCESS_DENIED"
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
