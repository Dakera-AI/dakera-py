"""
Dakera SDK Exceptions

Custom exception hierarchy for Dakera operations.
"""

from typing import Any, Optional


class DakeraError(Exception):
    """Base exception for all Dakera errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
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
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(message, status_code, response_body)
        self.retry_after = retry_after


class ServerError(DakeraError):
    """Raised when the server returns a 5xx error."""

    pass


class AuthenticationError(DakeraError):
    """Raised when authentication fails."""

    pass


class TimeoutError(DakeraError):
    """Raised when a request times out."""

    pass
