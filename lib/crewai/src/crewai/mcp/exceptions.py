"""Exceptions and error classification for MCP client connections.

When an MCP server refuses a connection with an HTTP status, the status is
observed by the HTTP client but the exception carrying it is raised inside the
transport's anyio task group. The awaiting coroutine therefore sees only a
bare ``CancelledError``, and the real cause surfaces separately as an
exception group once the transport unwinds. ``find_http_status`` recovers the
status from either shape so callers can report it instead of guessing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from http import HTTPStatus
import sys
from typing import NoReturn


if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup


AUTH_STATUS_CODES = frozenset({401, 403})

# Substrings identifying a RuntimeError that describes the unwinding itself.
# anyio and async generators report these while a task group is torn down, so
# they say nothing about why the connection failed.
_TEARDOWN_MARKERS = (
    "cancel scope",
    "async generator",
    "athrow",
    "asend",
    "different task",
    "event loop is closed",
)


def _status_phrase(status_code: int) -> str:
    """Render a status code with its reason phrase, e.g. ``401 Unauthorized``."""
    try:
        return f"{status_code} {HTTPStatus(status_code).phrase}"
    except ValueError:
        return str(status_code)


class MCPConnectionError(ConnectionError):
    """Raised when a connection to an MCP server cannot be established.

    Subclasses ``ConnectionError`` so that existing callers handling
    connection failures continue to work unchanged.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Create an MCPConnectionError.

        Args:
            message: Description of the failure.
            status_code: HTTP status the server responded with, when known.
        """
        super().__init__(message)
        self.status_code = status_code


class MCPAuthenticationError(MCPConnectionError):
    """Raised when an MCP server rejects the request's credentials."""

    def __init__(self, status_code: int, detail: str | None = None) -> None:
        """Create an MCPAuthenticationError.

        Args:
            status_code: HTTP status the server responded with (401 or 403).
            detail: Optional underlying error text to append.
        """
        message = (
            f"MCP server refused the connection with HTTP {_status_phrase(status_code)}. "
            "This is an authentication failure - check the credentials sent in the "
            "server's headers."
        )
        if detail:
            message = f"{message} Server response: {detail}"
        super().__init__(message, status_code=status_code)


class MCPHTTPError(MCPConnectionError):
    """Raised when an MCP server refuses the connection with a non-auth status."""

    def __init__(self, status_code: int, detail: str | None = None) -> None:
        """Create an MCPHTTPError.

        Args:
            status_code: HTTP status the server responded with.
            detail: Optional underlying error text to append.
        """
        message = f"MCP server refused the connection with HTTP {_status_phrase(status_code)}."
        if detail:
            message = f"{message} Server response: {detail}"
        super().__init__(message, status_code=status_code)


def error_for_status(status_code: int, detail: str | None = None) -> MCPConnectionError:
    """Build the exception matching an HTTP status.

    Args:
        status_code: HTTP status the server responded with.
        detail: Optional underlying error text to append.

    Returns:
        An ``MCPAuthenticationError`` for 401/403, otherwise an ``MCPHTTPError``.
    """
    if status_code in AUTH_STATUS_CODES:
        return MCPAuthenticationError(status_code, detail)
    return MCPHTTPError(status_code, detail)


def error_type_for_status(status_code: int | None) -> str | None:
    """Map an HTTP status to the ``error_type`` reported on MCP events.

    Args:
        status_code: HTTP status the server responded with, when known.

    Returns:
        ``"authentication"`` for 401/403, ``"http_error"`` for any other
        status, or ``None`` when no status was observed.
    """
    if status_code is None:
        return None
    return "authentication" if status_code in AUTH_STATUS_CODES else "http_error"


def tool_execution_error_type(exc: BaseException) -> str:
    """Map an exception to ``MCPToolExecutionFailedEvent.error_type``.

    Args:
        exc: The exception raised while executing an MCP tool.

    Returns:
        One of ``"timeout"``, ``"authentication"``, ``"validation"``,
        ``"http_error"``, or ``"server_error"``.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError) and (
        "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
    ):
        return "timeout"
    if isinstance(exc, ValueError):
        return "validation"
    if isinstance(exc, MCPConnectionError) and exc.status_code is not None:
        return error_type_for_status(exc.status_code) or "server_error"
    status_code = find_http_status(exc)
    if status_code is not None:
        return error_type_for_status(status_code) or "server_error"
    error_str = str(exc).lower()
    if "authentication" in error_str or "unauthorized" in error_str:
        return "authentication"
    return "server_error"


def raise_connection_failure(message: str, *errors: BaseException | None) -> NoReturn:
    """Raise a typed MCP connection error when an HTTP status is known.

    Args:
        message: Fallback message when no HTTP status can be recovered.
        *errors: Exceptions observed while connecting or unwinding.

    Raises:
        MCPConnectionError: When an HTTP status was observed.
        ConnectionError: When no HTTP status was observed.
    """
    present = [error for error in errors if error is not None]
    if not present:
        raise ConnectionError(message)

    for error in present:
        if isinstance(error, MCPConnectionError):
            raise error

    primary = present[0]
    status_code = find_http_status(*present)
    if status_code is not None:
        raise error_for_status(status_code, detail=str(primary)) from primary
    if isinstance(primary, asyncio.CancelledError):
        failure = find_transport_failure(*present)
        if failure is not None:
            raise failure from primary
    raise ConnectionError(message) from primary


def _leaves(exc: BaseException | None, seen: set[int]) -> Iterator[BaseException]:
    """Yield the individual exceptions reachable from *exc*.

    Exception groups are containers rather than causes, so their members are
    yielded in place of the group itself. ``__cause__`` and ``__context__`` are
    followed too, because the transport re-wraps failures while unwinding.

    Args:
        exc: Exception to traverse, or None.
        seen: Ids already visited, guarding against reference cycles.

    Yields:
        Each reachable exception that is not itself a group.
    """
    if exc is None or id(exc) in seen:
        return
    seen.add(id(exc))

    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            yield from _leaves(nested, seen)
    else:
        yield exc

    for linked in (exc.__cause__, exc.__context__):
        yield from _leaves(linked, seen)


def _is_teardown_noise(exc: BaseException) -> bool:
    """Report whether *exc* describes unwinding rather than a failure."""
    if isinstance(exc, asyncio.CancelledError | GeneratorExit):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return any(marker in message for marker in _TEARDOWN_MARKERS)
    return False


def find_http_status(*errors: BaseException | None) -> int | None:
    """Search exceptions for an HTTP status reported by an MCP server.

    Args:
        *errors: Exceptions to search, in order of preference.

    Returns:
        The first HTTP status found, or ``None`` if none of the exceptions
        were caused by an HTTP status.
    """
    import httpx

    seen: set[int] = set()
    for error in errors:
        for exc in _leaves(error, seen):
            if isinstance(exc, MCPConnectionError) and exc.status_code is not None:
                return exc.status_code
            if isinstance(exc, httpx.HTTPStatusError):
                return exc.response.status_code
    return None


def find_transport_failure(*errors: BaseException | None) -> BaseException | None:
    """Search exceptions for the one that explains why a connection failed.

    A failing transport cancels the coroutine that was waiting on it, so the
    cancellation reaching the caller is a symptom. The explanation is whichever
    reachable exception is not itself cancellation or teardown bookkeeping.

    Args:
        *errors: Exceptions to search, in order of preference.

    Returns:
        The first explanatory exception, or ``None`` when the exceptions
        describe only cancellation and teardown.
    """
    seen: set[int] = set()
    for error in errors:
        for exc in _leaves(error, seen):
            if not _is_teardown_noise(exc):
                return exc
    return None
