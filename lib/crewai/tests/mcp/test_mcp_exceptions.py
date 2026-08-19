import asyncio

import httpx
import pytest

from crewai.mcp.exceptions import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPHTTPError,
    error_for_status,
    error_type_for_status,
    find_http_status,
    find_transport_failure,
)


def _http_status_error(status_code: int, detail: str = "refused") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://mcp.example.com/mcp")
    response = httpx.Response(status_code, text=detail, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


def test_error_for_status_returns_authentication_error_for_401():
    error = error_for_status(401)

    assert isinstance(error, MCPAuthenticationError)
    assert error.status_code == 401
    assert "401 Unauthorized" in str(error)
    assert "authentication failure" in str(error)


def test_error_for_status_returns_authentication_error_for_403():
    error = error_for_status(403, detail="forbidden")

    assert isinstance(error, MCPAuthenticationError)
    assert error.status_code == 403
    assert "403 Forbidden" in str(error)
    assert "forbidden" in str(error)


def test_error_for_status_returns_http_error_for_non_auth_status():
    error = error_for_status(500, detail="internal error")

    assert isinstance(error, MCPHTTPError)
    assert error.status_code == 500
    assert "500 Internal Server Error" in str(error)


def test_error_type_for_status_maps_auth_and_http_errors():
    assert error_type_for_status(401) == "authentication"
    assert error_type_for_status(403) == "authentication"
    assert error_type_for_status(500) == "http_error"
    assert error_type_for_status(None) is None


def test_find_http_status_from_httpx_error():
    error = _http_status_error(401)

    assert find_http_status(error) == 401


def test_find_http_status_from_typed_connection_error():
    error = MCPConnectionError("failed", status_code=403)

    assert find_http_status(error) == 403


def test_find_http_status_from_cancelled_error_with_context():
    auth_error = _http_status_error(401)
    cancelled = asyncio.CancelledError()
    cancelled.__context__ = auth_error

    assert find_http_status(cancelled) == 401


def test_find_http_status_from_exception_group():
    auth_error = _http_status_error(401)
    group = ExceptionGroup("task group failed", [auth_error])

    assert find_http_status(group) == 401


def test_find_transport_failure_ignores_teardown_noise():
    cancelled = asyncio.CancelledError()
    teardown = RuntimeError("Attempted to exit cancel scope in a different task")

    assert find_transport_failure(cancelled, teardown) is None


def test_find_transport_failure_returns_underlying_http_error():
    auth_error = _http_status_error(401)
    cancelled = asyncio.CancelledError()
    cancelled.__context__ = auth_error

    assert find_transport_failure(cancelled) is auth_error
