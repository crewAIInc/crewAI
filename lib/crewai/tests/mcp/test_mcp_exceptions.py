import asyncio
import sys

import httpx
import pytest

if sys.version_info >= (3, 11):
    from builtins import ExceptionGroup
else:
    from exceptiongroup import ExceptionGroup

from crewai.mcp.exceptions import (
    MCPAuthenticationError,
    MCPConnectionError,
    MCPHTTPError,
    error_for_status,
    error_type_for_status,
    find_http_status,
    find_transport_failure,
    raise_connection_failure,
    tool_execution_error_type,
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


def test_raise_connection_failure_reraises_typed_error():
    auth_error = MCPAuthenticationError(401)

    with pytest.raises(MCPAuthenticationError) as exc_info:
        raise_connection_failure("unused", auth_error)

    assert exc_info.value is auth_error


def test_raise_connection_failure_builds_auth_error_from_http_status():
    with pytest.raises(MCPAuthenticationError) as exc_info:
        raise_connection_failure("unused", _http_status_error(401))

    assert exc_info.value.status_code == 401


def test_raise_connection_failure_prefers_teardown_http_status():
    cancelled = asyncio.CancelledError()
    auth_error = _http_status_error(401)

    with pytest.raises(MCPAuthenticationError) as exc_info:
        raise_connection_failure("unused", cancelled, auth_error)

    assert exc_info.value.status_code == 401


def test_raise_connection_failure_falls_back_to_connection_error():
    with pytest.raises(ConnectionError, match="host unreachable"):
        raise_connection_failure("host unreachable", ConnectionError("refused"))


def test_tool_execution_error_type_maps_authentication_failures():
    assert (
        tool_execution_error_type(MCPAuthenticationError(401)) == "authentication"
    )
    assert tool_execution_error_type(_http_status_error(403)) == "authentication"


def test_tool_execution_error_type_maps_timeout_and_validation():
    assert tool_execution_error_type(asyncio.TimeoutError()) == "timeout"
    assert (
        tool_execution_error_type(ConnectionError("Operation timed out after 30 seconds"))
        == "timeout"
    )
    assert tool_execution_error_type(ValueError("Resource not found")) == "validation"


def test_tool_execution_error_type_maps_http_and_server_errors():
    assert tool_execution_error_type(MCPHTTPError(500)) == "http_error"
    assert tool_execution_error_type(ConnectionError("unexpected failure")) == "server_error"
