"""Tests for HTTP transport."""

import pytest

from crewai.mcp.transports.http import HTTPTransport, _create_httpx_client_factory


def test_http_transport_verify_default():
    """Test HTTPTransport has verify=True by default."""
    transport = HTTPTransport(url="http://localhost:9999/mcp")
    assert transport.verify is True


def test_http_transport_verify_false():
    """Test HTTPTransport with verify=False."""
    transport = HTTPTransport(
        url="http://localhost:9999/mcp",
        verify=False,
    )
    assert transport.verify is False


def test_http_transport_verify_ca_bundle():
    """Test HTTPTransport with custom CA bundle path."""
    transport = HTTPTransport(
        url="http://localhost:9999/mcp",
        verify="/path/to/ca-bundle.crt",
    )
    assert transport.verify == "/path/to/ca-bundle.crt"


def test_http_transport_streamable_default():
    """Test HTTPTransport has streamable=True by default."""
    transport = HTTPTransport(url="http://localhost:9999/mcp")
    assert transport.streamable is True


def test_create_httpx_client_factory_returns_async_client():
    """Test _create_httpx_client_factory returns an AsyncClient."""
    import httpx

    factory = _create_httpx_client_factory(verify=False)
    client = factory()
    assert isinstance(client, httpx.AsyncClient)


def test_create_httpx_client_factory_preserves_mcp_defaults():
    """Test _create_httpx_client_factory preserves MCP default settings."""
    factory = _create_httpx_client_factory(verify=False)
    client = factory(headers={"X-Test": "value"})
    assert client.follow_redirects is True
    assert client.timeout.connect == 30.0
    assert client.headers.get("X-Test") == "value"


def test_create_httpx_client_factory_with_custom_timeout():
    """Test _create_httpx_client_factory respects custom timeout."""
    import httpx

    factory = _create_httpx_client_factory(verify=False)
    custom_timeout = httpx.Timeout(60.0)
    client = factory(timeout=custom_timeout)
    assert client.timeout.connect == 60.0


def test_create_httpx_client_factory_with_auth():
    """Test _create_httpx_client_factory passes auth parameter."""
    import httpx

    factory = _create_httpx_client_factory(verify=False)
    auth = httpx.BasicAuth(username="user", password="pass")
    client = factory(auth=auth)
    assert client.auth is not None
