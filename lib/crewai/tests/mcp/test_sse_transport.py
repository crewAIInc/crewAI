"""Tests for SSE transport."""

import pytest

from crewai.mcp.transports.sse import SSETransport, _create_httpx_client_factory


@pytest.mark.asyncio
async def test_sse_transport_connect_does_not_pass_invalid_args():
    """Test that SSETransport.connect() doesn't pass invalid args to sse_client.

    The sse_client function does not accept terminate_on_close parameter.
    """
    transport = SSETransport(
        url="http://localhost:9999/sse",
        headers={"Authorization": "Bearer test"},
    )

    with pytest.raises(ConnectionError) as exc_info:
        await transport.connect()

    assert "unexpected keyword argument" not in str(exc_info.value)


def test_sse_transport_verify_default():
    """Test SSETransport has verify=True by default."""
    transport = SSETransport(url="http://localhost:9999/sse")
    assert transport.verify is True


def test_sse_transport_verify_false():
    """Test SSETransport with verify=False."""
    transport = SSETransport(
        url="http://localhost:9999/sse",
        verify=False,
    )
    assert transport.verify is False


def test_sse_transport_verify_ca_bundle():
    """Test SSETransport with custom CA bundle path."""
    transport = SSETransport(
        url="http://localhost:9999/sse",
        verify="/path/to/ca-bundle.crt",
    )
    assert transport.verify == "/path/to/ca-bundle.crt"


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
