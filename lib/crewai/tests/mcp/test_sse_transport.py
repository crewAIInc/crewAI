"""Tests for SSE transport."""

import pytest

from crewai.mcp.transports.sse import SSETransport


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