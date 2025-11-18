import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.agent.core import Agent
from crewai.mcp import MCPServerSSE


@pytest.fixture(autouse=True)
def isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "storage"))


class FakeSSEClientError:
    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers

    async def __aenter__(self):
        raise Exception("SSE connection failed")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_mcp_sse_error():
    fake_mcp = ModuleType("mcp")
    fake_mcp_client = ModuleType("mcp.client")
    fake_mcp_client_sse = ModuleType("mcp.client.sse")

    sys.modules["mcp"] = fake_mcp
    sys.modules["mcp.client"] = fake_mcp_client
    sys.modules["mcp.client.sse"] = fake_mcp_client_sse

    mock_sse_client = MagicMock(side_effect=FakeSSEClientError)
    fake_mcp_client_sse.sse_client = mock_sse_client

    yield mock_sse_client

    del sys.modules["mcp.client.sse"]
    del sys.modules["mcp.client"]
    del sys.modules["mcp"]


def test_agent_get_native_mcp_tools_raises_runtime_error_not_unbound_local_error(
    mock_mcp_sse_error,
):
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
    )

    mcp_config = MCPServerSSE(
        url="https://example.com/sse",
        headers={"Authorization": "Bearer token"},
    )

    with pytest.raises(RuntimeError, match="Failed to get native MCP tools"):
        agent._get_native_mcp_tools(mcp_config)


def test_agent_get_native_mcp_tools_error_message_contains_original_error(
    mock_mcp_sse_error,
):
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
    )

    mcp_config = MCPServerSSE(
        url="https://example.com/sse",
    )

    with pytest.raises(RuntimeError) as exc_info:
        agent._get_native_mcp_tools(mcp_config)

    assert "Failed to get native MCP tools" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
