import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.tools.base_discovery_provider import DiscoveryEntry, BaseDiscoveryProvider
from crewai.tools.dynamic_discovery_tool import DynamicDiscoveryTool
from crewai.tools.mcp_discovery_provider import MCPDiscoveryProvider
from crewai.tools.base_tool import BaseTool


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_mcp_discovery_search_semantic_parses_json(mock_async_client_class):
    """MCPDiscoveryProvider.search_semantic should parse API JSON into DiscoveryEntry list."""
    # Arrange: mock HTTP response from mcp-discovery API
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "recommendations": [
            {
                "server": "test-server",
                "npm_package": "test-package",
                "install_command": "npm install test-package",
                "confidence": 0.95,
                "description": "Test MCP server",
                "capabilities": ["db", "auth"],
                "metrics": {
                    "avg_latency_ms": 10,
                    "uptime_pct": 99.9,
                    "last_checked": "2025-01-01T00:00:00Z",
                },
                "docs_url": "https://docs.example.com/test-server",
                "github_url": "https://github.com/example/test-server",
            },
            {
                # Missing server name should be skipped
                "server": "",
                "npm_package": "ignored",
                "install_command": "npm install ignored",
                "confidence": 0.1,
                "description": "Should be ignored",
                "capabilities": [],
                "metrics": {},
                "docs_url": None,
                "github_url": None,
            },
        ]
    }
    mock_response.raise_for_status.return_value = None

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response
    mock_async_client_class.return_value.__aenter__.return_value = mock_client_instance

    provider = MCPDiscoveryProvider(base_url="https://mock-api.example.com")

    # Act
    entries = await provider.search_semantic("database with auth", limit=3)

    # Assert
    mock_client_instance.post.assert_called_once_with(
        "https://mock-api.example.com/api/v1/discover",
        json={"need": "database with auth", "limit": 3},
    )

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, DiscoveryEntry)
    assert entry.id == "test-server"
    assert entry.name == "test-server"
    assert entry.description == "Test MCP server"
    # docs_url should be preferred as source_uri when present
    assert entry.source_uri == "https://docs.example.com/test-server"
    assert entry.provider_id == provider.provider_id
    assert entry.raw_metadata["npm_package"] == "test-package"
    assert entry.raw_metadata["install_command"] == "npm install test-package"
    assert entry.raw_metadata["capabilities"] == ["db", "auth"]
    assert entry.raw_metadata["metrics"]["avg_latency_ms"] == 10


class _DummyTool(BaseTool):
    """Simple dummy tool for DynamicDiscoveryTool tests."""

    def _run(self, *args, **kwargs) -> str:  # pragma: no cover - not used directly
        return "ok"


def test_dynamic_discovery_tool_calls_provider_and_returns_summary():
    """DynamicDiscoveryTool should call provider and return tool descriptions."""

    # Arrange: mock provider with async search_semantic_and_resolve
    provider_mock = MagicMock(spec=BaseDiscoveryProvider)

    tool1 = _DummyTool(name="tool_one", description="First tool")
    tool2 = _DummyTool(name="tool_two", description="Second tool")

    provider_mock.search_semantic_and_resolve = AsyncMock(
        return_value=[tool1, tool2]
    )

    # No agent – just return tools and summary
    discovery_tool = DynamicDiscoveryTool(
        provider=provider_mock,
        limit=5,
        agent=None,
        auto_register=False,
    )

    # Act: call the synchronous _run which internally uses asyncio.run(...)
    result = discovery_tool._run(search_query="find useful tools")

    # Assert: provider was called with correct arguments
    provider_mock.search_semantic_and_resolve.assert_awaited_once_with(
        "find useful tools", limit=5
    )

    assert result["count"] == 2
    assert result["registered"] is False
    summary = result["summary"]
    assert "Discovered 2 tool(s)" in summary
    assert "tool_one" in summary
    assert "tool_two" in summary

    # The tools list should be the actual tool instances returned by the provider
    tools = result["tools"]
    assert tools == [tool1, tool2]

