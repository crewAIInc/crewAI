"""Tests for the _amp_slug attribute set by MCPToolResolver._resolve_amp.

The slug is private metadata used downstream by enterprise tooling to recover
the canonical tool_id (e.g. ``crewai_oauth:<slug>|mcp``) for ACP rule
evaluation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.agent.core import Agent
from crewai.mcp.config import MCPServerHTTP
from crewai.mcp.tool_resolver import MCPToolResolver
from crewai.tools.mcp_native_tool import MCPNativeTool
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


@pytest.fixture
def agent():
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
    )


@pytest.fixture
def resolver(agent):
    return MCPToolResolver(agent=agent, logger=agent._logger)


class TestAmpSlugDefaultsNone:
    def test_native_tool_default_amp_slug_is_none(self):
        tool = MCPNativeTool(
            client_factory=lambda: None,
            tool_name="search",
            tool_schema={"description": "Search"},
            server_name="notion",
        )
        assert tool._amp_slug is None

    def test_wrapper_tool_default_amp_slug_is_none(self):
        tool = MCPToolWrapper(
            mcp_server_params={"url": "https://mcp.example.com"},
            tool_name="search",
            tool_schema={"description": "Search"},
            server_name="notion",
        )
        assert tool._amp_slug is None


class TestAmpSlugSetByResolveAmp:
    @patch("crewai.mcp.tool_resolver.MCPToolResolver._resolve_native")
    @patch("crewai.mcp.tool_resolver.MCPToolResolver._fetch_amp_mcp_configs")
    def test_resolve_amp_tags_each_tool_with_its_slug(
        self, mock_fetch_configs, mock_resolve_native, resolver
    ):
        mock_fetch_configs.return_value = {
            "notion": {"url": "https://mcp.crewai.com/notion"},
            "github": {"url": "https://mcp.crewai.com/github"},
        }

        notion_tool = MCPNativeTool(
            client_factory=lambda: None,
            tool_name="search",
            tool_schema={"description": "search Notion"},
            server_name="notion",
        )
        github_tool = MCPNativeTool(
            client_factory=lambda: None,
            tool_name="list_repos",
            tool_schema={"description": "list github repos"},
            server_name="github",
        )

        def fake_resolve_native(config):
            url = config.url if hasattr(config, "url") else config["url"]
            if "notion" in url:
                return ([notion_tool], [MagicMock()])
            return ([github_tool], [MagicMock()])

        mock_resolve_native.side_effect = fake_resolve_native

        tools, _ = resolver._resolve_amp(
            [("notion", None), ("github", None)]
        )

        assert {tool._amp_slug for tool in tools} == {"notion", "github"}

    @patch("crewai.mcp.tool_resolver.MCPToolResolver._fetch_amp_mcp_configs")
    def test_resolve_amp_does_not_tag_when_config_missing(
        self, mock_fetch_configs, resolver
    ):
        mock_fetch_configs.return_value = {}

        tools, _ = resolver._resolve_amp([("unknown", None)])

        assert tools == []


class TestAmpSlugUntaggedForOtherPaths:
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_resolve_external_does_not_set_amp_slug(self, mock_client_class, resolver):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "search", "description": "Search"}]
        )
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        with patch.object(
            resolver, "_get_mcp_tool_schemas", return_value={"search": {"description": "Search"}}
        ):
            tools = resolver._resolve_external("https://mcp.example.com/api")

        assert len(tools) == 1
        assert tools[0]._amp_slug is None

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_resolve_native_does_not_set_amp_slug(self, mock_client_class, resolver):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "search", "description": "Search"}]
        )
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        config = MCPServerHTTP(url="https://mcp.example.com/api")
        tools, _ = resolver._resolve_native(config)

        assert all(tool._amp_slug is None for tool in tools)
