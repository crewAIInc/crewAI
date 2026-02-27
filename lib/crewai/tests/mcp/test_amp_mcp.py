"""Tests for AMP MCP config fetching and tool resolution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.agent.core import Agent
from crewai.mcp.config import MCPServerHTTP, MCPServerSSE
from crewai.mcp.tool_resolver import MCPToolResolver
from crewai.tools.base_tool import BaseTool


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


@pytest.fixture
def mock_tool_definitions():
    return [
        {
            "name": "search",
            "description": "Search tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
        {
            "name": "create_page",
            "description": "Create a page",
            "inputSchema": {},
        },
    ]


class TestBuildMCPConfigFromDict:
    def test_builds_http_config(self):
        config_dict = {
            "type": "http",
            "url": "https://mcp.example.com/api",
            "headers": {"Authorization": "Bearer token123"},
            "streamable": True,
            "cache_tools_list": False,
        }

        result = MCPToolResolver._build_mcp_config_from_dict(config_dict)

        assert isinstance(result, MCPServerHTTP)
        assert result.url == "https://mcp.example.com/api"
        assert result.headers == {"Authorization": "Bearer token123"}
        assert result.streamable is True
        assert result.cache_tools_list is False

    def test_builds_sse_config(self):
        config_dict = {
            "type": "sse",
            "url": "https://mcp.example.com/sse",
            "headers": {"Authorization": "Bearer token123"},
            "cache_tools_list": True,
        }

        result = MCPToolResolver._build_mcp_config_from_dict(config_dict)

        assert isinstance(result, MCPServerSSE)
        assert result.url == "https://mcp.example.com/sse"
        assert result.headers == {"Authorization": "Bearer token123"}
        assert result.cache_tools_list is True

    def test_defaults_to_http(self):
        config_dict = {
            "url": "https://mcp.example.com/api",
        }

        result = MCPToolResolver._build_mcp_config_from_dict(config_dict)

        assert isinstance(result, MCPServerHTTP)
        assert result.streamable is True

    def test_http_defaults(self):
        config_dict = {
            "type": "http",
            "url": "https://mcp.example.com/api",
        }

        result = MCPToolResolver._build_mcp_config_from_dict(config_dict)

        assert result.headers is None
        assert result.streamable is True
        assert result.cache_tools_list is False


class TestFetchAmpMCPConfigs:
    @patch("crewai.cli.plus_api.PlusAPI")
    @patch("crewai_tools.tools.crewai_platform_tools.misc.get_platform_integration_token", return_value="test-api-key")
    def test_fetches_configs_successfully(self, mock_get_token, mock_plus_api_class, resolver):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "configs": {
                "notion": {
                    "type": "sse",
                    "url": "https://mcp.notion.so/sse",
                    "headers": {"Authorization": "Bearer notion-token"},
                },
                "github": {
                    "type": "http",
                    "url": "https://mcp.github.com/api",
                    "headers": {"Authorization": "Bearer gh-token"},
                },
            },
        }
        mock_plus_api = MagicMock()
        mock_plus_api.get_mcp_configs.return_value = mock_response
        mock_plus_api_class.return_value = mock_plus_api

        result = resolver._fetch_amp_mcp_configs(["notion", "github"])

        assert "notion" in result
        assert "github" in result
        assert result["notion"]["url"] == "https://mcp.notion.so/sse"
        mock_plus_api_class.assert_called_once_with(api_key="test-api-key")
        mock_plus_api.get_mcp_configs.assert_called_once_with(["notion", "github"])

    @patch("crewai.cli.plus_api.PlusAPI")
    @patch("crewai_tools.tools.crewai_platform_tools.misc.get_platform_integration_token", return_value="test-api-key")
    def test_omits_missing_slugs(self, mock_get_token, mock_plus_api_class, resolver):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "configs": {"notion": {"type": "sse", "url": "https://mcp.notion.so/sse"}},
        }
        mock_plus_api = MagicMock()
        mock_plus_api.get_mcp_configs.return_value = mock_response
        mock_plus_api_class.return_value = mock_plus_api

        result = resolver._fetch_amp_mcp_configs(["notion", "missing-server"])

        assert "notion" in result
        assert "missing-server" not in result

    @patch("crewai.cli.plus_api.PlusAPI")
    @patch("crewai_tools.tools.crewai_platform_tools.misc.get_platform_integration_token", return_value="test-api-key")
    def test_returns_empty_on_http_error(self, mock_get_token, mock_plus_api_class, resolver):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_plus_api = MagicMock()
        mock_plus_api.get_mcp_configs.return_value = mock_response
        mock_plus_api_class.return_value = mock_plus_api

        result = resolver._fetch_amp_mcp_configs(["notion"])

        assert result == {}

    @patch("crewai.cli.plus_api.PlusAPI")
    @patch("crewai_tools.tools.crewai_platform_tools.misc.get_platform_integration_token", return_value="test-api-key")
    def test_returns_empty_on_network_error(self, mock_get_token, mock_plus_api_class, resolver):
        import httpx

        mock_plus_api = MagicMock()
        mock_plus_api.get_mcp_configs.side_effect = httpx.ConnectError("Connection refused")
        mock_plus_api_class.return_value = mock_plus_api

        result = resolver._fetch_amp_mcp_configs(["notion"])

        assert result == {}

    @patch("crewai_tools.tools.crewai_platform_tools.misc.get_platform_integration_token", side_effect=Exception("No token"))
    def test_returns_empty_when_no_token(self, mock_get_token, resolver):
        result = resolver._fetch_amp_mcp_configs(["notion"])

        assert result == {}


class TestParseAmpRef:
    def test_bare_slug(self):
        slug, tool = MCPToolResolver._parse_amp_ref("notion")
        assert slug == "notion"
        assert tool is None

    def test_bare_slug_with_tool(self):
        slug, tool = MCPToolResolver._parse_amp_ref("notion#search")
        assert slug == "notion"
        assert tool == "search"

    def test_bare_slug_with_empty_tool(self):
        slug, tool = MCPToolResolver._parse_amp_ref("notion#")
        assert slug == "notion"
        assert tool is None

    def test_legacy_prefix_slug(self):
        slug, tool = MCPToolResolver._parse_amp_ref("crewai-amp:notion")
        assert slug == "notion"
        assert tool is None

    def test_legacy_prefix_with_tool(self):
        slug, tool = MCPToolResolver._parse_amp_ref("crewai-amp:notion#search")
        assert slug == "notion"
        assert tool == "search"


class TestGetMCPToolsAmpIntegration:
    @patch("crewai.mcp.tool_resolver.MCPClient")
    @patch.object(MCPToolResolver, "_fetch_amp_mcp_configs")
    def test_single_request_for_multiple_amp_refs(
        self, mock_fetch, mock_client_class, agent, mock_tool_definitions
    ):
        mock_fetch.return_value = {
            "notion": {
                "type": "sse",
                "url": "https://mcp.notion.so/sse",
                "headers": {"Authorization": "Bearer token"},
            },
            "github": {
                "type": "http",
                "url": "https://mcp.github.com/api",
                "headers": {"Authorization": "Bearer gh-token"},
                "streamable": True,
            },
        }

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        tools = agent.get_mcp_tools(["notion", "github"])

        mock_fetch.assert_called_once_with(["notion", "github"])
        assert len(tools) == 4  # 2 tools per server

    @patch("crewai.mcp.tool_resolver.MCPClient")
    @patch.object(MCPToolResolver, "_fetch_amp_mcp_configs")
    def test_tool_filter_with_hash_syntax(
        self, mock_fetch, mock_client_class, agent, mock_tool_definitions
    ):
        mock_fetch.return_value = {
            "notion": {
                "type": "sse",
                "url": "https://mcp.notion.so/sse",
                "headers": {"Authorization": "Bearer token"},
            },
        }

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        tools = agent.get_mcp_tools(["notion#search"])

        mock_fetch.assert_called_once_with(["notion"])
        assert len(tools) == 1
        assert tools[0].name == "mcp_notion_so_sse_search"

    @patch("crewai.mcp.tool_resolver.MCPClient")
    @patch.object(MCPToolResolver, "_fetch_amp_mcp_configs")
    def test_deduplicates_slugs(
        self, mock_fetch, mock_client_class, agent, mock_tool_definitions
    ):
        mock_fetch.return_value = {
            "notion": {
                "type": "sse",
                "url": "https://mcp.notion.so/sse",
                "headers": {"Authorization": "Bearer token"},
            },
        }

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        tools = agent.get_mcp_tools(["notion#search", "notion#create_page"])

        mock_fetch.assert_called_once_with(["notion"])
        assert len(tools) == 2

    @patch.object(MCPToolResolver, "_fetch_amp_mcp_configs")
    def test_skips_missing_configs_gracefully(self, mock_fetch, agent):
        mock_fetch.return_value = {}

        tools = agent.get_mcp_tools(["missing-server"])

        assert tools == []

    @patch("crewai.mcp.tool_resolver.MCPClient")
    @patch.object(MCPToolResolver, "_fetch_amp_mcp_configs")
    def test_legacy_crewai_amp_prefix_still_works(
        self, mock_fetch, mock_client_class, agent, mock_tool_definitions
    ):
        mock_fetch.return_value = {
            "notion": {
                "type": "sse",
                "url": "https://mcp.notion.so/sse",
                "headers": {"Authorization": "Bearer token"},
            },
        }

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        tools = agent.get_mcp_tools(["crewai-amp:notion"])

        mock_fetch.assert_called_once_with(["notion"])
        assert len(tools) == 2

    @patch("crewai.mcp.tool_resolver.MCPClient")
    @patch.object(MCPToolResolver, "_fetch_amp_mcp_configs")
    @patch.object(MCPToolResolver, "_resolve_external")
    def test_non_amp_items_unaffected(
        self,
        mock_external,
        mock_fetch,
        mock_client_class,
        agent,
        mock_tool_definitions,
    ):
        mock_fetch.return_value = {
            "notion": {
                "type": "sse",
                "url": "https://mcp.notion.so/sse",
            },
        }

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        mock_external_tool = MagicMock(spec=BaseTool)
        mock_external.return_value = [mock_external_tool]

        http_config = MCPServerHTTP(
            url="https://other.mcp.com/api",
            headers={"Authorization": "Bearer other"},
        )

        tools = agent.get_mcp_tools(
            [
                "notion",
                "https://external.mcp.com/api",
                http_config,
            ]
        )

        mock_fetch.assert_called_once_with(["notion"])
        mock_external.assert_called_once_with("https://external.mcp.com/api")
        # 2 from notion + 1 from external + 2 from http_config
        assert len(tools) == 5
