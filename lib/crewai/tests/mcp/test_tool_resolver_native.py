"""Tests for MCPToolResolver native (non-AMP) resolution paths."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.agent.core import Agent
from crewai.llms.providers.utils.common import safe_tool_conversion
from crewai.mcp.config import MCPServerHTTP, MCPServerStdio
from crewai.mcp.tool_resolver import MCPToolResolver


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
def http_config():
    return MCPServerHTTP(url="https://mcp.example.com/api")


class TestResolveNativeEmptyTools:
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_logs_warning_and_returns_empty_when_server_has_no_tools(
        self, mock_client_class, resolver, http_config
    ):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        mock_log = MagicMock()
        resolver._logger = MagicMock(log=mock_log)

        tools, clients = resolver._resolve_native(http_config)

        assert tools == []
        assert clients == []
        warning_calls = [
            call for call in mock_log.call_args_list if call.args[0] == "warning"
        ]
        assert any(
            "No tools discovered from MCP server" in call.args[1]
            for call in warning_calls
        )

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_logs_warning_when_tool_filter_removes_all_tools(
        self, mock_client_class, resolver
    ):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "search", "description": "Search"}]
        )
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        config = MCPServerHTTP(
            url="https://mcp.example.com/api",
            tool_filter=lambda _tool: False,
        )

        mock_log = MagicMock()
        resolver._logger = MagicMock(log=mock_log)

        tools, clients = resolver._resolve_native(config)

        assert tools == []
        assert clients == []
        warning_calls = [
            call for call in mock_log.call_args_list if call.args[0] == "warning"
        ]
        assert any(
            "No tools discovered from MCP server" in call.args[1]
            for call in warning_calls
        )


class TestResolveNativeRuntimeError:
    @patch("crewai.mcp.tool_resolver.asyncio.run")
    def test_unmatched_runtime_error_is_wrapped_not_swallowed(
        self, mock_asyncio_run, resolver, http_config
    ):
        mock_asyncio_run.side_effect = RuntimeError("some other failure")

        with pytest.raises(ConnectionError, match="Failed to get native MCP tools"):
            resolver._resolve_native(http_config)


class TestResolveNativeToolNames:
    @staticmethod
    def _resolve_tool_names(mock_client_class, resolver, config):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "add", "description": "Add numbers"}]
        )
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        if isinstance(config, str):
            config = MCPServerHTTP(url=config)
        tools, _ = resolver._resolve_native(config)
        return [tool.name for tool in tools]

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_ip_addressed_server_tool_name_passes_provider_validation(
        self, mock_client_class, resolver
    ):
        names = self._resolve_tool_names(
            mock_client_class, resolver, "http://127.0.0.1:8800/mcp"
        )

        name, _, _ = safe_tool_conversion(
            {"name": names[0], "description": "Add numbers"}, "OpenAI"
        )
        assert name == "mcp_127_0_0_1_8800_mcp_add"

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_hostname_server_tool_name_is_unchanged(
        self, mock_client_class, resolver
    ):
        names = self._resolve_tool_names(
            mock_client_class, resolver, "http://localhost:8800/mcp"
        )

        assert names == ["localhost:8800_mcp_add"]

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_url_credentials_are_not_in_tool_name(self, mock_client_class, resolver):
        names = self._resolve_tool_names(
            mock_client_class, resolver, "http://user:secret@10.0.0.5:8800/mcp"
        )

        assert names == ["mcp_10_0_0_5:8800_mcp_add"]

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_digit_leading_stdio_server_tool_name_passes_provider_validation(
        self, mock_client_class, resolver
    ):
        names = self._resolve_tool_names(
            mock_client_class, resolver, MCPServerStdio(command="3d-server")
        )

        name, _, _ = safe_tool_conversion(
            {"name": names[0], "description": "Add numbers"}, "OpenAI"
        )
        assert name == "mcp_3d_server_add"

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_non_ascii_host_that_sanitizes_to_a_letter_is_not_prefixed(
        self, mock_client_class, resolver
    ):
        names = self._resolve_tool_names(
            mock_client_class, resolver, "http://\u00f1andu.example/mcp"
        )

        assert names == ["\u00f1andu_example_mcp_add"]
