from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.adapters.tool_collection import ToolCollection
from crewai_tools.tools.proxies_sx_tool.proxies_sx_tool import (
    ProxiesSxBrowserTool,
    ProxiesSxProxyTool,
)


def _mock_tool(name: str, result: str):
    tool = MagicMock()
    tool.name = name
    tool.run.return_value = result
    return tool


@patch("crewai_tools.tools.proxies_sx_tool.proxies_sx_tool.MCPServerAdapter")
def test_proxy_tool_dispatches_to_mcp_action_with_default_country(
    mock_mcp_server_adapter,
):
    get_proxy_tool = _mock_tool("get_proxy", "proxy-result")
    rotate_ip_tool = _mock_tool("rotate_ip", "rotate-result")
    check_status_tool = _mock_tool("check_status", "status-result")

    adapter = MagicMock()
    adapter.tools = ToolCollection([get_proxy_tool, rotate_ip_tool, check_status_tool])
    mock_mcp_server_adapter.return_value = adapter

    tool = ProxiesSxProxyTool(country="US")
    assert mock_mcp_server_adapter.call_count == 0

    result = tool.run(action="get_proxy", arguments={"session_id": "session-123"})

    assert result == "proxy-result"
    get_proxy_tool.run.assert_called_once_with(session_id="session-123", country="US")
    mock_mcp_server_adapter.assert_called_once()


@patch("crewai_tools.tools.proxies_sx_tool.proxies_sx_tool.MCPServerAdapter")
def test_proxy_tool_preserves_explicit_country_argument(mock_mcp_server_adapter):
    get_proxy_tool = _mock_tool("get_proxy", "proxy-result")
    rotate_ip_tool = _mock_tool("rotate_ip", "rotate-result")
    check_status_tool = _mock_tool("check_status", "status-result")

    adapter = MagicMock()
    adapter.tools = ToolCollection([get_proxy_tool, rotate_ip_tool, check_status_tool])
    mock_mcp_server_adapter.return_value = adapter

    tool = ProxiesSxProxyTool(country="US")
    tool.run(action="get_proxy", arguments={"country": "DE"})

    get_proxy_tool.run.assert_called_once_with(country="DE")


@patch("crewai_tools.tools.proxies_sx_tool.proxies_sx_tool.MCPServerAdapter")
def test_browser_tool_dispatches_to_selected_action(mock_mcp_server_adapter):
    browser_create_tool = _mock_tool("browser_create", "create-result")
    browser_click_tool = _mock_tool("browser_click", "click-result")

    adapter = MagicMock()
    adapter.tools = ToolCollection([browser_create_tool, browser_click_tool])
    mock_mcp_server_adapter.return_value = adapter

    tool = ProxiesSxBrowserTool()
    result = tool.run(action="browser_click", arguments={"selector": "#submit"})

    assert result == "click-result"
    browser_click_tool.run.assert_called_once_with(selector="#submit")


@patch("crewai_tools.tools.proxies_sx_tool.proxies_sx_tool.MCPServerAdapter")
def test_proxy_tool_raises_when_action_unavailable(mock_mcp_server_adapter):
    get_proxy_tool = _mock_tool("get_proxy", "proxy-result")

    adapter = MagicMock()
    adapter.tools = ToolCollection([get_proxy_tool])
    mock_mcp_server_adapter.return_value = adapter

    tool = ProxiesSxProxyTool()
    with pytest.raises(ValueError, match="rotate_ip"):
        tool.run(action="rotate_ip", arguments={"proxy_id": "abc"})


@patch("crewai_tools.tools.proxies_sx_tool.proxies_sx_tool.MCPServerAdapter")
def test_proxy_tool_reuses_adapter_and_stops_on_close(mock_mcp_server_adapter):
    check_status_tool = _mock_tool("check_status", "status-result")
    get_proxy_tool = _mock_tool("get_proxy", "proxy-result")
    rotate_ip_tool = _mock_tool("rotate_ip", "rotate-result")

    adapter = MagicMock()
    adapter.tools = ToolCollection([get_proxy_tool, rotate_ip_tool, check_status_tool])
    mock_mcp_server_adapter.return_value = adapter

    tool = ProxiesSxProxyTool()
    tool.run(action="check_status", arguments={"proxy_id": "proxy-1"})
    tool.run(action="check_status", arguments={"proxy_id": "proxy-2"})
    tool.close()

    mock_mcp_server_adapter.assert_called_once()
    assert check_status_tool.run.call_count == 2
    adapter.stop.assert_called_once()
