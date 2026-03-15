from unittest.mock import patch, MagicMock
import pytest
from crewai_tools import MeyhemSearchTool, MeyhemDiscoverTool


def test_search_tool_initialization():
    tool = MeyhemSearchTool()
    assert tool.name == "MeyhemSearchTool"
    assert tool.base_url == "https://api.rhdxm.com"
    assert tool.agent_id == "crewai-agent"


def test_search_tool_custom_base_url():
    tool = MeyhemSearchTool(base_url="https://custom.example.com", agent_id="my-agent")
    assert tool.base_url == "https://custom.example.com"
    assert tool.agent_id == "my-agent"


def test_discover_tool_initialization():
    tool = MeyhemDiscoverTool()
    assert tool.name == "MeyhemDiscoverTool"
    assert tool.base_url == "https://api.rhdxm.com"


@patch("crewai_tools.tools.meyhem_tool.meyhem_tool.httpx")
def test_search_tool_run(mock_httpx):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": [{"url": "https://example.com", "title": "Test", "snippet": "A test result"}]}
    mock_resp.raise_for_status = MagicMock()
    mock_httpx.post.return_value = mock_resp
    tool = MeyhemSearchTool()
    result = tool._run(query="test query", max_results=3)
    assert "example.com" in result
    mock_httpx.post.assert_called_once()


@patch("crewai_tools.tools.meyhem_tool.meyhem_tool.httpx")
def test_discover_tool_run(mock_httpx):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": [{"name": "test/server", "ecosystem": "mcp", "stars": 100}]}
    mock_resp.raise_for_status = MagicMock()
    mock_httpx.post.return_value = mock_resp
    tool = MeyhemDiscoverTool()
    result = tool._run(task="query postgres")
    assert "test/server" in result
    mock_httpx.post.assert_called_once()


@patch("crewai_tools.tools.meyhem_tool.meyhem_tool.httpx")
def test_search_tool_error_handling(mock_httpx):
    mock_httpx.post.side_effect = Exception("Connection failed")
    tool = MeyhemSearchTool()
    result = tool._run(query="test")
    assert "Search failed" in result


@patch("crewai_tools.tools.meyhem_tool.meyhem_tool.httpx")
def test_discover_tool_error_handling(mock_httpx):
    mock_httpx.post.side_effect = Exception("Connection failed")
    tool = MeyhemDiscoverTool()
    result = tool._run(task="test")
    assert "Discovery failed" in result
