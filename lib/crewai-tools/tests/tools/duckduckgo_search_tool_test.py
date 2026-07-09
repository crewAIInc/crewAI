from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.duckduckgo_search_tool.duckduckgo_search_tool import (
    DuckDuckGoSearchTool,
)


@pytest.fixture
def ddg_tool():
    return DuckDuckGoSearchTool(n_results=2)


def test_ddg_tool_initialization():
    tool = DuckDuckGoSearchTool()
    assert tool.n_results == 10
    assert tool.region == "wt-wt"
    assert tool.safesearch == "moderate"
    assert tool.name == "DuckDuckGo Web Search"


def test_ddg_empty_query_guard(ddg_tool):
    # _run defensively guards against an empty query (the args_schema normally
    # enforces a non-empty search_query before _run is reached).
    result = ddg_tool._run(search_query="")
    assert "search_query" in result


@patch("ddgs.DDGS")
def test_ddg_tool_search(mock_ddgs, ddg_tool):
    instance = MagicMock()
    instance.text.return_value = [
        {
            "title": "Test Title",
            "href": "http://test.com",
            "body": "Test Snippet",
        }
    ]
    mock_ddgs.return_value = instance

    result = ddg_tool.run(search_query="test")

    assert "Test Title" in result
    assert "http://test.com" in result
    assert "Test Snippet" in result
    instance.text.assert_called_once_with(
        "test", region="wt-wt", safesearch="moderate", max_results=2
    )


@patch("ddgs.DDGS")
def test_ddg_tool_no_results(mock_ddgs, ddg_tool):
    instance = MagicMock()
    instance.text.return_value = []
    mock_ddgs.return_value = instance

    result = ddg_tool.run(search_query="nonexistent query")
    assert "No results found" in result


@patch("ddgs.DDGS")
def test_ddg_tool_handles_errors(mock_ddgs, ddg_tool):
    instance = MagicMock()
    instance.text.side_effect = RuntimeError("network down")
    mock_ddgs.return_value = instance

    result = ddg_tool.run(search_query="test")
    assert "Error performing DuckDuckGo search" in result
