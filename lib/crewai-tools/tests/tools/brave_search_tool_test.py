import json
from unittest.mock import patch

import pytest

from crewai_tools.tools.brave_search_tool.brave_search_tool import BraveSearchTool


@pytest.fixture
def brave_tool():
    return BraveSearchTool(n_results=2)


def test_brave_tool_initialization():
    tool = BraveSearchTool()
    assert tool.n_results == 10
    assert tool.save_file is False


@patch("requests.get")
def test_brave_tool_search(mock_get, brave_tool):
    mock_response = {
        "web": {
            "results": [
                {
                    "title": "Test Title",
                    "url": "http://test.com",
                    "description": "Test Description",
                }
            ]
        }
    }
    mock_get.return_value.json.return_value = mock_response

    result = brave_tool.run(query="test")
    assert "Test Title" in result
    assert "http://test.com" in result


@patch("requests.get")
def test_brave_tool(mock_get):
    mock_response = {
        "web": {
            "results": [
                {
                    "title": "Brave Browser",
                    "url": "https://brave.com",
                    "description": "Brave Browser description",
                }
            ]
        }
    }
    mock_get.return_value.json.return_value = mock_response

    tool = BraveSearchTool(n_results=2)
    result = tool.run(query="Brave Browser")
    assert result is not None

    # Parse JSON so we can examine the structure
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) >= 1

    # First item should have expected fields: title, url, and description
    first = data[0]
    assert "title" in first
    assert first["title"] == "Brave Browser"
    assert "url" in first
    assert first["url"] == "https://brave.com"
    assert "description" in first
    assert first["description"] == "Brave Browser description"


if __name__ == "__main__":
    test_brave_tool()
    test_brave_tool_initialization()
    # test_brave_tool_search(brave_tool)
