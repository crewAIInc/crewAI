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

    result = brave_tool.run(search_query="test")
    assert "Test Title" in result
    assert "http://test.com" in result


def test_brave_tool():
    tool = BraveSearchTool(
        n_results=2,
    )
    x = tool.run(search_query="ChatGPT")
    print(x)


if __name__ == "__main__":
    test_brave_tool()
    test_brave_tool_initialization()
    # test_brave_tool_search(brave_tool)
