import json
import os
from unittest.mock import patch

import pytest

from crewai_tools.tools.you_search_tool.you_search_tool import YouSearchTool


@pytest.fixture(autouse=True)
def mock_you_api_key():
    with patch.dict(os.environ, {"YOU_API_KEY": "test_key"}):
        yield


@pytest.fixture
def you_search_tool():
    return YouSearchTool(count=5)


def test_you_search_tool_initialization():
    tool = YouSearchTool()
    assert tool.count == 10
    assert tool.offset is None
    assert tool.country == "US"
    assert tool.language is None
    assert tool.safesearch == "moderate"
    assert tool.timeout == 60
    assert tool.search_url == "https://api.ydc-index.io/search"


def test_you_search_tool_custom_initialization():
    tool = YouSearchTool(
        count=20,
        offset=2,
        country="GB",
        language="EN-GB",
        freshness="week",
        safesearch="strict",
        livecrawl="web",
        livecrawl_formats="html",
        timeout=30,
    )
    assert tool.count == 20
    assert tool.offset == 2
    assert tool.country == "GB"
    assert tool.language == "EN-GB"
    assert tool.freshness == "week"
    assert tool.safesearch == "strict"
    assert tool.livecrawl == "web"
    assert tool.livecrawl_formats == "html"
    assert tool.timeout == 30


@patch("requests.get")
def test_you_search_tool_search(mock_get, you_search_tool):
    mock_response = {
        "hits": [
            {
                "title": "Test Result 1",
                "url": "https://test1.com",
                "description": "Test description 1",
                "snippets": ["Test snippet 1"],
            },
            {
                "title": "Test Result 2",
                "url": "https://test2.com",
                "description": "Test description 2",
                "snippets": ["Test snippet 2"],
            },
        ]
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    result = you_search_tool.run(query="test query")

    # Verify the result is valid JSON
    assert result is not None
    data = json.loads(result)
    assert "hits" in data
    assert len(data["hits"]) == 2
    assert data["hits"][0]["title"] == "Test Result 1"
    assert data["hits"][0]["url"] == "https://test1.com"


@patch("requests.get")
def test_you_search_tool_with_query_param(mock_get):
    tool = YouSearchTool()
    mock_response = {"hits": []}
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    # Test with query parameter
    result = tool.run(query="test")
    assert result is not None

    # Verify the API was called with correct parameters
    call_args = mock_get.call_args
    assert call_args is not None
    assert "params" in call_args.kwargs
    assert call_args.kwargs["params"]["query"] == "test"


@patch("requests.get")
def test_you_search_tool_with_params(mock_get):
    tool = YouSearchTool(count=5, country="US", freshness="day", safesearch="moderate")
    mock_response = {"hits": []}
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    tool.run(query="test")

    # Verify the API was called with correct parameters
    call_args = mock_get.call_args
    params = call_args.kwargs["params"]
    assert params["query"] == "test"
    assert params["count"] == 5
    assert params["country"] == "US"
    assert params["freshness"] == "day"
    assert params["safesearch"] == "moderate"


@patch("requests.get")
def test_you_search_tool_with_offset(mock_get):
    tool = YouSearchTool(offset=2)
    mock_response = {"hits": []}
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    tool.run(query="test")

    # Verify offset is included
    call_args = mock_get.call_args
    params = call_args.kwargs["params"]
    assert params["offset"] == 2


@patch("requests.get")
def test_you_search_tool_offset_clamping(mock_get):
    """Test that offset is clamped to 0-9 range"""
    mock_response = {"hits": []}
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    # Test offset too high
    tool = YouSearchTool(offset=15)
    tool.run(query="test")
    call_args = mock_get.call_args
    assert call_args.kwargs["params"]["offset"] == 9  # Should be clamped to max

    # Test offset too low
    tool2 = YouSearchTool(offset=-5)
    tool2.run(query="test")
    call_args = mock_get.call_args
    assert call_args.kwargs["params"]["offset"] == 0  # Should be clamped to min


@patch("requests.get")
def test_you_search_tool_with_language(mock_get):
    tool = YouSearchTool(language="EN-GB")
    mock_response = {"hits": []}
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    tool.run(query="test")

    # Verify language parameter is included
    call_args = mock_get.call_args
    params = call_args.kwargs["params"]
    assert params["language"] == "EN-GB"


@patch("requests.get")
def test_you_search_tool_with_livecrawl(mock_get):
    tool = YouSearchTool(livecrawl="web", livecrawl_formats="markdown")
    mock_response = {"hits": []}
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    tool.run(query="test")

    # Verify livecrawl parameters are included
    call_args = mock_get.call_args
    params = call_args.kwargs["params"]
    assert params["livecrawl"] == "web"
    assert params["livecrawl_formats"] == "markdown"


@patch("requests.get")
def test_you_search_tool_api_error(mock_get):
    import requests

    tool = YouSearchTool()
    mock_get.side_effect = requests.RequestException("API Error")

    result = tool.run(query="test")

    # Should return error string, not raise exception
    assert "Error performing search" in result
    assert "API Error" in result


@patch("requests.get")
def test_you_search_tool_request_exception(mock_get):
    import requests

    tool = YouSearchTool()
    mock_get.side_effect = requests.RequestException("Connection error")

    result = tool.run(query="test")

    assert "Error performing search" in result
    assert "Connection error" in result


def test_you_search_tool_missing_query():
    tool = YouSearchTool()

    # Should raise TypeError for missing required argument
    with pytest.raises(TypeError):
        tool.run()


@patch("requests.get")
def test_you_search_tool_headers(mock_get):
    tool = YouSearchTool()
    mock_response = {"hits": []}
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    tool.run(query="test")

    # Verify correct headers are sent
    call_args = mock_get.call_args
    headers = call_args.kwargs["headers"]
    assert "X-API-Key" in headers
    assert headers["X-API-Key"] == "test_key"
    assert headers["Accept"] == "application/json"


if __name__ == "__main__":
    pytest.main([__file__])
