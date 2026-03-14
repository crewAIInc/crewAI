import json
import os
from unittest.mock import patch

import pytest

from crewai_tools.tools.you_contents_tool.you_contents_tool import YouContentsTool


@pytest.fixture(autouse=True)
def mock_you_api_key():
    with patch.dict(os.environ, {"YOU_API_KEY": "test_key"}):
        yield


@pytest.fixture
def you_contents_tool():
    return YouContentsTool()


def test_you_contents_tool_initialization():
    tool = YouContentsTool()
    assert tool.formats == ["markdown"]
    assert tool.crawl_timeout == 10
    assert tool.timeout == 60
    assert tool.contents_url == "https://ydc-index.io/v1/contents"


def test_you_contents_tool_custom_initialization():
    tool = YouContentsTool(
        formats=["markdown", "html", "metadata"],
        crawl_timeout=45,
        timeout=90,
    )
    assert tool.formats == ["markdown", "html", "metadata"]
    assert tool.crawl_timeout == 45
    assert tool.timeout == 90


@patch("requests.post")
def test_you_contents_tool_single_url(mock_post, you_contents_tool):
    mock_response = {
        "results": [
            {
                "url": "https://example.com",
                "markdown": "# Test Content\n\nThis is test content.",
                "metadata": {
                    "title": "Test Page",
                    "description": "Test description",
                },
            }
        ]
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = you_contents_tool.run(urls="https://example.com")

    # Verify the result is valid JSON
    assert result is not None
    data = json.loads(result)
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["url"] == "https://example.com"
    assert "markdown" in data["results"][0]


@patch("requests.post")
def test_you_contents_tool_multiple_urls(mock_post, you_contents_tool):
    mock_response = {
        "results": [
            {
                "url": "https://example1.com",
                "markdown": "Content 1",
            },
            {
                "url": "https://example2.com",
                "markdown": "Content 2",
            },
        ]
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    urls = ["https://example1.com", "https://example2.com"]
    result = you_contents_tool.run(urls=urls)

    data = json.loads(result)
    assert len(data["results"]) == 2
    assert data["results"][0]["url"] == "https://example1.com"
    assert data["results"][1]["url"] == "https://example2.com"


@patch("requests.post")
def test_you_contents_tool_with_formats(mock_post):
    tool = YouContentsTool(formats=["markdown", "html"])
    mock_response = {"results": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool.run(urls="https://example.com")

    # Verify the API was called with correct payload
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert payload["urls"] == ["https://example.com"]
    assert payload["formats"] == ["markdown", "html"]


@patch("requests.post")
def test_you_contents_tool_with_crawl_timeout(mock_post):
    tool = YouContentsTool(crawl_timeout=45)
    mock_response = {"results": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool.run(urls="https://example.com")

    # Verify crawl_timeout is included and clamped to valid range (1-60)
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert payload["crawl_timeout"] == 45
    assert 1 <= payload["crawl_timeout"] <= 60


@patch("requests.post")
def test_you_contents_tool_crawl_timeout_clamping(mock_post):
    # Test that crawl_timeout is clamped to 1-60 range
    tool = YouContentsTool(crawl_timeout=100)  # Too high
    mock_response = {"results": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool.run(urls="https://example.com")

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert payload["crawl_timeout"] == 60  # Should be clamped to max

    # Test minimum clamping
    tool2 = YouContentsTool(crawl_timeout=0)  # Too low
    tool2.run(urls="https://example.com")

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert payload["crawl_timeout"] == 1  # Should be clamped to min


@patch("requests.post")
def test_you_contents_tool_api_error(mock_post):
    import requests

    tool = YouContentsTool()
    mock_post.side_effect = requests.RequestException("API Error")

    result = tool.run(urls="https://example.com")

    # Should return error string, not raise exception
    assert "Error extracting content" in result
    assert "API Error" in result


@patch("requests.post")
def test_you_contents_tool_request_exception(mock_post):
    import requests

    tool = YouContentsTool()
    mock_post.side_effect = requests.RequestException("Connection error")

    result = tool.run(urls="https://example.com")

    assert "Error extracting content" in result
    assert "Connection error" in result


def test_you_contents_tool_empty_urls():
    tool = YouContentsTool()

    result = tool.run(urls=[])

    # Should handle empty URL list gracefully
    assert "Invalid parameters" in result or "At least one URL is required" in result


@patch("requests.post")
def test_you_contents_tool_headers(mock_post):
    tool = YouContentsTool()
    mock_response = {"results": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool.run(urls="https://example.com")

    # Verify correct headers are sent
    call_args = mock_post.call_args
    headers = call_args.kwargs["headers"]
    assert "X-API-Key" in headers
    assert headers["X-API-Key"] == "test_key"
    assert headers["Content-Type"] == "application/json"


@patch("requests.post")
def test_you_contents_tool_url_normalization(mock_post):
    tool = YouContentsTool()
    mock_response = {"results": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    # Test that single URL string is normalized to list
    tool.run(urls="https://example.com")

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert isinstance(payload["urls"], list)
    assert payload["urls"] == ["https://example.com"]


if __name__ == "__main__":
    pytest.main([__file__])
