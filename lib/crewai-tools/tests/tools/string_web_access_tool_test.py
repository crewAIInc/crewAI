import json
import os
from unittest.mock import Mock, patch

from crewai_tools.tools.string_web_access_tool.string_web_access_tool import (
    StringWebAccessScrapeTool,
    StringWebAccessSearchTool,
)
import pytest


@pytest.fixture(autouse=True)
def mock_string_api_key():
    with patch.dict(os.environ, {"STRING_API_KEY": "test_key"}):
        yield


@pytest.fixture(autouse=True)
def mock_validate_url():
    # the scrape tool's real validate_url() does a DNS lookup (SSRF guard);
    # stub it to the identity function so tests never touch the network
    with patch(
        "crewai_tools.tools.string_web_access_tool.string_web_access_tool.validate_url",
        side_effect=lambda url: url,
    ):
        yield


def _mock_response(*, json_body=None, text=None):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_body
    response.text = text
    return response


def test_scrape_tool_initialization():
    tool = StringWebAccessScrapeTool()
    assert tool.name == "String Web Access scrape tool"
    assert tool.timeout == 120
    assert tool.ignore_failures is False
    assert [env.name for env in tool.env_vars] == ["STRING_API_KEY"]


def test_search_tool_initialization():
    tool = StringWebAccessSearchTool()
    assert tool.name == "String Web Access search tool"
    assert [env.name for env in tool.env_vars] == ["STRING_API_KEY"]


@patch("crewai_tools.tools.string_web_access_tool.string_web_access_tool.requests.post")
def test_scrape_returns_markdown(mock_post):
    mock_post.return_value = _mock_response(text="# Example Domain")

    result = StringWebAccessScrapeTool()._run(url="https://example.com")

    assert result == "# Example Domain"
    args, kwargs = mock_post.call_args
    assert args[0] == "https://request.usestring.ai/v1/fetch"
    assert kwargs["json"] == {"url": "https://example.com", "format": "markdown"}
    assert kwargs["headers"]["Authorization"] == "Bearer test_key"
    assert kwargs["timeout"] == 120


@patch("crewai_tools.tools.string_web_access_tool.string_web_access_tool.requests.post")
def test_scrape_passes_optional_fields_only_when_set(mock_post):
    mock_post.return_value = _mock_response(text="page")

    StringWebAccessScrapeTool()._run(
        url="https://example.com",
        main_content_only=True,
        execute_js=True,
        country_code="GB",
    )

    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {
        "url": "https://example.com",
        "format": "markdown",
        "mainContentOnly": True,
        "executeJS": True,
        "countryCode": "GB",
    }


@patch("crewai_tools.tools.string_web_access_tool.string_web_access_tool.requests.post")
def test_scrape_json_format_returns_the_envelope(mock_post):
    envelope = {"statusCode": 200, "headers": {}, "data": {"message": "Hello World"}}
    mock_post.return_value = _mock_response(json_body=envelope)

    result = StringWebAccessScrapeTool()._run(url="https://example.com", format="json")

    assert json.loads(result) == envelope


@patch("crewai_tools.tools.string_web_access_tool.string_web_access_tool.requests.post")
def test_scrape_raises_by_default_and_is_silent_when_ignoring_failures(mock_post):
    mock_post.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        StringWebAccessScrapeTool()._run(url="https://example.com")

    assert (
        StringWebAccessScrapeTool(ignore_failures=True)._run(url="https://example.com")
        is None
    )


@patch("crewai_tools.tools.string_web_access_tool.string_web_access_tool.requests.post")
def test_search_returns_organic_results(mock_post):
    results = [
        {
            "position": 1,
            "title": "First",
            "url": "https://example.com/1",
            "snippet": "One",
        },
        {
            "position": 2,
            "title": "Second",
            "url": "https://example.com/2",
            "snippet": "Two",
        },
    ]
    mock_post.return_value = _mock_response(
        json_body={"results": results, "zeroResults": False}
    )

    result = StringWebAccessSearchTool()._run(query="running shoes")

    assert json.loads(result) == results
    args, kwargs = mock_post.call_args
    assert args[0] == "https://request.usestring.ai/v1/search"
    assert kwargs["json"] == {
        "query": "running shoes",
        "engine": "google",
        "country": "US",
    }


@patch("crewai_tools.tools.string_web_access_tool.string_web_access_tool.requests.post")
def test_search_honours_engine_language_and_max_results(mock_post):
    mock_post.return_value = _mock_response(
        json_body={
            "results": [{"position": n} for n in range(1, 6)],
            "zeroResults": False,
        }
    )

    result = StringWebAccessSearchTool()._run(
        query="anything", engine="brave", country="GB", language="en", max_results=2
    )

    assert len(json.loads(result)) == 2
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {
        "query": "anything",
        "engine": "brave",
        "country": "GB",
        "language": "en",
    }


@patch("crewai_tools.tools.string_web_access_tool.string_web_access_tool.requests.post")
def test_search_reports_no_results(mock_post):
    mock_post.return_value = _mock_response(
        json_body={"results": [], "zeroResults": True}
    )

    assert "No results found" in StringWebAccessSearchTool()._run(
        query="nothing at all"
    )


@patch("crewai_tools.tools.string_web_access_tool.string_web_access_tool.requests.post")
def test_missing_api_key_raises_before_any_request(mock_post):
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="STRING_API_KEY"),
    ):
        StringWebAccessSearchTool()._run(query="anything")

    mock_post.assert_not_called()
