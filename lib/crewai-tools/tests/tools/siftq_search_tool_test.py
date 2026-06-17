import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests import RequestException

from crewai_tools.tools.siftq_search_tool.siftq_search_tool import (
    DEFAULT_SIFTQ_API_KEY,
    SIFTQ_API_URL,
    SiftqSearchTool,
)


@pytest.fixture(autouse=True)
def mock_siftq_api_key():
    with patch.dict(os.environ, {"SIFTQ_API_KEY": "test_key"}):
        yield


@pytest.fixture
def siftq_tool():
    return SiftqSearchTool()


def test_default_initialization(siftq_tool):
    assert siftq_tool.scope == "webpage"
    assert siftq_tool.include_summary is False
    assert siftq_tool.include_raw_content is False
    assert siftq_tool.concise_snippet is False
    assert siftq_tool.max_results == 5
    assert siftq_tool.timeout == 60
    assert siftq_tool.max_content_length_per_result == 1000


def test_custom_initialization():
    tool = SiftqSearchTool(
        scope="scholar",
        include_summary=True,
        include_raw_content=True,
        concise_snippet=True,
        max_results=10,
        timeout=30,
        max_content_length_per_result=500,
        api_key="custom_key",
    )
    assert tool.scope == "scholar"
    assert tool.include_summary is True
    assert tool.include_raw_content is True
    assert tool.concise_snippet is True
    assert tool.max_results == 10
    assert tool.timeout == 30
    assert tool.max_content_length_per_result == 500
    assert tool.api_key == "custom_key"


def test_api_key_falls_back_to_env(siftq_tool):
    assert siftq_tool.api_key == "test_key"


def test_api_key_uses_default_when_not_set():
    with patch.dict(os.environ, {}, clear=True):
        tool = SiftqSearchTool()
        assert tool.api_key is None


def test_get_api_key_uses_default_fallback():
    with patch.dict(os.environ, {}, clear=True):
        tool = SiftqSearchTool()
        assert tool._get_api_key() == DEFAULT_SIFTQ_API_KEY


def test_get_api_key_uses_env_value():
    with patch.dict(os.environ, {"SIFTQ_API_KEY": "env_key"}):
        tool = SiftqSearchTool()
        assert tool._get_api_key() == "env_key"


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_basic_search(mock_post):
    mock_response = {
        "credits": 1,
        "total": 2,
        "webpages": [
            {
                "title": "Result 1",
                "link": "https://example.com/1",
                "snippet": "Snippet 1",
                "content": "Full content 1",
                "position": 1,
            },
            {
                "title": "Result 2",
                "link": "https://example.com/2",
                "snippet": "Snippet 2",
                "content": "Full content 2",
                "position": 2,
            },
        ],
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = MagicMock()

    tool = SiftqSearchTool()
    result = json.loads(tool.run("test query"))

    mock_post.assert_called_once_with(
        SIFTQ_API_URL,
        headers={
            "Authorization": "Bearer test_key",
            "Content-Type": "application/json",
        },
        json={
            "q": "test query",
            "scope": "webpage",
            "includeSummary": False,
            "includeRawContent": False,
            "conciseSnippet": False,
            "size": 5,
        },
        timeout=60,
    )
    assert "webpages" in result
    assert result["webpages"][0]["title"] == "Result 1"
    assert result["webpages"][1]["title"] == "Result 2"


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_with_content_truncation(mock_post):
    long_content = "A" * 2000
    mock_response = {
        "credits": 1,
        "total": 1,
        "webpages": [
            {
                "title": "Long Result",
                "link": "https://example.com/long",
                "snippet": "Short snippet",
                "content": long_content,
                "position": 1,
            },
        ],
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool = SiftqSearchTool(max_content_length_per_result=100)
    result = json.loads(tool.run("test query"))

    assert len(result["webpages"][0]["content"]) == 100 + 3  # 100 + "..."
    assert result["webpages"][0]["content"].endswith("...")


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_uses_snippet_when_no_content(mock_post):
    mock_response = {
        "credits": 1,
        "total": 1,
        "webpages": [
            {
                "title": "No Content",
                "link": "https://example.com/nocontent",
                "snippet": "This is a snippet without content field",
                "position": 1,
            },
        ],
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool = SiftqSearchTool(max_content_length_per_result=500)
    result = json.loads(tool.run("test query"))

    assert result["webpages"][0]["snippet"] == "This is a snippet without content field"
    assert "content" not in result["webpages"][0]


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_truncates_snippet_in_place(mock_post):
    long_snippet = "B" * 2000
    mock_response = {
        "credits": 1,
        "total": 1,
        "webpages": [
            {
                "title": "Long Snippet",
                "link": "https://example.com/longsnippet",
                "snippet": long_snippet,
                "position": 1,
            },
        ],
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool = SiftqSearchTool(max_content_length_per_result=100)
    result = json.loads(tool.run("test query"))

    assert len(result["webpages"][0]["snippet"]) == 103  # 100 + "..."
    assert result["webpages"][0]["snippet"].endswith("...")
    assert "content" not in result["webpages"][0]


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_with_scholar_scope(mock_post):
    mock_response = {
        "credits": 1,
        "total": 1,
        "scholars": [
            {
                "title": "A Scholarly Paper",
                "authors": ["Author A", "Author B"],
                "link": "https://scholar.example.com/paper",
                "snippet": "Abstract of the paper",
                "year": 2024,
                "venue": "Journal of AI",
                "citationCount": 42,
            },
        ],
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool = SiftqSearchTool(scope="scholar")
    result = json.loads(tool.run("machine learning"))

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["scope"] == "scholar"
    assert result["scholars"][0]["title"] == "A Scholarly Paper"
    assert result["scholars"][0]["citationCount"] == 42


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_with_include_raw_content(mock_post):
    mock_response = {"credits": 1, "total": 0, "webpages": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    tool = SiftqSearchTool(include_raw_content=True)
    tool.run("test query")

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["includeRawContent"] is True


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_api_error_handling(mock_post):
    mock_post.side_effect = RequestException("API Error")

    tool = SiftqSearchTool()
    with pytest.raises(RuntimeError, match="API Error"):
        tool.run("test query")


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_http_error_handling(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    mock_post.return_value = mock_response

    tool = SiftqSearchTool()
    with pytest.raises(RuntimeError, match="401"):
        tool.run("test query")


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
@pytest.mark.asyncio
async def test_arun(mock_post):
    mock_response = {
        "credits": 1,
        "total": 1,
        "webpages": [
            {
                "title": "Async Result",
                "link": "https://example.com/async",
                "snippet": "Async snippet",
                "position": 1,
            },
        ],
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = MagicMock()

    tool = SiftqSearchTool()
    result = json.loads(await tool._arun("async query"))

    assert result["webpages"][0]["title"] == "Async Result"


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_with_malformed_response(mock_post):
    mock_post.return_value.json.side_effect = ValueError("No JSON")
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = MagicMock()

    tool = SiftqSearchTool()
    with pytest.raises(RuntimeError, match="invalid JSON"):
        tool.run("test query")


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_with_empty_response(mock_post):
    mock_post.return_value.json.return_value = {}
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = MagicMock()

    tool = SiftqSearchTool()
    result = json.loads(tool.run("test query"))
    assert result == {}


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_timeout_error(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

    tool = SiftqSearchTool()
    with pytest.raises(TimeoutError, match="timed out"):
        tool.run("test query")


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("No connection")

    tool = SiftqSearchTool()
    with pytest.raises(ConnectionError, match="Could not connect"):
        tool.run("test query")


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_scope_override_per_query(mock_post):
    mock_response = {
        "credits": 1,
        "total": 1,
        "videos": [
            {
                "title": "A Video",
                "link": "https://example.com/video",
                "snippet": "Video description",
                "position": 1,
            },
        ],
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = MagicMock()

    tool = SiftqSearchTool()  # default scope is "webpage"
    result = json.loads(tool.run("test query", scope="video"))

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["scope"] == "video"
    assert result["videos"][0]["title"] == "A Video"


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_scope_override_none_uses_default(mock_post):
    mock_response = {"credits": 1, "total": 0, "webpages": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = MagicMock()

    tool = SiftqSearchTool(scope="scholar")
    tool.run("test query")

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["scope"] == "scholar"


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_per_query_overrides(mock_post):
    mock_response = {"credits": 1, "total": 0, "scholars": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = MagicMock()

    tool = SiftqSearchTool()  # all defaults
    tool.run(
        "test query",
        scope="scholar",
        include_summary=True,
        include_raw_content=True,
        concise_snippet=True,
        max_results=20,
    )

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["scope"] == "scholar"
    assert called_payload["includeSummary"] is True
    assert called_payload["includeRawContent"] is True
    assert called_payload["conciseSnippet"] is True
    assert called_payload["size"] == 20


@patch("crewai_tools.tools.siftq_search_tool.siftq_search_tool.requests.post")
def test_run_per_query_overrides_do_not_mutate_instance(mock_post):
    mock_response = {"credits": 1, "total": 0, "webpages": []}
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200
    mock_post.return_value.raise_for_status = MagicMock()

    tool = SiftqSearchTool(include_summary=False, max_results=5)
    tool.run(
        "test query",
        include_summary=True,
        max_results=50,
    )

    # Instance defaults should remain unchanged
    assert tool.include_summary is False
    assert tool.max_results == 5
