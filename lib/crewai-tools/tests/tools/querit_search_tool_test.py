"""Tests for the Querit search tool."""

from collections.abc import Generator
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools.tools.querit_search_tool import QueritSearchTool


@pytest.fixture(autouse=True)
def querit_api_key() -> Generator[None, None, None]:
    """Provide a Querit API key for tests that instantiate the tool."""
    with patch.dict(os.environ, {"QUERIT_API_KEY": "test-api-key"}):
        yield


def _mock_response(json_data: Any) -> MagicMock:
    """Create a mocked requests response returning the given JSON payload."""
    response = MagicMock()
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response


def _mock_http_error_response(status_code: int) -> MagicMock:
    """Create a mocked requests response that raises an HTTP status error."""
    response = MagicMock()
    response.status_code = status_code
    error = requests.HTTPError(f"{status_code} error")
    error.response = response
    response.raise_for_status.side_effect = error
    return response


def test_missing_api_key_raises() -> None:
    """Verify that the tool requires a Querit API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="QUERIT_API_KEY"):
            QueritSearchTool()


def test_querit_search_tool_posts_expected_payload() -> None:
    """Verify that the tool posts the expected request and returns raw results."""
    tool = QueritSearchTool(count=2, chunksPerDoc=None)
    response_data = {
        "took": "300ms",
        "error_code": 200,
        "error_msg": "",
        "search_id": 11099848653006015581,
        "query_context": {"query": "expanded query", "count": 99},
        "results": {
            "result": [
                {
                    "url": "https://example.com/llm",
                    "title": "LLM progress",
                    "snippet": "Recent LLM progress summary",
                    "site_name": "example.com",
                    "site_icon": "https://example.com/favicon.ico",
                    "page_age": "2025-07-20T16:00:00Z",
                }
            ]
        },
    }

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        return_value=_mock_response(response_data),
    ) as post:
        result = tool.run(query="latest LLM progress")

    post.assert_called_once_with(
        "https://api.querit.ai/v1/search",
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json",
        },
        json={"query": "latest LLM progress", "count": 2},
        timeout=30,
    )
    assert result == response_data


def test_querit_search_tool_uses_default_count_and_chunks_per_doc() -> None:
    """Verify that default result count and chunks per document are sent."""
    tool = QueritSearchTool()

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        return_value=_mock_response({"results": {"result": []}}),
    ) as post:
        tool.run(query="AI news")

    assert post.call_args.kwargs["json"] == {
        "query": "AI news",
        "count": 10,
        "chunksPerDoc": 3,
    }


def test_querit_search_tool_accepts_runtime_count() -> None:
    """Verify that runtime count overrides the tool default."""
    tool = QueritSearchTool(count=5, chunksPerDoc=None)

    response_data = {"results": {"result": []}}
    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        return_value=_mock_response(response_data),
    ) as post:
        result = tool.run(query="AI news", count=1)

    assert post.call_args.kwargs["json"] == {
        "query": "AI news",
        "count": 1,
    }
    assert result == response_data


def test_querit_search_tool_posts_flat_filter_parameters() -> None:
    """Verify that flat filter parameters are mapped to Querit filters."""
    tool = QueritSearchTool(
        count=5,
        chunksPerDoc=1,
        site_include=["example.com"],
        site_exclude=["archive.example.com"],
        time_range="w1",
        country_include=["US", "CA"],
        language_include=["en", "fr"],
    )

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        return_value=_mock_response({"results": {"result": []}}),
    ) as post:
        tool.run(query="AI search news")

    assert post.call_args.kwargs["json"] == {
        "query": "AI search news",
        "count": 5,
        "chunksPerDoc": 1,
        "filters": {
            "sites": {
                "include": ["example.com"],
                "exclude": ["archive.example.com"],
            },
            "timeRange": {"date": "w1"},
            "geo": {"countries": {"include": ["US", "CA"]}},
            "languages": {"include": ["en", "fr"]},
        },
    }


def test_querit_search_tool_serializes_time_range_filter() -> None:
    """Verify that time range input is serialized into the Querit filter shape."""
    tool = QueritSearchTool(count=5, chunksPerDoc=None)

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        return_value=_mock_response({"results": {"result": []}}),
    ) as post:
        tool.run(query="AI news", time_range="2026-01-01to2026-01-31")

    assert post.call_args.kwargs["json"] == {
        "query": "AI news",
        "count": 5,
        "filters": {"timeRange": {"date": "2026-01-01to2026-01-31"}},
    }


def test_querit_search_tool_rejects_invalid_time_range_format() -> None:
    """Verify that unsupported time range values fail validation."""
    tool = QueritSearchTool()

    with pytest.raises(ValueError, match="String should match pattern"):
        tool.run(query="AI news", time_range="past_week")


def test_querit_search_tool_retries_request_exceptions_three_times() -> None:
    """Verify that transient request exceptions are retried up to three times."""
    tool = QueritSearchTool()
    response_data = {"results": {"result": []}}
    response = _mock_response(response_data)

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        side_effect=[
            requests.ConnectTimeout("first timeout"),
            requests.ConnectTimeout("second timeout"),
            response,
        ],
    ) as post:
        result = tool.run(query="AI news")

    assert post.call_count == 3
    assert result == response_data


def test_querit_search_tool_retries_transient_http_errors_three_times() -> None:
    """Verify that transient HTTP status errors are retried up to three times."""
    tool = QueritSearchTool()
    response_data = {"results": {"result": []}}
    response = _mock_response(response_data)

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        side_effect=[
            _mock_http_error_response(429),
            _mock_http_error_response(500),
            response,
        ],
    ) as post:
        result = tool.run(query="AI news")

    assert post.call_count == 3
    assert result == response_data


def test_querit_search_tool_does_not_retry_auth_http_errors() -> None:
    """Verify that non-transient HTTP status errors are not retried."""
    tool = QueritSearchTool()

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        return_value=_mock_http_error_response(401),
    ) as post:
        with pytest.raises(requests.HTTPError):
            tool.run(query="AI news")

    post.assert_called_once()


def test_querit_search_tool_rejects_invalid_chunks_per_doc() -> None:
    """Verify that chunks per document cannot exceed the Querit limit."""
    tool = QueritSearchTool()

    with pytest.raises(ValueError, match="less than or equal to 3"):
        tool.run(query="AI news", chunksPerDoc=4)


def test_querit_search_tool_returns_raw_response() -> None:
    """Verify that the tool returns the original Querit API response."""
    raw_response = {"results": {"result": {"url": "https://example.com"}}}
    tool = QueritSearchTool()

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        return_value=_mock_response(raw_response),
    ):
        result = tool.run(query="AI news")

    assert result == raw_response


def test_querit_search_tool_rejects_non_object_response() -> None:
    """Verify that non-object Querit API responses are rejected."""
    tool = QueritSearchTool()

    with patch(
        "crewai_tools.tools.querit_search_tool.querit_search_tool.requests.post",
        return_value=_mock_response(["unexpected"]),
    ):
        with pytest.raises(ValueError, match="Querit API response"):
            tool.run(query="AI news")
