from typing import Any
from unittest.mock import MagicMock, patch

import requests

from crewai_tools.tools.edgepedia_tool.edgepedia_tool import EdgepediaTool


SAMPLE_SEARCH_RESPONSE: dict[str, Any] = {
    "query": "Mount Fuji",
    "match": "exact",
    "total": 1,
    "results": [
        {
            "id": "epew8wpqdv",
            "slug": "mount-fuji",
            "title": "Mount Fuji (富士山)",
            "topic": "Places and geography",
            "url": "https://www.edgepedia.org/mount-fuji",
        }
    ],
}

SAMPLE_ARTICLE_EN: dict[str, Any] = {
    "id": "epew8wpqdv",
    "slug": "mount-fuji",
    "title": "Mount Fuji (富士山)",
    "updated": "2026-10-21",
    "markdown": (
        "# Mount Fuji\n\n"
        "Mount Fuji is Japan's highest peak at 3,776 m and last erupted in 1707."
    ),
    "url": "https://www.edgepedia.org/mount-fuji",
    "credit": (
        '"Mount Fuji (富士山)", Edgepedia (EdgeChat), '
        "https://www.edgepedia.org/mount-fuji. Edgepedia Community License 1.0."
    ),
}

SAMPLE_ARTICLE_ZH: dict[str, Any] = {
    **SAMPLE_ARTICLE_EN,
    "title": "富士山",
    "lang": "zh",
    "url": "https://www.edgepedia.org/zh/mount-fuji",
    "markdown": "# 富士山\n\n富士山是日本最高峰，海拔3,776米 [1]。",
}


def _mock_response(payload: dict[str, Any], status_code: int = 200) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_edgepedia_tool_search_and_fetch_english() -> None:
    tool = EdgepediaTool()
    with patch("requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(SAMPLE_SEARCH_RESPONSE),
            _mock_response(SAMPLE_ARTICLE_EN),
        ]
        result = tool._run(query="Mount Fuji")

    assert "Mount Fuji" in result
    assert "3,776 m" in result
    assert "https://www.edgepedia.org/mount-fuji" in result
    assert "Credit:" in result
    assert "Edgepedia Community License 1.0" in result

    assert mock_get.call_count == 2
    search_call = mock_get.call_args_list[0]
    article_call = mock_get.call_args_list[1]
    assert search_call.args[0] == "https://www.edgepedia.org/api/v1/search"
    assert search_call.kwargs["params"] == {"q": "Mount Fuji", "limit": 1}
    assert article_call.args[0] == "https://www.edgepedia.org/api/v1/articles/mount-fuji"
    assert article_call.kwargs["params"] is None


def test_edgepedia_tool_multilingual_and_compact_citations() -> None:
    tool = EdgepediaTool()
    with patch("requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(SAMPLE_SEARCH_RESPONSE),
            _mock_response(SAMPLE_ARTICLE_ZH),
        ]
        result = tool._run(query="Mount Fuji", lang="zh", citations=False)

    assert "富士山" in result
    assert "https://www.edgepedia.org/zh/mount-fuji" in result
    assert "Credit:" in result
    article_call = mock_get.call_args_list[1]
    assert article_call.kwargs["params"] == {"lang": "zh", "citations": "false"}


def test_edgepedia_tool_preserves_credit_when_truncated() -> None:
    long_article = {
        **SAMPLE_ARTICLE_EN,
        "markdown": "Mount Fuji fact. " * 200,
    }
    tool = EdgepediaTool(max_chars_per_article=220)
    with patch("requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(SAMPLE_SEARCH_RESPONSE),
            _mock_response(long_article),
        ]
        result = tool._run(query="Mount Fuji")

    assert "Credit:" in result
    assert "Edgepedia Community License 1.0" in result


def test_edgepedia_tool_handles_empty_and_http_errors() -> None:
    tool = EdgepediaTool()
    assert "non-empty" in tool._run(query="   ")

    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response({"query": "unknown", "results": []})
        empty_result = tool._run(query="nonexistent-subject-xyz")
    assert "No Edgepedia articles found" in empty_result

    with patch("requests.get", side_effect=requests.Timeout("timed out")):
        error_result = tool._run(query="Mount Fuji")
    assert "Failed to query Edgepedia API" in error_result
