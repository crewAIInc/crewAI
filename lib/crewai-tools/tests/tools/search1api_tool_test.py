import json
import os
import socket
from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
import pytest
import requests

from crewai_tools import (
    Search1APICrawlTool,
    Search1APINewsTool,
    Search1APISearchTool,
)


POST = "crewai_tools.tools.search1api_tool.search1api_tool.requests.post"


def _response(
    status_code: int = 200, json_data: object = None, text: str = ""
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.text = text
    if json_data is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = json_data
    return resp


@pytest.fixture(autouse=True)
def _api_key():
    with patch.dict(os.environ, {"SEARCH1API_API_KEY": "s1-test-key"}):
        yield


@pytest.fixture(autouse=True)
def _public_example_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    # The crawl tool resolves the URL host before posting; keep tests offline.
    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any):
        if host == "example.com":
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
            ]
        return original_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def test_search_sends_request_and_normalizes_results():
    tool = Search1APISearchTool(
        search_service="github",
        max_results=3,
        crawl_results=1,
        time_range="week",
        include_sites=["github.com"],
        language="en",
        max_content_length_per_result=5,
    )
    body = {
        "searchParameters": {"query": "crewai"},
        "results": [
            {
                "title": "crewAI",
                "link": "https://github.com/crewAIInc/crewAI",
                "snippet": "Framework for orchestrating agents",
                "content": "full page content",
            },
            {"title": "Docs", "link": "https://docs.crewai.com"},
        ],
    }
    with patch(POST, return_value=_response(json_data=body)) as post:
        output = json.loads(tool.run(query="crewai"))

    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == "https://api.search1api.com/search"
    assert kwargs["headers"]["Authorization"] == "Bearer s1-test-key"
    assert kwargs["json"] == {
        "query": "crewai",
        "max_results": 3,
        "crawl_results": 1,
        "search_service": "github",
        "language": "en",
        "time_range": "week",
        "include_sites": ["github.com"],
    }
    assert output == {
        "query": "crewai",
        "results": [
            {
                "title": "crewAI",
                "url": "https://github.com/crewAIInc/crewAI",
                "snippet": "Framework for orchestrating agents",
                "content": "full ...",
            },
            {"title": "Docs", "url": "https://docs.crewai.com", "snippet": ""},
        ],
    }


def test_search_omits_unset_options():
    with patch(POST, return_value=_response(json_data={"results": []})) as post:
        output = json.loads(Search1APISearchTool().run(query="q"))

    assert post.call_args.kwargs["json"] == {
        "query": "q",
        "max_results": 5,
        "crawl_results": 0,
    }
    assert output == {"query": "q", "results": []}


def test_news_uses_news_endpoint_and_keeps_publication_date():
    body = {
        "results": [
            {
                "title": "Release",
                "link": "https://example.com/news",
                "snippet": "New version",
                "published_date": "2026-10-01T00:00:00Z",
            }
        ]
    }
    tool = Search1APINewsTool(search_service="hackernews")
    with patch(POST, return_value=_response(json_data=body)) as post:
        output = json.loads(tool.run(query="crewai release"))

    assert post.call_args.args[0] == "https://api.search1api.com/news"
    assert post.call_args.kwargs["json"]["search_service"] == "hackernews"
    assert output["results"][0]["published_date"] == "2026-10-01T00:00:00Z"


def test_crawl_returns_page_content():
    body = {
        "crawlParameters": {"url": "https://example.com"},
        "results": {
            "title": "Example Domain",
            "link": "https://example.com",
            "content": "This domain is for use in documentation examples.",
        },
    }
    with patch(POST, return_value=_response(json_data=body)) as post:
        output = json.loads(
            Search1APICrawlTool(max_content_length=11).run(url="https://example.com")
        )

    assert post.call_args.args[0] == "https://api.search1api.com/crawl"
    assert post.call_args.kwargs["json"] == {"url": "https://example.com"}
    assert output == {
        "title": "Example Domain",
        "url": "https://example.com",
        "content": "This domain...",
    }


def test_crawl_rejects_malformed_response():
    with patch(POST, return_value=_response(json_data={"results": None})):
        with pytest.raises(RuntimeError, match="malformed"):
            Search1APICrawlTool().run(url="https://example.com")


@pytest.mark.parametrize(
    "body",
    [{}, {"results": None}, {"results": ["not an object"]}, ["not", "a", "dict"]],
)
def test_search_rejects_malformed_results(body):
    with patch(POST, return_value=_response(json_data=body)):
        with pytest.raises(RuntimeError, match="malformed"):
            Search1APISearchTool().run(query="q")


def test_http_error_surfaces_api_message():
    body = {
        "error": "Unauthorized: Invalid bearer credential",
        "message": "Unauthorized: Invalid bearer credential",
        "ok": False,
    }
    with patch(POST, return_value=_response(401, json_data=body)):
        with pytest.raises(RuntimeError) as exc:
            Search1APISearchTool().run(query="q")

    assert "HTTP 401" in str(exc.value)
    assert "Invalid bearer credential" in str(exc.value)


def test_http_error_without_json_body_uses_text():
    with patch(POST, return_value=_response(502, text="Bad Gateway")):
        with pytest.raises(RuntimeError, match="HTTP 502: Bad Gateway"):
            Search1APINewsTool().run(query="q")


def test_network_failure_is_reported():
    with patch(POST, side_effect=requests.ConnectionError("connection refused")):
        with pytest.raises(RuntimeError, match="connection refused"):
            Search1APISearchTool().run(query="q")


def test_missing_api_key_raises_before_request():
    with patch.dict(os.environ, {}, clear=True), patch(POST) as post:
        with pytest.raises(ValueError, match="SEARCH1API_API_KEY"):
            Search1APISearchTool().run(query="q")
    post.assert_not_called()


def test_explicit_api_key_overrides_environment():
    tool = Search1APISearchTool(api_key="explicit-key")
    with patch(POST, return_value=_response(json_data={"results": []})) as post:
        tool.run(query="q")

    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer explicit-key"


def test_api_key_is_not_exposed():
    tool = Search1APISearchTool(api_key="secret-key")

    assert "secret-key" not in repr(tool)
    assert "api_key" not in tool.model_dump()


def test_crawl_results_cannot_exceed_max_results():
    with pytest.raises(ValidationError, match="crawl_results"):
        Search1APISearchTool(max_results=2, crawl_results=3)


def test_search_skips_results_without_link():
    body = {
        "results": [
            {"title": "No link", "snippet": "missing"},
            {"title": "Empty link", "link": "  "},
            {"title": "Not a string", "link": 42},
            {"title": "Kept", "link": "https://example.com"},
        ]
    }
    with patch(POST, return_value=_response(json_data=body)):
        output = json.loads(Search1APISearchTool().run(query="q"))

    assert output["results"] == [
        {"title": "Kept", "url": "https://example.com", "snippet": ""}
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_content_limits_must_be_positive(limit):
    with pytest.raises(ValidationError, match="max_content_length"):
        Search1APICrawlTool(max_content_length=limit)
    with pytest.raises(ValidationError, match="max_content_length_per_result"):
        Search1APISearchTool(max_content_length_per_result=limit)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "http://127.0.0.1:8080/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://[::1]/",
        # Userinfo must not hide a loopback host.
        "http://user:pass@127.0.0.1/",
    ],
)
def test_crawl_rejects_unsafe_urls_without_calling_api(url):
    with patch(POST) as post:
        with pytest.raises(ValueError):
            Search1APICrawlTool().run(url=url)
    post.assert_not_called()
