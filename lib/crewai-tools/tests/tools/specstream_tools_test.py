import json
import os
from unittest.mock import MagicMock, patch

from crewai_tools import SpecStreamExtractTool, SpecStreamSearchTool
import pytest
import requests as requests_lib


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests_lib.Response)
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.text = json.dumps(json_data or {})
    resp.json.return_value = json_data or {}
    return resp


@pytest.fixture(autouse=True)
def _specstream_env():
    with patch.dict(os.environ, {"SPECSTREAM_API_KEY": "test-key"}):
        yield


SEARCH_PATH = "crewai_tools.tools.specstream_search_tool.specstream_search_tool.requests.get"
EXTRACT_PATH = "crewai_tools.tools.specstream_extract_tool.specstream_extract_tool.requests.post"


def test_search_returns_results():
    payload = {
        "data": {
            "sources": [
                {
                    "url": "https://example.com/a",
                    "title": "A",
                    "snippet": "first",
                    "published": "2026-02-25",
                    "site": "example.com",
                },
                {"url": "https://example.com/b", "title": "B", "snippet": "second"},
            ]
        }
    }
    with patch(SEARCH_PATH, return_value=_mock_response(json_data=payload)) as get:
        result = json.loads(SpecStreamSearchTool().run(query="test"))

    assert result == [
        {
            "title": "A",
            "url": "https://example.com/a",
            "snippet": "first",
            "published": "2026-02-25",
        },
        {
            "title": "B",
            "url": "https://example.com/b",
            "snippet": "second",
            "published": None,
        },
    ]
    _, kwargs = get.call_args
    assert get.call_args.args[0] == "https://api.specstream.ai/v1/web-search"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["params"] == {"q": "test", "num": 10}


def test_search_passes_max_results_and_speed():
    with patch(SEARCH_PATH, return_value=_mock_response(json_data={"data": {}})) as get:
        result = SpecStreamSearchTool().run(query="test", max_results=3, speed="fast")

    assert json.loads(result) == []
    assert get.call_args.kwargs["params"] == {"q": "test", "num": 3, "speed": "fast"}


def test_search_raises_on_http_error():
    error = {"error": {"code": "invalid_token", "message": "bad key"}}
    with patch(SEARCH_PATH, return_value=_mock_response(401, error)):
        with pytest.raises(RuntimeError, match="HTTP 401"):
            SpecStreamSearchTool().run(query="test")


def test_search_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="SPECSTREAM_API_KEY"):
            SpecStreamSearchTool().run(query="test")


def test_extract_accepts_a_single_url():
    payload = {
        "data": {
            "results": [
                {
                    "url": "https://example.com",
                    "status": "ok",
                    "title": "Example Domain",
                    "text": "Example text",
                    "read_from": "live",
                }
            ]
        }
    }
    with patch(EXTRACT_PATH, return_value=_mock_response(json_data=payload)) as post:
        result = json.loads(SpecStreamExtractTool().run(urls="https://example.com"))

    assert result == [
        {
            "url": "https://example.com",
            "status": "ok",
            "title": "Example Domain",
            "text": "Example text",
        }
    ]
    assert post.call_args.args[0] == "https://api.specstream.ai/v1/extract"
    assert post.call_args.kwargs["json"] == {
        "urls": ["https://example.com"],
        "format": "markdown",
    }


def test_extract_rejects_more_than_twenty_urls():
    urls = [f"https://example.com/{i}" for i in range(21)]
    with patch(EXTRACT_PATH) as post:
        with pytest.raises(ValueError, match="between 1 and 20"):
            SpecStreamExtractTool().run(urls=urls)
    post.assert_not_called()


def test_extract_raises_on_connection_error():
    with patch(EXTRACT_PATH, side_effect=requests_lib.ConnectionError("down")):
        with pytest.raises(RuntimeError, match="request failed"):
            SpecStreamExtractTool().run(urls=["https://example.com"])
