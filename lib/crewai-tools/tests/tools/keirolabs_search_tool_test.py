import os
from unittest.mock import MagicMock, patch

from crewai_tools.tools.keirolabs_search_tool.keirolabs_search_tool import (
    KeirolabsSearchTool,
)
import pytest
import requests


@pytest.fixture(autouse=True)
def mock_keirolabs_api_key():
    with patch.dict(os.environ, {"KEIROLABS_API_KEY": "test_key"}):
        yield


def _response(payload, status_code=200):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.side_effect = (
        requests.HTTPError(f"{status_code} Error") if status_code >= 400 else None
    )
    return response


@patch("requests.post")
def test_search_sends_query_and_maps_results(mock_post):
    mock_post.return_value = _response(
        {
            "query": "asyncio",
            "results": [
                {
                    "title": "asyncio docs",
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "snippet": "asyncio is a library to write concurrent code",
                    "domain": "docs.python.org",
                    "position": 1,
                    "score": None,
                },
                {"title": "No snippet", "url": "https://example.org/", "snippet": None},
                {"title": "No url", "snippet": "dropped"},
            ],
        }
    )

    results = KeirolabsSearchTool(n_results=5).run(search_query="asyncio")

    assert results == [
        {
            "title": "asyncio docs",
            "link": "https://docs.python.org/3/library/asyncio.html",
            "snippet": "asyncio is a library to write concurrent code",
        },
        {"title": "No snippet", "link": "https://example.org/", "snippet": ""},
    ]
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "https://api.keirolabs.cloud/api/v2/search/lite"
    assert mock_post.call_args.kwargs["json"] == {"query": "asyncio", "maxResults": 5}
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer test_key"


@patch("requests.post")
def test_search_caps_results_at_n_results(mock_post):
    mock_post.return_value = _response(
        {"results": [{"title": str(i), "url": f"https://e.com/{i}"} for i in range(5)]}
    )

    assert len(KeirolabsSearchTool(n_results=2).run(search_query="q")) == 2


@patch("requests.post")
def test_search_raises_on_http_error(mock_post):
    mock_post.return_value = _response({"error": "Invalid API key"}, status_code=401)

    with pytest.raises(requests.HTTPError):
        KeirolabsSearchTool().run(search_query="q")


def test_search_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="KEIROLABS_API_KEY"):
            KeirolabsSearchTool().run(search_query="q")
