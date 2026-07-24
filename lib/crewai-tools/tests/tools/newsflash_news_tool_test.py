from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.newsflash_news_tool.newsflash_news_tool import NewsflashNewsTool


def _mock_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


SEARCH_PAYLOAD = {
    "count": 2,
    "events": [
        {
            "id": 1,
            "canonical_title": "Fed cuts rates by 25bps",
            "summary": "The Federal Reserve cut its benchmark rate.",
            "category": "tradfi",
            "first_seen_at": "2026-07-24T10:00:00.000Z",
            "last_seen_at": "2026-07-24T12:00:00.000Z",
            "sources": ["reuters", "apnews", "bloomberg"],
            "corroboration": 3,
            "confidence": 1.0,
        },
        {
            "id": 2,
            "canonical_title": "Obscure single-source rumor",
            "summary": "Only one outlet reported this.",
            "category": "tradfi",
            "first_seen_at": "2026-07-24T11:00:00.000Z",
            "last_seen_at": "2026-07-24T11:00:00.000Z",
            "sources": ["someblog"],
            "corroboration": 1,
            "confidence": 0.3333333333333333,
        },
    ],
}

DETAIL_PAYLOADS = {
    "1": {
        "id": 1,
        "articles": [
            {
                "source": "reuters",
                "title": "Fed cuts rates by 25bps",
                "url": "https://reuters.com/fed-cut",
                "published_at": "2026-07-24T10:00:00.000Z",
            }
        ],
    },
    "2": {
        "id": 2,
        "articles": [
            {
                "source": "someblog",
                "title": "Obscure single-source rumor",
                "url": "https://someblog.example/rumor",
                "published_at": "2026-07-24T11:00:00.000Z",
            }
        ],
    },
}


def _routed_get(url, **kwargs):
    if url.endswith("/events"):
        return _mock_response(SEARCH_PAYLOAD)
    event_id = url.rsplit("/", 1)[-1]
    return _mock_response(DETAIL_PAYLOADS[event_id])


@pytest.fixture
def newsflash_tool():
    return NewsflashNewsTool()


def test_newsflash_tool_initialization():
    tool = NewsflashNewsTool()
    assert tool.base_url == "https://newsflash.sh/api"
    assert tool.name == "Newsflash News Search"
    assert any(var.name == "NEWSFLASH_API_KEY" for var in tool.env_vars)
    # Keyless usage is supported, so the key must not be required
    assert all(
        not var.required for var in tool.env_vars if var.name == "NEWSFLASH_API_KEY"
    )


@patch("requests.get")
def test_newsflash_tool_search(mock_get, newsflash_tool):
    mock_get.side_effect = _routed_get

    result = newsflash_tool.run(query="fed rates")

    assert "Fed cuts rates by 25bps" in result
    assert "3 source(s)" in result
    assert "reuters, apnews, bloomberg" in result
    assert "https://reuters.com/fed-cut" in result
    # Low-confidence event is included by default (min_confidence=0.0)
    assert "Obscure single-source rumor" in result


@patch("requests.get")
def test_newsflash_tool_min_confidence_filters_uncorroborated(mock_get, newsflash_tool):
    mock_get.side_effect = _routed_get

    result = newsflash_tool.run(query="fed rates", min_confidence=0.6)

    assert "Fed cuts rates by 25bps" in result
    assert "Obscure single-source rumor" not in result
    # Details must only be fetched for events that pass the filter:
    # 1 search call + 1 detail call
    detail_calls = [
        call for call in mock_get.call_args_list if "/events/" in call.args[0]
    ]
    assert len(detail_calls) == 1
    assert detail_calls[0].args[0].endswith("/events/1")


@patch("requests.get")
def test_newsflash_tool_limit_caps_detail_fetches(mock_get, newsflash_tool):
    mock_get.side_effect = _routed_get

    result = newsflash_tool.run(query="fed rates", limit=1)

    assert "Fed cuts rates by 25bps" in result
    assert "Obscure single-source rumor" not in result
    detail_calls = [
        call for call in mock_get.call_args_list if "/events/" in call.args[0]
    ]
    assert len(detail_calls) == 1


@patch("requests.get")
def test_newsflash_tool_query_params(mock_get, newsflash_tool):
    mock_get.side_effect = _routed_get

    newsflash_tool.run(query="bitcoin", semantic=False, category="crypto", limit=5)

    search_call = mock_get.call_args_list[0]
    params = search_call.kwargs["params"]
    assert params["q"] == "bitcoin"
    assert params["semantic"] == "0"
    assert params["category"] == "crypto"
    assert params["limit"] == 5


@patch("requests.get")
def test_newsflash_tool_sends_bearer_key_when_set(
    mock_get, newsflash_tool, monkeypatch
):
    monkeypatch.setenv("NEWSFLASH_API_KEY", "nf_test_key")
    mock_get.side_effect = _routed_get

    newsflash_tool.run(query="fed rates", limit=1)

    headers = mock_get.call_args_list[0].kwargs["headers"]
    assert headers["Authorization"] == "Bearer nf_test_key"


@patch("requests.get")
def test_newsflash_tool_keyless_sends_no_auth_header(
    mock_get, newsflash_tool, monkeypatch
):
    monkeypatch.delenv("NEWSFLASH_API_KEY", raising=False)
    mock_get.side_effect = _routed_get

    newsflash_tool.run(query="fed rates", limit=1)

    headers = mock_get.call_args_list[0].kwargs["headers"]
    assert "Authorization" not in headers


@patch("requests.get")
def test_newsflash_tool_no_results(mock_get, newsflash_tool):
    mock_get.return_value = _mock_response(
        {
            "count": 0,
            "events": [],
            "window": {"note": "the test tier can query the last 24 hours"},
        }
    )

    result = newsflash_tool.run(query="nonexistent topic", min_confidence=0.9)

    assert "No corroborated news events found" in result
    assert "nonexistent topic" in result
