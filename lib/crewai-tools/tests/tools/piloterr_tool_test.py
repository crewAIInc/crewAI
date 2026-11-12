import json
import os
from unittest.mock import MagicMock, patch

from crewai_tools.tools.piloterr_tool.piloterr_tool import (
    PiloterrGoogleSearchTool,
    PiloterrScrapeWebsiteTool,
)
import pytest
import requests


@pytest.fixture(autouse=True)
def piloterr_api_key():
    with patch.dict(os.environ, {"PILOTERR_API_KEY": "test_key"}):
        yield


def _response(status=200, body=None, text=None):
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.ok = status < 400
    if isinstance(body, Exception):
        response.json.side_effect = body
    else:
        response.json.return_value = body
    response.text = text if text is not None else json.dumps(body)
    return response


PAGE = (
    "<html><head><title>Example Domain</title><style>p{color:red}</style></head>"
    "<body><script>track()</script><h1>Example Domain</h1>"
    "<p>For use in   documentation examples.</p><ul><li>One</li><li>Two &amp; three</li></ul></body></html>"
)


@patch("requests.get")
def test_google_search(mock_get):
    rows = [
        {"position": 1, "title": "CrewAI", "link": "https://www.crewai.com", "snippet": "Agents."},
        "not a result",
        {"position": 2, "title": "Docs", "link": "https://docs.crewai.com", "snippet": "Docs."},
    ]
    mock_get.return_value = _response(body={"organic_results": rows})

    tool = PiloterrGoogleSearchTool(n_results=5, country="FR", language="fr", location="Paris")
    result = json.loads(tool.run(search_query=" crewai "))

    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.piloterr.com/v2/google/search"
    assert kwargs["params"] == {"query": "crewai", "num": 5, "gl": "fr", "hl": "fr", "location": "Paris"}
    assert kwargs["headers"]["x-api-key"] == "test_key"
    assert result == [
        {"title": "CrewAI", "link": "https://www.crewai.com", "snippet": "Agents.", "position": 1},
        {"title": "Docs", "link": "https://docs.crewai.com", "snippet": "Docs.", "position": 2},
    ]


@patch("requests.get")
def test_google_search_limit_and_no_results(mock_get):
    rows = [{"title": f"R{i}", "link": f"https://r{i}.example", "snippet": ""} for i in range(5)]
    mock_get.return_value = _response(body={"organic_results": rows})
    assert len(json.loads(PiloterrGoogleSearchTool(n_results=2).run(search_query="q"))) == 2

    mock_get.return_value = _response(body={"organic_results": []})
    assert PiloterrGoogleSearchTool().run(search_query="q") == "No results for: q"


@patch("requests.get")
def test_scrape_with_the_crawler(mock_get):
    mock_get.return_value = _response(body=PAGE)

    result = PiloterrScrapeWebsiteTool().run(website_url=" https://example.com ")

    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.piloterr.com/v2/website/crawler"
    assert kwargs["params"] == {"query": "https://example.com"}
    assert result == "# Example Domain\n\nExample Domain\nFor use in documentation examples.\nOne\nTwo & three"


@patch("requests.get")
def test_scrape_rendering_wait_and_raw_html(mock_get):
    mock_get.return_value = _response(body=ValueError("not JSON"), text=PAGE)

    result = PiloterrScrapeWebsiteTool(engine="rendering", wait_in_seconds=3).run(website_url="https://example.com")

    assert mock_get.call_args.args[0] == "https://api.piloterr.com/v2/website/rendering"
    assert mock_get.call_args.kwargs["params"] == {"query": "https://example.com", "wait_in_seconds": 3}
    assert result.startswith("# Example Domain")


@patch("requests.get")
def test_scrape_wait_is_only_for_rendering_and_text_is_cut(mock_get):
    mock_get.return_value = _response(body=PAGE)

    tool = PiloterrScrapeWebsiteTool(engine="webunlocker", wait_in_seconds=3, max_characters=7)
    result = tool.run(website_url="https://example.com")

    assert mock_get.call_args.args[0] == "https://api.piloterr.com/v2/website/webunlocker"
    assert mock_get.call_args.kwargs["params"] == {"query": "https://example.com"}
    assert result == "# Example Domain\n\nExample"


@pytest.mark.parametrize("url", ["", "example.com", "file:///etc/passwd", "ftp://example.com/x"])
@patch("requests.get")
def test_scrape_rejects_bad_urls_without_a_request(mock_get, url):
    assert "absolute http(s) URL" in PiloterrScrapeWebsiteTool().run(website_url=url)
    mock_get.assert_not_called()


@patch("requests.get")
def test_scrape_unexpected_body(mock_get):
    mock_get.return_value = _response(body={"html": "x"})
    assert "expected the page HTML" in PiloterrScrapeWebsiteTool().run(website_url="https://example.com")


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": "Invalid key test_key"}, "Piloterr rejected the API key (HTTP 401): Invalid key ***"),
        (401, {"error": "Rate limit exceeded"}, "Piloterr rate limit exceeded (HTTP 401): Rate limit exceeded"),
        (402, {"error": {"message": "No credits"}}, "Piloterr account is out of credits (HTTP 402): No credits"),
        (403, {"error": "Domain not allowed"}, "Piloterr API error (HTTP 403): Domain not allowed"),
    ],
)
@patch("requests.get")
def test_api_errors_are_returned_without_the_key(mock_get, status, body, expected):
    mock_get.return_value = _response(status=status, body=body)

    assert PiloterrGoogleSearchTool().run(search_query="q") == expected
    assert PiloterrScrapeWebsiteTool().run(website_url="https://example.com") == expected


@patch("requests.get")
def test_network_errors(mock_get):
    mock_get.side_effect = requests.Timeout()
    assert PiloterrScrapeWebsiteTool().run(website_url="https://a.b") == "Piloterr did not answer within 120 seconds."

    mock_get.side_effect = requests.ConnectionError("failed with test_key")
    assert PiloterrGoogleSearchTool().run(search_query="q") == "Could not reach Piloterr: failed with ***"


@patch("requests.get")
def test_missing_key_and_empty_query_send_no_request(mock_get):
    assert PiloterrGoogleSearchTool().run(search_query=" ") == "The search query is empty."
    with patch.dict(os.environ, {"PILOTERR_API_KEY": ""}):
        assert "PILOTERR_API_KEY is not set" in PiloterrGoogleSearchTool().run(search_query="q")
        assert "PILOTERR_API_KEY is not set" in PiloterrScrapeWebsiteTool().run(website_url="https://a.b")
    mock_get.assert_not_called()


def test_declare_their_env_var():
    for tool in (PiloterrGoogleSearchTool(), PiloterrScrapeWebsiteTool()):
        assert [env.name for env in tool.env_vars] == ["PILOTERR_API_KEY"]
