from unittest.mock import patch

from crewai_tools.tools.perplexity_tools.perplexity_search_tool import (
    PerplexitySearchTool,
)


def test_requires_env_var(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    tool = PerplexitySearchTool()
    result = tool.run(query="test")
    assert "PERPLEXITY_API_KEY" in result


def test_accepts_pplx_api_key_alias(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("PPLX_API_KEY", "test-alias")

    with patch(
        "crewai_tools.tools.perplexity_tools.perplexity_search_tool.requests.post"
    ) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "x", "results": []}

        tool = PerplexitySearchTool()
        tool.run(query="hello")

        assert mock_post.called
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-alias"


@patch("crewai_tools.tools.perplexity_tools.perplexity_search_tool.requests.post")
def test_happy_path_formats_results(mock_post, monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test")

    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "id": "search_123",
        "results": [
            {
                "title": "History of the United Nations",
                "url": "https://www.un.org/en/about-us/history-of-the-un",
                "snippet": "The United Nations officially began on 24 October 1945.",
                "date": "1945-10-24",
            },
            {
                "title": "UN Charter",
                "url": "https://www.un.org/en/about-us/un-charter",
                "snippet": "Founding document of the UN.",
            },
        ],
    }

    tool = PerplexitySearchTool()
    result = tool.run(
        query="When was the UN established?",
        max_results=2,
        search_recency_filter="year",
    )

    assert "1. History of the United Nations" in result
    assert "https://www.un.org/en/about-us/history-of-the-un" in result
    assert "Date: 1945-10-24" in result
    assert "2. UN Charter" in result

    payload = mock_post.call_args.kwargs["json"]
    assert payload["query"] == "When was the UN established?"
    assert payload["max_results"] == 2
    assert payload["search_recency_filter"] == "year"
    assert "search_domain_filter" not in payload


@patch("crewai_tools.tools.perplexity_tools.perplexity_search_tool.requests.post")
def test_passes_domain_filter(mock_post, monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test")
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"results": []}

    tool = PerplexitySearchTool()
    tool.run(
        query="legal docs",
        search_domain_filter=["nytimes.com", "-pinterest.com"],
    )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["search_domain_filter"] == ["nytimes.com", "-pinterest.com"]


@patch("crewai_tools.tools.perplexity_tools.perplexity_search_tool.requests.post")
def test_http_error_returned_as_string(mock_post, monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test")
    mock_post.return_value.status_code = 401
    mock_post.return_value.text = "unauthorized"

    tool = PerplexitySearchTool()
    result = tool.run(query="hello")
    assert "401" in result
    assert "Perplexity Search API error" in result


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "from-env")

    with patch(
        "crewai_tools.tools.perplexity_tools.perplexity_search_tool.requests.post"
    ) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"results": []}

        tool = PerplexitySearchTool(api_key="explicit")
        tool.run(query="hi")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer explicit"
