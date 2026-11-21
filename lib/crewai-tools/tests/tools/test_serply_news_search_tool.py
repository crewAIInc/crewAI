import json
from unittest.mock import patch

from crewai_tools import SerplyNewsSearchTool
import pytest
import requests


@pytest.mark.parametrize("location", [None, "/article", "https://publisher.example/article"])
def test_news_search_returns_final_response_url(
    monkeypatch: pytest.MonkeyPatch, location: str | None
) -> None:
    monkeypatch.setenv("SERPLY_API_KEY", "test-key")
    search_response = requests.Response()
    search_response.status_code = 200
    search_response._content = json.dumps(
        {
            "entries": [
                {
                    "title": "News title",
                    "link": "https://publisher.example/article"
                    if location is None
                    else "https://publisher.example/start",
                    "source": {"title": "Publisher"},
                    "published": "2026-10-01",
                }
            ]
        }
    ).encode()
    article_response = requests.Response()
    article_response.status_code = 200
    article_response.url = "https://publisher.example/article"
    if location is not None:
        redirect = requests.Response()
        redirect.status_code = 302
        redirect.headers["Location"] = location
        article_response.history = [redirect]

    with (
        patch("requests.request", return_value=search_response),
        patch("requests.get", return_value=article_response),
    ):
        result = SerplyNewsSearchTool().run(search_query="agent news")

    assert "Title: News title" in result
    assert "Link: https://publisher.example/article" in result
    assert "Source: Publisher" in result
