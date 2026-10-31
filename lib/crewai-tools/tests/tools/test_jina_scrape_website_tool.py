from unittest.mock import Mock, patch

from crewai_tools import JinaScrapeWebsiteTool
import pytest
import requests


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )


@pytest.mark.parametrize("fixed_url", [False, True])
def test_scrape_configured_or_run_time_url(fixed_url: bool) -> None:
    url = "https://example.com/article"
    tool = JinaScrapeWebsiteTool(
        website_url=url if fixed_url else None,
        api_key="synthetic-key",
        custom_headers={"X-Target-Selector": "article"},
    )
    response = Mock(text="# Article\nReader content")
    with patch(
        "crewai_tools.tools.jina_scrape_website_tool.jina_scrape_website_tool.requests.get",
        return_value=response,
    ) as get:
        result = tool.run(**({} if fixed_url else {"website_url": url}))

    assert result == "# Article\nReader content"
    get.assert_called_once_with(
        f"https://r.jina.ai/{url}",
        headers={"X-Target-Selector": "article", "Authorization": "Bearer synthetic-key"},
        timeout=15,
    )
    response.raise_for_status.assert_called_once_with()
    assert ("website_url" in tool.args_schema.model_fields) is not fixed_url


def test_unconfigured_tool_still_requires_url() -> None:
    with pytest.raises(ValueError, match="website_url"):
        JinaScrapeWebsiteTool().run()


def test_fixed_url_still_validates_url() -> None:
    with patch(
        "crewai_tools.tools.jina_scrape_website_tool.jina_scrape_website_tool.requests.get"
    ) as get:
        with pytest.raises(ValueError):
            JinaScrapeWebsiteTool(website_url="file:///tmp/example").run()
    get.assert_not_called()


def test_fixed_url_propagates_http_error() -> None:
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("503")
    with patch(
        "crewai_tools.tools.jina_scrape_website_tool.jina_scrape_website_tool.requests.get",
        return_value=response,
    ):
        with pytest.raises(requests.HTTPError, match="503"):
            JinaScrapeWebsiteTool(website_url="https://example.com/article").run()
