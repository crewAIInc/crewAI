import pytest

from crewai_tools.tools.firecrawl_crawl_website_tool.firecrawl_crawl_website_tool import (
    FirecrawlCrawlWebsiteTool,
)

@pytest.mark.vcr()
def test_firecrawl_crawl_tool_integration():
    tool = FirecrawlCrawlWebsiteTool(config={
        "limit": 2,
        "max_discovery_depth": 1,
        "scrape_options": {"formats": ["markdown"]}
    })
    result = tool.run(url="https://firecrawl.dev")

    assert result is not None
    assert hasattr(result, 'status')
    assert result.status in ["completed", "scraping"]


from unittest.mock import patch


def test_crawl_tool_accepts_explicit_none_config():
    """config=None means "no extra Firecrawl options"; it must not raise TypeError."""
    with patch("firecrawl.FirecrawlApp") as app, patch(
        "crewai_tools.tools.firecrawl_crawl_website_tool.firecrawl_crawl_website_tool.validate_url"
    ) as validate:
        app.return_value.crawl.return_value = {"status": "completed"}
        validate.return_value = "https://example.com"
        tool = FirecrawlCrawlWebsiteTool(api_key="test", config=None)
        result = tool._run(url="https://example.com")

    app.return_value.crawl.assert_called_once_with(
        url="https://example.com", poll_interval=2
    )
    assert result == {"status": "completed"}

