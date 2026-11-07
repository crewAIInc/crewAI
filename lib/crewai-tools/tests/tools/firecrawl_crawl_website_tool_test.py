from unittest.mock import MagicMock, patch

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


def test_crawl_tool_accepts_explicit_none_config():
    """An explicit config=None means no extra crawl options, not a crash."""
    with patch("firecrawl.FirecrawlApp"):
        tool = FirecrawlCrawlWebsiteTool(api_key="test-key", config=None)

    tool._firecrawl = MagicMock()

    tool._run(url="https://example.com")

    tool._firecrawl.crawl.assert_called_once_with(
        url="https://example.com", poll_interval=2
    )
