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
