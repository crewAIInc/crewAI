import pytest

from crewai_tools.tools.firecrawl_scrape_website_tool.firecrawl_scrape_website_tool import (
    FirecrawlScrapeWebsiteTool,
)

@pytest.mark.vcr()
def test_firecrawl_scrape_tool_integration():
    tool = FirecrawlScrapeWebsiteTool()
    result = tool.run(url="https://firecrawl.dev")

    assert result is not None
    assert hasattr(result, 'markdown')
    assert len(result.markdown) > 0
    assert "Firecrawl" in result.markdown or "firecrawl" in result.markdown.lower()
