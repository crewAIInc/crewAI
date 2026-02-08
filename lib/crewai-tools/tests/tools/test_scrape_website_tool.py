import pytest
from unittest.mock import patch, MagicMock

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PLAYWRIGHT_AVAILABLE = False

from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool

def test_scrape_website_tool_basic():
    """Normal tool performance test without rendering"""
    tool = ScrapeWebsiteTool()
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.text = "<html><body>Hello World</body></html>"
        mock_response.apparent_encoding = 'utf-8'
        mock_get.return_value = mock_response
        
        result = tool._run(website_url="https://example.com")
        assert "Hello World" in result

@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="playwright is not installed")
@patch('playwright.sync_api.sync_playwright')
def test_scrape_website_tool_with_playwright(mock_playwright):
    """Test rendering with PlayWrite (if installed)"""

    mock_browser = mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value
    mock_page = mock_browser.new_context.return_value.new_page.return_value
    mock_page.content.return_value = "<html><body>JS Content</body></html>"
    
    tool = ScrapeWebsiteTool(render_js=True)
    result = tool._run(website_url="https://example.com")
    
    assert "JS Content" in result