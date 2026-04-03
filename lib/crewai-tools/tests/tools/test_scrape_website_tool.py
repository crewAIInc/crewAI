import pytest
from unittest.mock import patch, MagicMock
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool

try:
    from playwright.sync_api import sync_playwright  # noqa: F401
    PLAYWRIGHT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PLAYWRIGHT_AVAILABLE = False

def test_scrape_website_tool_basic():
    """Test basic scraping without JS rendering."""
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
    """Test scraping with JS rendering using mocked playwright."""
    # Setup mock chain
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_browser = MagicMock()
    
    mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page
    mock_page.content.return_value = "<html><body>JS Content</body></html>"
    
    tool = ScrapeWebsiteTool(render_js=True)
    result = tool._run(website_url="https://example.com")
    
    assert "JS Content" in result