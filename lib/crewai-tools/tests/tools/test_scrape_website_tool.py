import pytest
from unittest.mock import patch, MagicMock
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool

def test_scrape_website_tool_render_js_logic():
    """JavaScript rendering performance validation test with playwright"""
    tool = ScrapeWebsiteTool(website_url="https://example.com", render_js=True)
    
    with patch("playwright.sync_api.sync_playwright") as mock_playwright:
        # Simulate Playwright structure
        mock_context = mock_playwright.return_value.__enter__.return_value
        mock_browser = mock_context.chromium.launch.return_value
        mock_page = mock_browser.new_page.return_value
        mock_page.content.return_value = "<html><body>JS Content</body></html>"
        
        result = tool._run()
        
        assert "JS Content" in result
        mock_playwright.assert_called_once()

def test_scrape_website_tool_default_behavior():
    """Test that there is no change to the old behavior in the default state"""
    tool = ScrapeWebsiteTool(website_url="https://example.com")
    
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = "Normal Content"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = tool._run()
        
        assert "Normal Content" in result
        # We make sure that it doesn't go to the playwrite in normal mode
        mock_get.assert_called_once()
