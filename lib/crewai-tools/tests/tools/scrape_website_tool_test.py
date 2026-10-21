from unittest.mock import Mock, patch

from crewai_tools import ScrapeWebsiteTool


@patch("crewai_tools.tools.scrape_website_tool.scrape_website_tool.safe_get")
def test_scraped_text_keeps_words_whole_across_inline_elements(mock_get):
    response = Mock()
    response.text = (
        "<html><body><p>That is un<b>believ</b>able, see <a href='/docs'>the docs</a>.</p>"
        "<p>Next<br>line</p></body></html>"
    )
    response.apparent_encoding = "utf-8"
    mock_get.return_value = response

    result = ScrapeWebsiteTool(website_url="https://example.com").run()

    lines = [line for line in result.splitlines() if line]
    assert lines[1:] == ["That is unbelievable, see the docs.", "Next", "line"]
