from unittest.mock import Mock, patch

from crewai_tools import ScrapeWebsiteTool


def test_scraped_text_keeps_words_whole_across_inline_elements():
    """Inline markup must not split words, and blocks must start new lines."""
    page = Mock()
    page.apparent_encoding = "utf-8"
    page.text = (
        "<html><body><h1>Setup</h1><p>That is un<b>believ</b>able, "
        'see <a href="/docs">the docs</a>.</p><ul><li>one</li><li>two</li></ul>'
        "</body></html>"
    )

    with patch(
        "crewai_tools.tools.scrape_website_tool.scrape_website_tool.safe_get",
        return_value=page,
    ):
        result = ScrapeWebsiteTool().run(website_url="https://example.com")

    assert result.endswith(
        "Setup\nThat is unbelievable, see the docs.\none\ntwo"
    )
