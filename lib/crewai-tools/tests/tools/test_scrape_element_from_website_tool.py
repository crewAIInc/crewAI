from unittest.mock import MagicMock, patch

from crewai_tools import ScrapeElementFromWebsiteTool
import pytest


MODULE = "crewai_tools.tools.scrape_element_from_website.scrape_element_from_website"
PAGE = MagicMock(content=b"<html><h1>Title</h1><p class='price'>$10</p></html>")


@pytest.fixture
def safe_get():
    with patch(f"{MODULE}.safe_get", return_value=PAGE) as mocked:
        yield mocked


def test_fixed_url_still_asks_the_agent_for_the_css_element(safe_get):
    tool = ScrapeElementFromWebsiteTool(website_url="https://example.com")

    assert list(tool.args_schema.model_fields) == ["css_element"]
    assert tool.run(css_element="h1") == "Title"
    assert safe_get.call_args.args[0] == "https://example.com"


def test_fixed_css_element_is_kept_without_a_fixed_url(safe_get):
    tool = ScrapeElementFromWebsiteTool(css_element=".price")

    assert tool.css_element == ".price"
    assert list(tool.args_schema.model_fields) == ["website_url"]
    assert tool.run(website_url="https://example.com") == "$10"


def test_both_fixed_needs_no_arguments(safe_get):
    tool = ScrapeElementFromWebsiteTool(
        website_url="https://example.com", css_element=".price"
    )

    assert list(tool.args_schema.model_fields) == []
    assert tool.run() == "$10"


def test_nothing_fixed_asks_for_both(safe_get):
    tool = ScrapeElementFromWebsiteTool()

    assert list(tool.args_schema.model_fields) == ["website_url", "css_element"]
    assert tool.run(website_url="https://example.com", css_element="h1") == "Title"
