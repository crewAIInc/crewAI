"""Regression tests for issue #9007: the documented cookies mapping must be
stored as given instead of being treated as an {"name", "value"} env-var pair."""

import pytest

from crewai_tools.tools.scrape_element_from_website.scrape_element_from_website import (
    ScrapeElementFromWebsiteTool,
)
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool


@pytest.mark.parametrize(
    "tool_cls",
    [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool],
)
def test_documented_cookies_mapping_is_stored_verbatim(tool_cls):
    """cookies={"sessionid": "abc123"} must not raise KeyError and must be kept."""
    tool = tool_cls(
        website_url="https://example.com",
        cookies={"sessionid": "abc123"},
    )
    assert tool.cookies == {"sessionid": "abc123"}


@pytest.mark.parametrize(
    "tool_cls",
    [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool],
)
def test_no_cookies_keeps_default(tool_cls):
    tool = tool_cls(website_url="https://example.com")
    assert tool.cookies is None


def test_runtime_url_tool_accepts_cookies_argument_without_pinned_url():
    tool = ScrapeWebsiteTool()
    assert tool.website_url is None
    assert tool.cookies is None
