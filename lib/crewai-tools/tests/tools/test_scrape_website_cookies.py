from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.scrape_element_from_website.scrape_element_from_website import (
    ScrapeElementFromWebsiteTool,
)
from crewai_tools.tools.scrape_website_tool import (
    scrape_website_tool as scrape_website_module,
)
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import (
    ScrapeWebsiteTool,
)

COOKIES = {"sessionid": "abc123"}


def test_scrape_website_accepts_cookie_mapping() -> None:
    """The documented ``dict[str, str]`` form must not raise."""
    tool = ScrapeWebsiteTool(website_url="https://example.com", cookies=COOKIES)

    assert tool.cookies == COOKIES


def test_scrape_website_ignores_cookies_without_pinned_url() -> None:
    """Cookies are origin-scoped, so an unpinned tool must not keep them.

    Without a pinned ``website_url`` the agent can name any origin at runtime;
    attaching a developer-supplied cookie jar to that request would leak the
    credentials to a model-chosen host.
    """
    tool = ScrapeWebsiteTool(cookies=COOKIES)

    assert tool.cookies is None


def test_scrape_element_accepts_cookie_mapping() -> None:
    tool = ScrapeElementFromWebsiteTool(
        website_url="https://example.com",
        css_element=".main",
        cookies=COOKIES,
    )

    assert tool.cookies == COOKIES


def test_scrape_website_sends_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    page = MagicMock()
    page.text = "<html><body>hello</body></html>"
    page.apparent_encoding = "utf-8"
    safe_get = MagicMock(return_value=page)
    monkeypatch.setattr(scrape_website_module, "safe_get", safe_get)

    tool = ScrapeWebsiteTool(website_url="https://example.com", cookies=COOKIES)
    tool.run()

    assert safe_get.call_args.kwargs["cookies"] == COOKIES
