from typing import Any
from unittest.mock import Mock, patch

from crewai_tools import ScrapeElementFromWebsiteTool, ScrapeWebsiteTool
import pytest


@pytest.mark.parametrize("tool_cls", [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool])
@pytest.mark.parametrize("fixed_url", [False, True])
def test_configured_cookie_reaches_scraper_request(
    monkeypatch: pytest.MonkeyPatch,
    tool_cls: type[ScrapeWebsiteTool | ScrapeElementFromWebsiteTool],
    fixed_url: bool,
) -> None:
    monkeypatch.setenv("TEST_SCRAPER_COOKIE", "synthetic-session")
    kwargs: dict[str, Any] = {"cookies": {"name": "session", "value": "TEST_SCRAPER_COOKIE"}}
    run_kwargs = {}
    if fixed_url:
        kwargs["website_url"] = "https://example.com/article"
        kwargs["css_element"] = ".member"
    else:
        run_kwargs = {"website_url": "https://example.com/article", "css_element": ".member"}
    tool = tool_cls(**kwargs)

    def fetch(url: str, **request_kwargs: Any) -> Mock:
        text = (
            '<p class="member">Member content</p>'
            if request_kwargs.get("cookies") == {"session": "synthetic-session"}
            else "Login required"
        )
        return Mock(text=text, content=text.encode(), apparent_encoding="utf-8")

    with patch(f"{tool_cls.__module__}.safe_get", side_effect=fetch):
        result = tool.run(**run_kwargs)

    assert "Member content" in result


@pytest.mark.parametrize("tool_cls", [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool])
def test_scraper_without_cookie_configuration(
    tool_cls: type[ScrapeWebsiteTool | ScrapeElementFromWebsiteTool],
) -> None:
    page = Mock(text='<p class="public">Public content</p>', content=b'<p class="public">Public content</p>', apparent_encoding="utf-8")
    with patch(f"{tool_cls.__module__}.safe_get", return_value=page) as fetch:
        result = tool_cls().run(website_url="https://example.com/article", css_element=".public")

    assert "Public content" in result
    assert fetch.call_args.kwargs["cookies"] == {}
