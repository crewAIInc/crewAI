from typing import Any
from unittest.mock import Mock, patch

from crewai_tools import ScrapeElementFromWebsiteTool, ScrapeWebsiteTool
import pytest
import requests


@pytest.mark.parametrize("tool_cls", [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool])
def test_configured_cookie_reaches_fixed_scraper_request(
    monkeypatch: pytest.MonkeyPatch,
    tool_cls: type[ScrapeWebsiteTool | ScrapeElementFromWebsiteTool],
) -> None:
    monkeypatch.setenv("TEST_SCRAPER_COOKIE", "synthetic-session")
    tool = tool_cls(
        website_url="https://example.com/article",
        css_element=".member",
        cookies={"name": "session", "value": "TEST_SCRAPER_COOKIE"},
    )

    def fetch(url: str, **request_kwargs: Any) -> Mock:
        text = (
            '<p class="member">Member content</p>'
            if request_kwargs.get("cookies") == {"session": "synthetic-session"}
            else "Login required"
        )
        return Mock(text=text, content=text.encode(), apparent_encoding="utf-8")

    with patch(f"{tool_cls.__module__}.safe_get", side_effect=fetch):
        result = tool.run()

    assert "Member content" in result


@pytest.mark.parametrize("tool_cls", [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool])
def test_scraper_without_cookie_configuration(
    tool_cls: type[ScrapeWebsiteTool | ScrapeElementFromWebsiteTool],
) -> None:
    page = Mock(
        text='<p class="public">Public content</p>',
        content=b'<p class="public">Public content</p>',
        apparent_encoding="utf-8",
    )
    with patch(f"{tool_cls.__module__}.safe_get", return_value=page) as fetch:
        result = tool_cls().run(
            website_url="https://example.com/article", css_element=".public"
        )

    assert "Public content" in result
    assert fetch.call_args.kwargs["cookies"] == {}


@pytest.mark.parametrize("tool_cls", [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool])
@pytest.mark.parametrize("website_url", [None, ""])
def test_cookie_configuration_requires_fixed_url(
    tool_cls: type[ScrapeWebsiteTool | ScrapeElementFromWebsiteTool],
    website_url: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_SCRAPER_COOKIE", "synthetic-session")
    with patch(f"{tool_cls.__module__}.safe_get") as fetch:
        with pytest.raises(ValueError, match="cookies require a fixed website_url"):
            tool_cls(
                website_url=website_url,
                cookies={"name": "session", "value": "TEST_SCRAPER_COOKIE"},
            )
    fetch.assert_not_called()


@pytest.mark.parametrize("tool_cls", [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "runtime_url", ["https://other.example/article", "http://127.0.0.1/"]
)
def test_authenticated_scraper_cannot_be_retargeted(
    tool_cls: type[ScrapeWebsiteTool | ScrapeElementFromWebsiteTool],
    structured: bool,
    runtime_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_SCRAPER_COOKIE", "synthetic-session")
    pinned_url = "https://example.com/article"
    tool = tool_cls(
        website_url=pinned_url,
        css_element=".member",
        cookies={"name": "session", "value": "TEST_SCRAPER_COOKIE"},
    )
    page = Mock(
        text='<p class="member">Member content</p>',
        content=b'<p class="member">Member content</p>',
        apparent_encoding="utf-8",
    )
    with patch(f"{tool_cls.__module__}.safe_get", return_value=page) as fetch:
        if structured:
            result = tool.to_structured_tool().invoke({"website_url": runtime_url})
        else:
            result = tool.run(website_url=runtime_url)

    assert "Member content" in result
    assert fetch.call_args.args[0] == pinned_url
    assert fetch.call_args.kwargs["cookies"] == {"session": "synthetic-session"}


@pytest.mark.parametrize("tool_cls", [ScrapeWebsiteTool, ScrapeElementFromWebsiteTool])
@pytest.mark.parametrize("same_origin", [False, True])
def test_scraper_redirects_only_keep_same_origin_cookies(
    tool_cls: type[ScrapeWebsiteTool | ScrapeElementFromWebsiteTool],
    same_origin: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_SCRAPER_COOKIE", "synthetic-session")
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    pinned_url = "https://example.com/article"
    target = "https://example.com/member" if same_origin else "https://other.example/"
    tool = tool_cls(
        website_url=pinned_url,
        css_element=".member",
        cookies={"name": "session", "value": "TEST_SCRAPER_COOKIE"},
    )
    redirect = requests.Response()
    redirect.status_code = 302
    redirect.url = pinned_url
    redirect.headers["Location"] = target
    redirect._content = b""
    page = requests.Response()
    page.status_code = 200
    page.url = target
    page._content = b'<p class="member">Member content</p>'
    with patch(
        "crewai_tools.security.safe_requests._raw_get", side_effect=[redirect, page]
    ) as fetch:
        result = tool.run()

    assert "Member content" in result
    assert fetch.call_args_list[0].args[0] == pinned_url
    assert fetch.call_args_list[0].kwargs["cookies"] == {"session": "synthetic-session"}
    assert fetch.call_args_list[1].args[0] == target
    assert fetch.call_args_list[1].kwargs.get("cookies", {}) == (
        {"session": "synthetic-session"} if same_origin else {}
    )
