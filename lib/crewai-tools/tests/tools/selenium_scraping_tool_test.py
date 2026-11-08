import os
import socket
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup
from crewai_tools.tools.selenium_scraping_tool.selenium_scraping_tool import (
    FixedSeleniumScrapingToolSchema,
    SeleniumScrapingTool,
    SeleniumScrapingToolSchema,
)
from pydantic import ValidationError
from selenium.webdriver.chrome.options import Options


@pytest.fixture(autouse=True)
def public_example_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve example.com to a public IP so URL validation is deterministic.

    Mirrors the fixture in tests/rag/conftest.py: validate_url resolves DNS,
    and the real example.com answer depends on the host resolver.
    """
    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        if host == "example.com":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("93.184.216.34", port),
                )
            ]
        return original_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def mock_driver_with_html(html_content):
    driver = MagicMock()
    mock_element = MagicMock()
    mock_element.get_attribute.return_value = html_content
    bs = BeautifulSoup(html_content, "html.parser")
    mock_element.text = bs.get_text()

    driver.find_elements.return_value = [mock_element]
    driver.find_element.return_value = mock_element

    return driver


def initialize_tool_with(mock_driver):
    tool = SeleniumScrapingTool(driver=mock_driver)
    return tool


@patch("selenium.webdriver.Chrome")
def test_tool_initialization(mocked_chrome):
    temp_dir = tempfile.mkdtemp()
    mocked_chrome.return_value = MagicMock()

    tool = SeleniumScrapingTool()

    assert tool.website_url is None
    assert tool.css_element is None
    assert tool.cookie is None
    assert tool.wait_time == 3
    assert tool.return_html is False

    try:
        os.rmdir(temp_dir)
    except:
        pass


@patch("selenium.webdriver.Chrome")
def test_tool_initialization_with_options(mocked_chrome):
    mocked_chrome.return_value = MagicMock()

    options = Options()
    options.add_argument("--disable-gpu")

    SeleniumScrapingTool(options=options)

    mocked_chrome.assert_called_once_with(options=options)


@patch("selenium.webdriver.Chrome")
def test_scrape_without_css_selector(_mocked_chrome_driver):
    html_content = "<html><body><div>test content</div></body></html>"
    mock_driver = mock_driver_with_html(html_content)
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="https://example.com")

    assert "test content" in result
    mock_driver.get.assert_called_once_with("https://example.com")
    mock_driver.find_element.assert_called_with("tag name", "body")
    mock_driver.close.assert_not_called()
    mock_driver.quit.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_scrape_with_css_selector(_mocked_chrome_driver):
    html_content = "<html><body><div>test content</div><div class='test'>test content in a specific div</div></body></html>"
    mock_driver = mock_driver_with_html(html_content)
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="https://example.com", css_element="div.test")

    assert "test content in a specific div" in result
    mock_driver.get.assert_called_once_with("https://example.com")
    mock_driver.find_elements.assert_called_with("css selector", "div.test")
    mock_driver.close.assert_not_called()
    mock_driver.quit.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_scrape_with_return_html_true(_mocked_chrome_driver):
    html_content = "<html><body><div>HTML content</div></body></html>"
    mock_driver = mock_driver_with_html(html_content)
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="https://example.com", return_html=True)

    assert html_content in result
    mock_driver.get.assert_called_once_with("https://example.com")
    mock_driver.find_element.assert_called_with("tag name", "body")
    mock_driver.close.assert_not_called()
    mock_driver.quit.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_scrape_with_return_html_false(_mocked_chrome_driver):
    html_content = "<html><body><div>HTML content</div></body></html>"
    mock_driver = mock_driver_with_html(html_content)
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="https://example.com", return_html=False)

    assert "HTML content" in result
    mock_driver.get.assert_called_once_with("https://example.com")
    mock_driver.find_element.assert_called_with("tag name", "body")
    mock_driver.close.assert_not_called()
    mock_driver.quit.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_scrape_with_driver_error(_mocked_chrome_driver):
    mock_driver = MagicMock()
    mock_driver.find_element.side_effect = Exception("WebDriver error occurred")
    tool = initialize_tool_with(mock_driver)
    result = tool._run(website_url="https://example.com")
    assert result == "Error scraping website: WebDriver error occurred"
    mock_driver.close.assert_not_called()
    mock_driver.quit.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_initialization_with_driver(_mocked_chrome_driver):
    mock_driver = MagicMock()
    tool = initialize_tool_with(mock_driver)
    assert tool.driver == mock_driver


class FakeWindowDriver:
    """Mimics a real chromedriver: closing the only window ends navigation."""

    def __init__(self) -> None:
        self.get_calls = 0
        self.window_closed = False
        self.quit_calls = 0

    def get(self, url: str) -> None:
        if self.window_closed:
            raise Exception("no such window: target window already closed")
        self.get_calls += 1

    def find_element(self, by: str, selector: str):
        if self.window_closed:
            raise Exception("no such window: target window already closed")

        class Element:
            text = "body content"

            def get_attribute(self, name: str) -> str:
                return "<html><body>body content</body></html>"

        return Element()

    def find_elements(self, by: str, selector: str):
        if self.window_closed:
            raise Exception("no such window: target window already closed")
        return [self.find_element(by, selector)]

    def close(self) -> None:
        self.window_closed = True

    def quit(self) -> None:
        self.quit_calls += 1


@patch("selenium.webdriver.Chrome")
def test_driver_stays_reusable_across_runs(_mocked_chrome_driver):
    """The driver must survive a run so the tool can be called again.

    Real chromedriver raises 'no such window' once the only window has been
    closed, so closing the driver after every run made every call after the
    first fail.
    """
    driver = FakeWindowDriver()
    tool = SeleniumScrapingTool(driver=driver, wait_time=0)

    first = tool._run(website_url="https://example.com")
    second = tool._run(website_url="https://example.com", css_element="div.test")

    assert "body content" in first
    assert "body content" in second
    assert driver.get_calls == 2
    assert not driver.window_closed


@patch("selenium.webdriver.Chrome")
def test_close_ends_the_session(_mocked_chrome_driver):
    """close() must end the session so the browser process does not linger."""
    driver = FakeWindowDriver()
    tool = SeleniumScrapingTool(driver=driver, wait_time=0)

    tool.close()

    assert driver.quit_calls == 1


def test_schema_rejects_loopback_url():
    with pytest.raises(ValidationError):
        SeleniumScrapingToolSchema(
            website_url="http://127.0.0.1/admin", css_element="div"
        )


def test_schema_rejects_cloud_metadata_url():
    with pytest.raises(ValidationError):
        SeleniumScrapingToolSchema(
            website_url="http://169.254.169.254/latest/meta-data/",
            css_element="div",
        )


@patch("selenium.webdriver.Chrome")
def test_fixed_localhost_url_rejected_before_driver_creation(mocked_chrome):
    """A private fixed URL must fail before any WebDriver is created."""
    with pytest.raises(ValueError, match="private/reserved IP"):
        SeleniumScrapingTool(website_url="http://localhost/admin")

    mocked_chrome.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_run_rejects_loopback_url_without_navigating(_mocked_chrome_driver):
    mock_driver = MagicMock()
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="http://127.0.0.1/admin")

    assert "Error scraping website" in result
    assert "private/reserved IP" in result
    mock_driver.get.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_run_rejects_cloud_metadata_url_without_navigating(_mocked_chrome_driver):
    mock_driver = MagicMock()
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="http://169.254.169.254/latest/meta-data/")

    assert "Error scraping website" in result
    assert "private/reserved IP" in result
    mock_driver.get.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_fixed_public_url_is_accepted_and_navigated(_mocked_chrome_driver):
    html_content = "<html><body><div>fixed url content</div></body></html>"
    mock_driver = mock_driver_with_html(html_content)
    tool = SeleniumScrapingTool(
        website_url="https://example.com", driver=mock_driver, wait_time=0
    )

    assert tool.args_schema is FixedSeleniumScrapingToolSchema

    result = tool._run()

    assert "fixed url content" in result
    mock_driver.get.assert_called_once_with("https://example.com")


# A backslash is an authority delimiter to Chrome's WHATWG URL parser but
# ordinary data to urllib.parse, so this URL validates as the public host
# example.com yet navigates a real browser to 127.0.0.1:8899.
BACKSLASH_URL = r"http://127.0.0.1:8899\@example.com/"


def test_schema_rejects_backslash_url():
    with pytest.raises(ValidationError, match="backslash"):
        SeleniumScrapingToolSchema(
            website_url=BACKSLASH_URL, css_element="div"
        )


@patch("selenium.webdriver.Chrome")
def test_fixed_backslash_url_rejected_before_driver_creation(mocked_chrome):
    with pytest.raises(ValueError, match="backslash"):
        SeleniumScrapingTool(website_url=BACKSLASH_URL)

    mocked_chrome.assert_not_called()


@patch("selenium.webdriver.Chrome")
def test_run_rejects_backslash_url_without_navigating(_mocked_chrome_driver):
    mock_driver = MagicMock()
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url=BACKSLASH_URL)

    assert "Error scraping website" in result
    assert "backslash" in result
    mock_driver.get.assert_not_called()


class RedirectingDriver:
    """Driver double whose get() lands on a different URL, as a browser
    does after following a redirect, and serves that page's content."""

    def __init__(self, final_url: str, page_text: str) -> None:
        self._final_url = final_url
        self._page_text = page_text
        self.current_url = "about:blank"
        self.get_calls: list[str] = []

    def get(self, url: str) -> None:
        self.get_calls.append(url)
        self.current_url = self._final_url

    def find_element(self, by: str, selector: str):
        page_text = self._page_text

        class Element:
            def __init__(self, text: str) -> None:
                self.text = text

            def get_attribute(self, name: str) -> str:
                return f"<html><body>{page_text}</body></html>"

        return Element(page_text)

    def find_elements(self, by: str, selector: str):
        return [self.find_element(by, selector)]


@patch("selenium.webdriver.Chrome")
def test_run_rejects_redirect_to_loopback_url(_mocked_chrome_driver):
    """A public URL that redirects to loopback must not leak its content."""
    driver = RedirectingDriver("http://127.0.0.1/admin", "internal secret")
    tool = SeleniumScrapingTool(driver=driver, wait_time=0)

    result = tool._run(website_url="https://example.com")

    assert driver.get_calls == ["https://example.com"]
    assert "Error scraping website" in result
    assert "private/reserved IP" in result
    assert "internal secret" not in result


@patch("selenium.webdriver.Chrome")
def test_run_rejects_redirect_to_cloud_metadata_url(_mocked_chrome_driver):
    driver = RedirectingDriver(
        "http://169.254.169.254/latest/meta-data/", "metadata secret"
    )
    tool = SeleniumScrapingTool(driver=driver, wait_time=0)

    result = tool._run(website_url="https://example.com")

    assert "Error scraping website" in result
    assert "private/reserved IP" in result
    assert "metadata secret" not in result


@patch("selenium.webdriver.Chrome")
def test_run_allows_redirect_to_public_url(_mocked_chrome_driver):
    """Redirects that land on another public URL keep working."""
    driver = RedirectingDriver("https://example.com/landing", "landed content")
    tool = SeleniumScrapingTool(driver=driver, wait_time=0)

    result = tool._run(website_url="https://example.com")

    assert "landed content" in result
    assert driver.get_calls == ["https://example.com"]


class WaitRefreshDriver:
    """Driver double that lands on the requested (public) URL and then
    navigates elsewhere while the tool waits, as a meta refresh or a
    script on the initially landed page would."""

    def __init__(self, refresh_url: str, page_text: str) -> None:
        self._refresh_url = refresh_url
        self._page_text = page_text
        self.current_url = "about:blank"
        self.get_calls: list[str] = []

    def get(self, url: str) -> None:
        self.get_calls.append(url)
        self.current_url = url

    def navigate_during_wait(self) -> None:
        self.current_url = self._refresh_url

    def find_element(self, by: str, selector: str):
        page_text = self._page_text

        class Element:
            def __init__(self, text: str) -> None:
                self.text = text

            def get_attribute(self, name: str) -> str:
                return f"<html><body>{page_text}</body></html>"

        return Element(page_text)

    def find_elements(self, by: str, selector: str):
        return [self.find_element(by, selector)]


def run_with_wait_refresh(driver: WaitRefreshDriver) -> str:
    """Run the tool while the page navigates away during the wait."""
    tool = SeleniumScrapingTool(driver=driver, wait_time=3)
    module = "crewai_tools.tools.selenium_scraping_tool.selenium_scraping_tool"
    with patch(f"{module}.time") as mock_time:
        mock_time.sleep.side_effect = lambda *_args: driver.navigate_during_wait()
        return tool._run(website_url="https://example.com")


@patch("selenium.webdriver.Chrome")
def test_run_rejects_wait_refresh_to_loopback_url(_mocked_chrome_driver):
    """A page that refreshes to loopback during wait_time must not leak."""
    driver = WaitRefreshDriver("http://127.0.0.1/admin", "internal secret")

    result = run_with_wait_refresh(driver)

    assert "Error scraping website" in result
    assert "private/reserved IP" in result
    assert "internal secret" not in result


@patch("selenium.webdriver.Chrome")
def test_run_rejects_wait_refresh_to_cloud_metadata_url(_mocked_chrome_driver):
    driver = WaitRefreshDriver(
        "http://169.254.169.254/latest/meta-data/", "metadata secret"
    )

    result = run_with_wait_refresh(driver)

    assert "Error scraping website" in result
    assert "private/reserved IP" in result
    assert "metadata secret" not in result


@patch("selenium.webdriver.Chrome")
def test_run_allows_wait_refresh_to_public_url(_mocked_chrome_driver):
    """A page that refreshes to another public URL during the wait works."""
    driver = WaitRefreshDriver("https://example.com/landing", "landed content")

    result = run_with_wait_refresh(driver)

    assert "landed content" in result
