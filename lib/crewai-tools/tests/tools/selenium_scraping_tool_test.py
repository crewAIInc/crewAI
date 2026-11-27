import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup
from crewai_tools.tools.selenium_scraping_tool.selenium_scraping_tool import (
    SeleniumScrapingTool,
)
from selenium.webdriver.chrome.options import Options


from crewai_tools.security.safe_path import validate_url as real_validate_url


MODULE = "crewai_tools.tools.selenium_scraping_tool.selenium_scraping_tool"


@pytest.fixture(autouse=True)
def _allow_example_com():
    """Skip real DNS in tests that aren't about URL validation."""
    with patch(f"{MODULE}.validate_url", side_effect=lambda url: url):
        yield


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
        self.current_url = "about:blank"

    def get(self, url: str) -> None:
        if self.window_closed:
            raise Exception("no such window: target window already closed")
        self.get_calls += 1
        self.current_url = url

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


class RedirectingDriver:
    """Records navigation and cookies; ``redirects`` maps a URL to where it lands."""

    def __init__(self, redirects: dict[str, str] | None = None) -> None:
        self.redirects = redirects or {}
        self.current_url = "about:blank"
        self.visited: list[str] = []
        self.cookies: list[tuple[str, dict]] = []

    def get(self, url: str) -> None:
        self.visited.append(url)
        self.current_url = self.redirects.get(url, url)

    def add_cookie(self, cookie: dict) -> None:
        self.cookies.append((self.current_url, cookie))


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8765\\@example.com/",
        "https://example.com\\@169.254.169.254/",
    ],
)
def test_backslash_urls_are_blocked_before_navigation(url):
    """urlparse and Chrome disagree on the host when a backslash is present."""
    driver = RedirectingDriver()
    tool = SeleniumScrapingTool(driver=driver, wait_time=0)

    result = tool._run(website_url=url)

    assert "backslash" in result
    assert driver.visited == []


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/admin",
        "http://localhost/",
        "http://10.0.0.5/",
    ],
)
def test_internal_urls_are_blocked_before_navigation(url):
    driver = RedirectingDriver()
    tool = SeleniumScrapingTool(driver=driver, wait_time=0)

    with patch(f"{MODULE}.validate_url", wraps=real_validate_url):
        result = tool._run(website_url=url)

    assert result.startswith("Error scraping website:")
    assert driver.visited == []


def test_redirect_to_internal_address_is_rejected():
    driver = RedirectingDriver(
        {"https://public.example/": "http://169.254.169.254/latest/meta-data/"}
    )
    tool = SeleniumScrapingTool(driver=driver, wait_time=0)

    def fake_validate(url: str) -> str:
        if "169.254" in url:
            raise ValueError("private address")
        return url

    with patch(f"{MODULE}.validate_url", side_effect=fake_validate):
        result = tool._run(website_url="https://public.example/")

    assert "Error scraping website" in result


def test_cookie_is_not_sent_to_a_host_outside_its_domain():
    driver = RedirectingDriver()
    cookie = {"name": "session", "value": "secret", "domain": "trusted.example"}
    tool = SeleniumScrapingTool(driver=driver, cookie=cookie, wait_time=0)

    tool._run(website_url="https://attacker.example/")

    assert driver.cookies == []


def test_cookie_is_not_sent_when_redirected_off_domain():
    driver = RedirectingDriver({"https://trusted.example/": "https://attacker.example/"})
    cookie = {"name": "session", "value": "secret", "domain": "trusted.example"}
    tool = SeleniumScrapingTool(driver=driver, cookie=cookie, wait_time=0)

    tool._run(website_url="https://trusted.example/")

    assert driver.cookies == []


@pytest.mark.parametrize(
    "url", ["https://trusted.example/", "https://app.trusted.example/page"]
)
def test_cookie_is_sent_to_its_own_domain(url):
    driver = RedirectingDriver()
    cookie = {"name": "session", "value": "secret", "domain": ".trusted.example"}
    tool = SeleniumScrapingTool(driver=driver, cookie=cookie, wait_time=0)

    tool._run(website_url=url)

    assert driver.cookies == [(url, cookie)]


def test_cookie_without_domain_is_bound_to_the_fixed_website_url():
    cookie = {"name": "session", "value": "secret"}

    fixed = RedirectingDriver()
    SeleniumScrapingTool(
        driver=fixed,
        website_url="https://trusted.example/",
        cookie=cookie,
        wait_time=0,
    )._run()
    assert len(fixed.cookies) == 1

    open_ended = RedirectingDriver()
    SeleniumScrapingTool(driver=open_ended, cookie=cookie, wait_time=0)._run(
        website_url="https://attacker.example/"
    )
    assert open_ended.cookies == []
