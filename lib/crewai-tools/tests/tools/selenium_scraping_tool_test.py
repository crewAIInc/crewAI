import os
import tempfile
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup
from crewai_tools.tools.selenium_scraping_tool.selenium_scraping_tool import (
    SeleniumScrapingTool,
)
from selenium.webdriver.chrome.options import Options


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
    mock_driver.close.assert_called_once()


@patch("selenium.webdriver.Chrome")
def test_scrape_with_css_selector(_mocked_chrome_driver):
    html_content = "<html><body><div>test content</div><div class='test'>test content in a specific div</div></body></html>"
    mock_driver = mock_driver_with_html(html_content)
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="https://example.com", css_element="div.test")

    assert "test content in a specific div" in result
    mock_driver.get.assert_called_once_with("https://example.com")
    mock_driver.find_elements.assert_called_with("css selector", "div.test")
    mock_driver.close.assert_called_once()


@patch("selenium.webdriver.Chrome")
def test_scrape_with_return_html_true(_mocked_chrome_driver):
    html_content = "<html><body><div>HTML content</div></body></html>"
    mock_driver = mock_driver_with_html(html_content)
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="https://example.com", return_html=True)

    assert html_content in result
    mock_driver.get.assert_called_once_with("https://example.com")
    mock_driver.find_element.assert_called_with("tag name", "body")
    mock_driver.close.assert_called_once()


@patch("selenium.webdriver.Chrome")
def test_scrape_with_return_html_false(_mocked_chrome_driver):
    html_content = "<html><body><div>HTML content</div></body></html>"
    mock_driver = mock_driver_with_html(html_content)
    tool = initialize_tool_with(mock_driver)

    result = tool._run(website_url="https://example.com", return_html=False)

    assert "HTML content" in result
    mock_driver.get.assert_called_once_with("https://example.com")
    mock_driver.find_element.assert_called_with("tag name", "body")
    mock_driver.close.assert_called_once()


@patch("selenium.webdriver.Chrome")
def test_scrape_with_driver_error(_mocked_chrome_driver):
    mock_driver = MagicMock()
    mock_driver.find_element.side_effect = Exception("WebDriver error occurred")
    tool = initialize_tool_with(mock_driver)
    result = tool._run(website_url="https://example.com")
    assert result == "Error scraping website: WebDriver error occurred"
    mock_driver.close.assert_called_once()


@patch("selenium.webdriver.Chrome")
def test_initialization_with_driver(_mocked_chrome_driver):
    mock_driver = MagicMock()
    tool = initialize_tool_with(mock_driver)
    assert tool.driver == mock_driver
