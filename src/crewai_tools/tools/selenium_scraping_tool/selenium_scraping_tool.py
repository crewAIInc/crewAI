import re
import time
from typing import Any, Optional, Type
from urllib.parse import urlparse

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator


class FixedSeleniumScrapingToolSchema(BaseModel):
    """Input for SeleniumScrapingTool."""


class SeleniumScrapingToolSchema(FixedSeleniumScrapingToolSchema):
    """Input for SeleniumScrapingTool."""

    website_url: str = Field(
        ...,
        description="Mandatory website url to read the file. Must start with http:// or https://",
    )
    css_element: str = Field(
        ...,
        description="Mandatory css reference for element to scrape from the website",
    )

    @field_validator("website_url")
    def validate_website_url(cls, v):
        if not v:
            raise ValueError("Website URL cannot be empty")

        if len(v) > 2048:  # Common maximum URL length
            raise ValueError("URL is too long (max 2048 characters)")

        if not re.match(r"^https?://", v):
            raise ValueError("URL must start with http:// or https://")

        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")
        except Exception as e:
            raise ValueError(f"Invalid URL: {str(e)}")

        if re.search(r"\s", v):
            raise ValueError("URL cannot contain whitespace")

        return v


class SeleniumScrapingTool(BaseTool):
    name: str = "Read a website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: Type[BaseModel] = SeleniumScrapingToolSchema
    website_url: Optional[str] = None
    driver: Optional[Any] = None
    cookie: Optional[dict] = None
    wait_time: Optional[int] = 3
    css_element: Optional[str] = None
    return_html: Optional[bool] = False
    _by: Optional[Any] = None

    def __init__(
        self,
        website_url: Optional[str] = None,
        cookie: Optional[dict] = None,
        css_element: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'selenium' and 'webdriver-manager' packages. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(
                    ["uv", "pip", "install", "selenium", "webdriver-manager"],
                    check=True,
                )
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.common.by import By
            else:
                raise ImportError(
                    "`selenium` and `webdriver-manager` package not found, please run `uv add selenium webdriver-manager`"
                )

        options: Options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)
        self._by = By
        if cookie is not None:
            self.cookie = cookie

        if css_element is not None:
            self.css_element = css_element

        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = FixedSeleniumScrapingToolSchema

        self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        website_url = kwargs.get("website_url", self.website_url)
        css_element = kwargs.get("css_element", self.css_element)
        return_html = kwargs.get("return_html", self.return_html)
        try:
            self._make_request(website_url, self.cookie, self.wait_time)
            content = self._get_content(css_element, return_html)
            return "\n".join(content)
        except Exception as e:
            return f"Error scraping website: {str(e)}"
        finally:
            self.driver.close()

    def _get_content(self, css_element, return_html):
        content = []

        if self._is_css_element_empty(css_element):
            content.append(self._get_body_content(return_html))
        else:
            content.extend(self._get_elements_content(css_element, return_html))

        return content

    def _is_css_element_empty(self, css_element):
        return css_element is None or css_element.strip() == ""

    def _get_body_content(self, return_html):
        body_element = self.driver.find_element(self._by.TAG_NAME, "body")

        return (
            body_element.get_attribute("outerHTML")
            if return_html
            else body_element.text
        )

    def _get_elements_content(self, css_element, return_html):
        elements_content = []

        for element in self.driver.find_elements(self._by.CSS_SELECTOR, css_element):
            elements_content.append(
                element.get_attribute("outerHTML") if return_html else element.text
            )

        return elements_content

    def _make_request(self, url, cookie, wait_time):
        if not url:
            raise ValueError("URL cannot be empty")

        # Validate URL format
        if not re.match(r"^https?://", url):
            raise ValueError("URL must start with http:// or https://")

        self.driver.get(url)
        time.sleep(wait_time)
        if cookie:
            self.driver.add_cookie(cookie)
            time.sleep(wait_time)
            self.driver.get(url)
            time.sleep(wait_time)

    def close(self):
        self.driver.close()
