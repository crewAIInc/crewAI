import re
import time
from typing import Any
from urllib.parse import urlparse

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from crewai_tools.security.safe_path import validate_url


def _validate_browser_url(url: str) -> str:
    """Validate *url* for navigation by a real browser.

    Extends ``validate_url`` with a parser-differential guard. Chrome's
    WHATWG URL parser treats a backslash as an authority delimiter for
    http(s) URLs, while ``urllib.parse`` treats it as ordinary data, so
    ``http://127.0.0.1:8899\\@example.com/`` validates as the public host
    ``example.com`` but navigates Chrome to ``127.0.0.1``. A raw backslash
    is never valid in an http(s) URL (RFC 3986 requires ``%5C``), so reject
    it outright before the shared checks run.
    """
    if "\\" in url:
        raise ValueError(
            f"URL contains a backslash: '{url}'. Browsers interpret "
            "backslashes in URLs differently from this validator, so the "
            "URL is rejected."
        )
    return validate_url(url)


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
    @classmethod
    def validate_website_url(cls, v: str) -> str:
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
            raise ValueError(f"Invalid URL: {e!s}") from e

        if re.search(r"\s", v):
            raise ValueError("URL cannot contain whitespace")

        return _validate_browser_url(v)


class SeleniumScrapingTool(BaseTool):
    name: str = "Read a website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: type[BaseModel] = SeleniumScrapingToolSchema
    website_url: str | None = None
    driver: Any | None = None
    cookie: dict[str, Any] | None = None
    wait_time: int | None = 3
    css_element: str | None = None
    return_html: bool | None = False
    _by: Any | None = None
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["selenium", "webdriver-manager"]
    )

    def __init__(
        self,
        website_url: str | None = None,
        cookie: dict[str, Any] | None = None,
        css_element: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if website_url is not None:
            _validate_browser_url(website_url)
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import (
                Options,
            )
            from selenium.webdriver.common.by import (
                By,
            )
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'selenium' and 'webdriver-manager' packages. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(
                    ["uv", "pip", "install", "selenium", "webdriver-manager"],  # noqa: S607
                    check=True,
                )
                from selenium import webdriver
                from selenium.webdriver.chrome.options import (
                    Options,
                )
                from selenium.webdriver.common.by import (
                    By,
                )
            else:
                raise ImportError(
                    "`selenium` and `webdriver-manager` package not found, please run `uv add selenium webdriver-manager`"
                ) from None

        if "driver" not in kwargs:
            if "options" not in kwargs:
                options: Options = Options()
                options.add_argument("--headless")
            else:
                options = kwargs["options"]
            self.driver = webdriver.Chrome(options=options)
        else:
            self.driver = kwargs["driver"]

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
            return f"Error scraping website: {e!s}"

    def _get_content(
        self, css_element: str | None, return_html: bool | None
    ) -> list[str]:
        self._validate_landed_url()
        content: list[str] = []

        if self._is_css_element_empty(css_element):
            content.append(self._get_body_content(return_html))
        else:
            content.extend(self._get_elements_content(css_element, return_html))

        return content

    def _is_css_element_empty(self, css_element: str | None) -> bool:
        return css_element is None or css_element.strip() == ""

    def _get_body_content(self, return_html: bool | None) -> str:
        if self.driver is None or self._by is None:
            raise RuntimeError("Driver not initialized. Call _run first.")
        body_element = self.driver.find_element(self._by.TAG_NAME, "body")

        return str(
            body_element.get_attribute("outerHTML")
            if return_html
            else body_element.text
        )

    def _get_elements_content(
        self, css_element: str | None, return_html: bool | None
    ) -> list[str]:
        if self.driver is None or self._by is None:
            raise RuntimeError("Driver not initialized. Call _run first.")
        elements_content: list[str] = []

        for element in self.driver.find_elements(self._by.CSS_SELECTOR, css_element):
            elements_content.append(  # noqa: PERF401
                element.get_attribute("outerHTML") if return_html else element.text
            )

        return elements_content

    def _make_request(
        self, url: str | None, cookie: dict[str, Any] | None, wait_time: int | None
    ) -> None:
        if not url:
            raise ValueError("URL cannot be empty")

        if not re.match(r"^https?://", url):
            raise ValueError("URL must start with http:// or https://")

        _validate_browser_url(url)

        if self.driver is None:
            raise RuntimeError("Driver not initialized. Call _run first.")
        sleep_time = wait_time or 0
        self.driver.get(url)
        self._validate_landed_url()
        time.sleep(sleep_time)
        if cookie:
            self.driver.add_cookie(cookie)
            time.sleep(sleep_time)
            self.driver.get(url)
            self._validate_landed_url()
            time.sleep(sleep_time)
        self._validate_landed_url()

    def _validate_landed_url(self) -> None:
        """Re-validate the URL the browser actually landed on.

        ``_validate_browser_url`` checks the requested URL string, but the
        browser follows redirects on its own: a public URL can 3xx-redirect
        to a private or cloud-metadata address, and the tool would return
        that page. If the final URL is blocked, raise so no content is read.

        Called after each navigation, again after the wait, and at the
        start of content extraction: a page can also navigate during
        ``wait_time`` (meta refresh, script), so the only check that
        matters is the one immediately before content is read.

        This is mitigation, not prevention, and two limits are deliberate:
        the browser has already made the redirected request by the time
        this runs, and the check validates the landed URL string, not the
        IP the browser connected to, so DNS rebinding between validation
        and navigation is not fully closed for a browser driver. Preventing
        the request itself would require intercepting the browser's
        network traffic (CDP/BiDi), which selenium's Python API does not
        support reliably for arbitrary user-supplied drivers.
        """
        if self.driver is None:
            return
        current_url = getattr(self.driver, "current_url", None)
        if not isinstance(current_url, str) or not re.match(r"^https?://", current_url):
            return
        _validate_browser_url(current_url)

    def close(self) -> None:
        """End the browser session so the process does not linger."""
        if self.driver is not None:
            self.driver.quit()
