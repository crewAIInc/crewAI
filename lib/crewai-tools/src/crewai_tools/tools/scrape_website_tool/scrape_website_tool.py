import os
import re
from typing import Any

from pydantic import Field
import requests


try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
from crewai.tools import BaseTool
from pydantic import BaseModel


class FixedScrapeWebsiteToolSchema(BaseModel):
    """Input for ScrapeWebsiteTool."""


class ScrapeWebsiteToolSchema(FixedScrapeWebsiteToolSchema):
    """Input for ScrapeWebsiteTool."""

    website_url: str = Field(..., description="Mandatory website url to read the file")


class ScrapeWebsiteTool(BaseTool):
    name: str = "Read website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: type[BaseModel] = ScrapeWebsiteToolSchema
    website_url: str | None = None
    cookies: dict | None = None
    headers: dict | None = Field(
        default_factory=lambda: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    )
    render_js: bool = False

    def __init__(
        self,
        website_url: str | None = None,
        cookies: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError(
                "beautifulsoup4 is not installed. Please install it with `pip install crewai-tools[beautifulsoup4]`"
            )

        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = FixedScrapeWebsiteToolSchema
            self._generate_description()
            if cookies is not None:
                self.cookies = {cookies["name"]: os.getenv(cookies["value"])}

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        website_url: str | None = kwargs.get("website_url", self.website_url)
        if website_url is None:
            raise ValueError("Website URL must be provided.")
        # html_content = ""
        if self.render_js:
            try:
                from playwright.sync_api import sync_playwright # noqa: E402
            except ImportError:
                raise ImportError(
                    "Playwright is required for JS rendering. "
                    "Please install it with: pip install 'crewai-tools[playwright]'"
                ) from None

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                try:
                    page.goto(website_url, wait_until="networkidle")
                    content = page.content()
                    return self._process_html(content)
                finally:
                    browser.close()
        else:
            response = requests.get(
                website_url,
                timeout=15,
                headers=self.headers,
                cookies=self.cookies if self.cookies else {},
            )
            response.encoding = response.apparent_encoding
            return self._process_html(response.text)
    
    def _process_html(self, html_content: str) -> str:
        """Processes HTML content and returns a clean text representation."""
        parsed = BeautifulSoup(html_content, "html.parser")
        text = "The following text is scraped website content:\n\n"
        text += parsed.get_text(" ")
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\s+\n\s+", "\n", text)
