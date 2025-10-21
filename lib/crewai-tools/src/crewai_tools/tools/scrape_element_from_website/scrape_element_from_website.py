import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False


class FixedScrapeElementFromWebsiteToolSchema(BaseModel):
    """Input for ScrapeElementFromWebsiteTool."""


class ScrapeElementFromWebsiteToolSchema(FixedScrapeElementFromWebsiteToolSchema):
    """Input for ScrapeElementFromWebsiteTool."""

    website_url: str = Field(..., description="Mandatory website url to read the file")
    css_element: str = Field(
        ...,
        description="Mandatory css reference for element to scrape from the website",
    )


class ScrapeElementFromWebsiteTool(BaseTool):
    name: str = "Read a website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: type[BaseModel] = ScrapeElementFromWebsiteToolSchema
    website_url: str | None = None
    cookies: dict | None = None
    css_element: str | None = None
    headers: dict | None = Field(
        default_factory=lambda: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Accept-Encoding": "gzip, deflate, br",
        }
    )

    def __init__(
        self,
        website_url: str | None = None,
        cookies: dict | None = None,
        css_element: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if website_url is not None:
            self.website_url = website_url
            self.css_element = css_element
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = FixedScrapeElementFromWebsiteToolSchema
            self._generate_description()
            if cookies is not None:
                self.cookies = {cookies["name"]: os.getenv(cookies["value"])}

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError(
                "beautifulsoup4 is not installed. Please install it with `pip install crewai-tools[beautifulsoup4]`"
            )

        website_url = kwargs.get("website_url", self.website_url)
        css_element = kwargs.get("css_element", self.css_element)

        if website_url is None or css_element is None:
            raise ValueError("Both website_url and css_element must be provided.")

        page = requests.get(
            website_url,
            headers=self.headers,
            cookies=self.cookies if self.cookies else {},
            timeout=30,
        )
        parsed = BeautifulSoup(page.content, "html.parser")
        elements = parsed.select(css_element)
        return "\n".join([element.get_text() for element in elements])
