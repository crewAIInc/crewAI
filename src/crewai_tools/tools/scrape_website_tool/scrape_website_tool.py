import os
import re
from typing import Any, Optional, Type

import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FixedScrapeWebsiteToolSchema(BaseModel):
    """Input for ScrapeWebsiteTool."""


class ScrapeWebsiteToolSchema(FixedScrapeWebsiteToolSchema):
    """Input for ScrapeWebsiteTool."""

    website_url: str = Field(..., description="Mandatory website url to read the file")


class ScrapeWebsiteTool(BaseTool):
    name: str = "Read website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: Type[BaseModel] = ScrapeWebsiteToolSchema
    website_url: Optional[str] = None
    cookies: Optional[dict] = None
    headers: Optional[dict] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(
        self,
        website_url: Optional[str] = None,
        cookies: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        website_url = kwargs.get("website_url", self.website_url)
        page = requests.get(
            website_url,
            timeout=15,
            headers=self.headers,
            cookies=self.cookies if self.cookies else {},
        )

        page.encoding = page.apparent_encoding
        parsed = BeautifulSoup(page.text, "html.parser")

        text = parsed.get_text(" ")
        text = re.sub("[ \t]+", " ", text)
        text = re.sub("\\s+\n\\s+", "\n", text)
        return text
