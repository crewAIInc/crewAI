from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


class JinaScrapeWebsiteToolInput(BaseModel):
    """Input schema for JinaScrapeWebsiteTool."""

    website_url: str = Field(..., description="Mandatory website url to read the file")


class JinaScrapeWebsiteTool(BaseTool):
    name: str = "JinaScrapeWebsiteTool"
    description: str = "A tool that can be used to read a website content using Jina.ai reader and return markdown content."
    args_schema: type[BaseModel] = JinaScrapeWebsiteToolInput
    website_url: str | None = None
    api_key: str | None = None
    headers: dict = Field(default_factory=dict)

    def __init__(
        self,
        website_url: str | None = None,
        api_key: str | None = None,
        custom_headers: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if website_url is not None:
            self.website_url = website_url
            self.description = f"A tool that can be used to read {website_url}'s content and return markdown content."
            self._generate_description()

        if custom_headers is not None:
            self.headers = custom_headers

        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _run(self, website_url: str | None = None) -> str:
        url = website_url or self.website_url
        if not url:
            raise ValueError(
                "Website URL must be provided either during initialization or execution"
            )

        response = requests.get(
            f"https://r.jina.ai/{url}", headers=self.headers, timeout=15
        )
        response.raise_for_status()
        return response.text
