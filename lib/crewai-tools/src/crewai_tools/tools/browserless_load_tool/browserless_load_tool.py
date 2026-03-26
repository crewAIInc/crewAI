import os
from typing import Any

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class BrowserlessLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL to load")


class BrowserlessLoadTool(BaseTool):
    name: str = "Browserless web load tool"
    description: str = (
        "Load a webpage using Browserless smart scrape and return its content. "
        "Automatically handles JavaScript rendering, anti-bot measures, and captchas."
    )
    args_schema: type[BaseModel] = BrowserlessLoadToolSchema
    api_token: str | None = None
    base_url: str | None = None
    formats: list[str] = Field(default_factory=lambda: ["markdown"])
    timeout: int = 60000
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BROWSERLESS_API_TOKEN",
                description="API token for Browserless services",
                required=False,
            ),
            EnvVar(
                name="BROWSERLESS_BASE_URL",
                description="Base URL for Browserless instance",
                required=False,
            ),
        ]
    )

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        formats: list[str] | None = None,
        timeout: int = 60000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_token = api_token or os.getenv("BROWSERLESS_API_TOKEN")
        self.base_url = (
            base_url
            or os.getenv("BROWSERLESS_BASE_URL")
            or "https://production-sfo.browserless.io"
        )
        if formats is not None:
            self.formats = formats
        self.timeout = timeout

        if not self.api_token:
            raise EnvironmentError(
                "BROWSERLESS_API_TOKEN environment variable is required "
                "for initialization. Get a token from https://browserless.io"
            )

    def _run(self, url: str) -> str:
        endpoint = f"{self.base_url}/smart-scrape/"
        params = {"token": self.api_token}
        payload = {"url": url, "formats": self.formats}
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            endpoint,
            params=params,
            json=payload,
            headers=headers,
            timeout=self.timeout / 1000,  # requests uses seconds
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("ok"):
            return f"Error scraping {url}: {data.get('message', 'Unknown error')}"

        # Return markdown if available, otherwise HTML content
        if "markdown" in self.formats and data.get("markdown"):
            return data["markdown"]
        if "html" in self.formats and data.get("content"):
            return data["content"] if isinstance(data["content"], str) else str(data["content"])
        if "links" in self.formats and data.get("links"):
            return "\n".join(data["links"])

        # Fallback to whatever content is available
        return data.get("content") or data.get("markdown") or str(data)
