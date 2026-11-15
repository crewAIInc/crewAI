import json
import os
from typing import Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class SpecStreamExtractToolSchema(BaseModel):
    """Input schema for SpecStreamExtractTool."""

    urls: list[str] | str = Field(
        ...,
        description="The URL or list of URLs (up to 20) to read.",
    )


class SpecStreamExtractTool(BaseTool):
    """Reads the main text of web pages with the SpecStream Extract API.

    Attributes:
        api_key: The SpecStream API key. Read from SPECSTREAM_API_KEY if not given.
        base_url: The SpecStream API address.
        format: Return each page as "markdown" or plain "text".
        timeout: Request timeout in seconds.
    """

    name: str = "SpecStream Extract"
    description: str = (
        "Reads the main text of one or more web pages with the SpecStream Extract "
        "API. Returns a JSON list with url, status, title and text for each page."
    )
    args_schema: type[BaseModel] = SpecStreamExtractToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SPECSTREAM_API_KEY"),
        description="The SpecStream API key. Read from SPECSTREAM_API_KEY if not given.",
    )
    base_url: str = "https://api.specstream.ai"
    format: Literal["markdown", "text"] = "markdown"
    timeout: int = 120
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SPECSTREAM_API_KEY",
                description="API key for SpecStream",
                required=True,
            ),
        ]
    )

    def _run(self, urls: list[str] | str) -> str:
        if not self.api_key:
            raise ValueError("SPECSTREAM_API_KEY environment variable is required")

        url_list = [urls] if isinstance(urls, str) else list(urls)
        if not 1 <= len(url_list) <= 20:
            raise ValueError("Provide between 1 and 20 URLs.")

        try:
            response = requests.post(
                f"{self.base_url}/v1/extract",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                json={"urls": url_list, "format": self.format},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"SpecStream API request failed: {exc}") from exc

        if not response.ok:
            raise RuntimeError(
                f"SpecStream API error (HTTP {response.status_code}): {response.text[:500]}"
            )

        pages = response.json().get("data", {}).get("results") or []
        results = [
            {
                "url": page.get("url"),
                "status": page.get("status"),
                "title": page.get("title"),
                "text": page.get("text"),
            }
            for page in pages
        ]
        return json.dumps(results, indent=2)
