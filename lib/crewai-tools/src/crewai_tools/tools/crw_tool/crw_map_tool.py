from __future__ import annotations

from typing import Any

import requests as http_requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


class CrwMapWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL to discover links from")


class CrwMapWebsiteTool(BaseTool):
    """Discover all URLs on a website using CRW's map endpoint.

    Useful for understanding site structure before targeted scraping.
    Uses sitemap.xml and link discovery to find all pages.

    Args:
        api_url: CRW server URL. Default: http://localhost:3000
        api_key: Optional API key (required for fastcrw.com cloud).
        config: Map configuration options.

    Configuration options:
        maxDepth (int): Maximum discovery depth. Default: 2
        useSitemap (bool): Also read sitemap.xml. Default: True
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "CRW website map tool"
    description: str = (
        "Discover all URLs on a website. Useful for understanding site structure "
        "before scraping specific pages."
    )
    args_schema: type[BaseModel] = CrwMapWebsiteToolSchema
    api_url: str = "http://localhost:3000"
    api_key: str | None = None
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "maxDepth": 2,
            "useSitemap": True,
        }
    )

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="CRW_API_URL",
                description="CRW server URL (default: http://localhost:3000)",
                required=False,
                default="http://localhost:3000",
            ),
            EnvVar(
                name="CRW_API_KEY",
                description="API key for CRW (required for fastcrw.com cloud)",
                required=False,
            ),
        ]
    )

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        import os

        self.api_url = api_url or os.getenv("CRW_API_URL", "http://localhost:3000")
        self.api_key = api_key or os.getenv("CRW_API_KEY")

    def _run(self, url: str) -> Any:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"url": url, **self.config}

        response = http_requests.post(
            f"{self.api_url.rstrip('/')}/v1/map",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            raise RuntimeError(
                f"CRW map failed: {result.get('error', 'Unknown error')}"
            )

        links = result.get("data", {}).get("links", [])
        return "\n".join(links) if links else "No links discovered."
