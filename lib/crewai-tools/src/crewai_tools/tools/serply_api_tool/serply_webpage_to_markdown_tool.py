import os
from typing import Any, Literal

from crewai.tools import EnvVar
from pydantic import BaseModel, Field
import requests

from crewai_tools.tools.rag.rag_tool import RagTool


class SerplyWebpageToMarkdownToolSchema(BaseModel):
    """Input for Serply Search."""

    url: str = Field(
        ...,
        description="Mandatory url you want to use to fetch and convert to markdown",
    )


class SerplyWebpageToMarkdownTool(RagTool):
    name: str = "Webpage to Markdown"
    description: str = "A tool to perform convert a webpage to markdown to make it easier for LLMs to understand"
    args_schema: type[BaseModel] = SerplyWebpageToMarkdownToolSchema
    request_url: str = "https://api.serply.io/v1/request"
    proxy_location: Literal[
        "US", "CA", "IE", "GB", "FR", "DE", "SE", "IN", "JP", "KR", "SG", "AU", "BR"
    ] = "US"
    headers: dict[str, Any] = Field(
        default_factory=lambda: {
            "X-API-KEY": os.environ["SERPLY_API_KEY"],
            "User-Agent": "crew-tools",
        }
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SERPLY_API_KEY",
                description="API key for Serply services",
                required=True,
            ),
        ]
    )

    def _run(  # type: ignore[override]
        self,
        url: str,
    ) -> str:
        if self.proxy_location and not self.headers.get("X-Proxy-Location"):
            self.headers["X-Proxy-Location"] = self.proxy_location

        data = {"url": url, "method": "GET", "response_type": "markdown"}
        response = requests.request(
            "POST",
            self.request_url,
            headers=self.headers,
            json=data,
            timeout=30,
        )
        return response.text
