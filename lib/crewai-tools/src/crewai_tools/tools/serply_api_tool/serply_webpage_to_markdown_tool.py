import os

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
    proxy_location: str | None = "US"
    headers: dict | None = Field(default_factory=dict)
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SERPLY_API_KEY",
                description="API key for Serply services",
                required=True,
            ),
        ]
    )

    def __init__(self, proxy_location: str | None = "US", **kwargs):
        """proxy_location: (str): Where to perform the search, specifically for a specific country results.
        ['US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'] (defaults to US).
        """
        super().__init__(**kwargs)
        self.proxy_location = proxy_location
        self.headers = {
            "X-API-KEY": os.environ["SERPLY_API_KEY"],
            "User-Agent": "crew-tools",
            "X-Proxy-Location": proxy_location,
        }

    def _run(
        self,
        url: str,
    ) -> str:
        data = {"url": url, "method": "GET", "response_type": "markdown"}
        response = requests.request(
            "POST",
            self.request_url,
            headers=self.headers,
            json=data,
            timeout=30,
        )
        return response.text
