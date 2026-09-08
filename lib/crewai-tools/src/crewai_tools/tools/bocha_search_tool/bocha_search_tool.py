from datetime import datetime
import os
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, field_validator
import requests


class BochaSearchToolSchema(BaseModel):
    """Input schema for BochaSearchTool."""

    query: str = Field(
        ..., description="Mandatory search query to search the internet"
    )
    count: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of search results to return (1-50)",
    )


class BochaSearchTool(BaseTool):
    """BochaSearchTool - A production-ready tool for web search using Bocha AI API."""

    name: str = "Bocha Web Search"
    description: str = (
        "Searches the internet using Bocha AI's search API. "
        "Returns ranked results with titles, URLs, summaries, site names."
    )
    args_schema: type[BaseModel] = BochaSearchToolSchema
    api_url: str = "https://api.bochaai.com/v1/web-search"  # Fixed: removed trailing spaces
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BOCHA_API_KEY",
                description="API key for Bocha Search (get from https://bochaai.com)",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.environ.get("BOCHA_API_KEY")
        if not api_key:
            raise ValueError(
                "BOCHA_API_KEY environment variable is required. "
                "Set it via `export BOCHA_API_KEY='your_key'` or .env file."
            )
        self._api_key = api_key

    def _run(self, **kwargs: Any) -> str:
        # === Parse arguments ===
        try:
            query = kwargs.get("query") or kwargs.get("search_query")
            if not query:
                return "Error: Missing required parameter 'query'"

            count = kwargs.get("count", 10)
        except Exception as e:
            return f"Error parsing input parameters: {e}"

        # === Prepare API request ===
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "count": count,
        }

        # === Execute request with timeout and error handling ===
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # === Handle API-level errors ===
            if data.get("code") != 200:
                return (
                    f"Bocha API error: {data.get('msg', 'Unknown error')} "
                    f"(code: {data.get('code', 'N/A')})"
                )

            webpages = data.get("data", {}).get("webPages", {}).get("value", [])
            if not webpages:
                result = "No results found for the query."
            else:
                result_lines = []
                for idx, page in enumerate(webpages[:count], start=1):
                    title = page.get("name", "No title")
                    url = page.get("url", "No URL")
                    summary_text = page.get("summary", "No summary available")
                    site_name = page.get("siteName", "Unknown site")
                    site_icon = page.get("siteIcon", "")
                    crawled = page.get("dateLastCrawled", "Unknown date")

                    result_lines.extend([
                        f"Result {idx}",
                        f"Title: {title}",
                        f"URL: {url}",
                        f"Summary: {summary_text}",
                        f"Site: {site_name}",
                        f"Icon: {site_icon}",
                        f"Crawled: {crawled}",
                        "-" * 50,
                    ])
                result = "\n".join(result_lines)

            return result

        except requests.exceptions.Timeout:
            return "Error: Request timed out (30s). Bocha API may be slow or unavailable."
        except requests.exceptions.ConnectionError:
            return "Error: Failed to connect to Bocha API. Check your network connection."
        except requests.exceptions.HTTPError as e:
            return f"HTTP error from Bocha API: {e.response.status_code} - {e.response.text[:200]}"
        except ValueError as e:  # JSON decode errors
            return f"Error parsing Bocha API response: {e}"
        except Exception as e:
            return f"Unexpected error during Bocha search: {type(e).__name__}: {e}"
