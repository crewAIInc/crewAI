import json
from typing import Any, Literal

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.decodo_base_tool.decodo_base_tool import DecodoBaseTool


__all__ = ["DecodoSearchArgs", "DecodoSearchTool"]


class DecodoSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    source: Literal["google", "amazon"] = Field(
        default="google",
        description="Where to search: 'google' for web results, 'amazon' for product listings",
    )
    geo: str | None = Field(
        default=None,
        description=(
            "Location to search from. Google: a country name, e.g. 'United States'. "
            "Amazon: a US ZIP code, e.g. '10001', or a two-letter country code for delivery outside the US, e.g. 'DE'"
        ),
    )


class DecodoSearchTool(DecodoBaseTool):
    """Search Google or Amazon with the Decodo Web Scraping API and return the parsed results as JSON.

    Args:
        api_key: Decodo Web Scraping API key. Defaults to the ``DECODO_API_KEY`` environment variable.
    """

    name: str = "Decodo Search"
    description: str = (
        "Search Google or Amazon and return the parsed results page as JSON. "
        "Use source='google' for up-to-date web results and rankings, "
        "source='amazon' to find and compare product listings."
    )
    args_schema: type[BaseModel] = DecodoSearchArgs

    def _run(
        self,
        query: str,
        source: Literal["google", "amazon"] = "google",
        geo: str | None = None,
    ) -> str | ToolFailure:
        from decodo import AmazonSearchParams, GoogleSearchParams

        params: Any
        if source == "amazon":
            params = AmazonSearchParams(query=query, geo=geo or None, parse=True)
        else:
            params = GoogleSearchParams(query=query, geo=geo or None, parse=True)

        content = self._scrape(params)
        if isinstance(content, ToolFailure | str):
            return content
        return json.dumps(content)
