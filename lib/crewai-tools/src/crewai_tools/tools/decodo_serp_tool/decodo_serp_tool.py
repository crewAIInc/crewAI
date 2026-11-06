import json
from typing import Any, Literal

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.decodo_base_tool.decodo_base_tool import DecodoBaseTool


__all__ = ["DecodoSERPArgs", "DecodoSERPTool"]

Engine = Literal["google", "bing", "google_ai_mode"]


class DecodoSERPArgs(BaseModel):
    query: str = Field(description="Search query")
    engine: Engine = Field(
        default="google",
        description=(
            "'google' or 'bing' for a results page; 'google_ai_mode' for Google's AI-written answer "
            "with the links it cites"
        ),
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Results page to fetch, starting at 1. Ignored by google_ai_mode",
    )
    geo: str | None = Field(
        default=None,
        description="Country to search from, as a country name, e.g. 'United States'",
    )


class DecodoSERPTool(DecodoBaseTool):
    """Search Google or Bing, or ask Google AI Mode, and return the parsed results as JSON.

    Args:
        api_key: Decodo Web Scraping API key. Defaults to the ``DECODO_API_KEY`` environment variable.
    """

    name: str = "Decodo SERP"
    description: str = (
        "Get search engine results as structured JSON. "
        "Use engine='google' or engine='bing' for a full results page, with page for deeper results. "
        "Use engine='google_ai_mode' for Google's AI-written answer and the sources it cites."
    )
    args_schema: type[BaseModel] = DecodoSERPArgs

    def _run(
        self,
        query: str,
        engine: Engine = "google",
        page: int = 1,
        geo: str | None = None,
    ) -> str | ToolFailure:
        from decodo import BingSearchParams, GoogleAiModeParams, GoogleSearchParams

        geo = geo or None
        page_from = page if page > 1 else None
        params: Any
        if engine == "google_ai_mode":
            params = GoogleAiModeParams(query=query, geo=geo, parse=True)
        elif engine == "bing":
            # Without JavaScript rendering, Bing fails with status 613 (target could not be scraped).
            params = BingSearchParams(
                query=query,
                geo=geo,
                page_from=page_from,
                headless="html",
                parse=True,
            )
        else:
            params = GoogleSearchParams(
                query=query, geo=geo, page_from=page_from, parse=True
            )

        content = self._scrape(params)
        if isinstance(content, ToolFailure | str):
            return content
        return json.dumps(content)
