from typing import Any

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.oxylabs_base_tool.oxylabs_base_tool import OxylabsBaseTool


__all__ = ["OxylabsGoogleSearchScraperConfig", "OxylabsGoogleSearchScraperTool"]


class OxylabsGoogleSearchScraperArgs(BaseModel):
    query: str = Field(description="Search query")


class OxylabsGoogleSearchScraperConfig(BaseModel):
    """Google Search Scraper configuration options:
    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/google/search/search.
    """

    domain: str | None = Field(
        None, description="The domain to limit the search results to."
    )
    start_page: int | None = Field(None, description="The starting page number.")
    pages: int | None = Field(None, description="The number of pages to scrape.")
    limit: int | None = Field(
        None, description="Number of results to retrieve in each page."
    )
    locale: str | None = Field(
        None,
        description="`Accept-Language` header value which changes your Google "
        "search page web interface language.",
    )
    geo_location: str | None = Field(None, description="The Deliver to location.")
    user_agent_type: str | None = Field(None, description="Device type and browser.")
    render: str | None = Field(None, description="Enables JavaScript rendering.")
    callback_url: str | None = Field(None, description="URL to your callback endpoint.")
    context: list[Any] | None = Field(
        None,
        description="Additional advanced settings and controls for specialized requirements.",
    )
    parse: bool | None = Field(None, description="True will return structured data.")
    parsing_instructions: dict[str, Any] | None = Field(
        None, description="Instructions for parsing the results."
    )


class OxylabsGoogleSearchScraperTool(OxylabsBaseTool):
    """Scrape Google Search results with OxylabsGoogleSearchScraperTool.

    Get Oxylabs account:
    https://dashboard.oxylabs.io/en

    Args:
        username (str): Oxylabs username.
        password (str): Oxylabs password.
        config: Configuration options. See ``OxylabsGoogleSearchScraperConfig``
    """

    name: str = "Oxylabs Google Search Scraper tool"
    description: str = "Scrape Google Search results with Oxylabs Google Search Scraper"
    args_schema: type[BaseModel] = OxylabsGoogleSearchScraperArgs

    config: OxylabsGoogleSearchScraperConfig

    def _run(self, query: str) -> str | ToolFailure:
        response = self.oxylabs_api.google.scrape_search(
            query,
            **self.config.model_dump(exclude_none=True),
        )

        return self._handle_response(response)
