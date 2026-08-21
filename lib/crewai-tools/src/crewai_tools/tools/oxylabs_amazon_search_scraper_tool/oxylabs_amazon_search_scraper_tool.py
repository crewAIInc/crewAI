from typing import Any

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.oxylabs_base_tool.oxylabs_base_tool import OxylabsBaseTool


__all__ = ["OxylabsAmazonSearchScraperConfig", "OxylabsAmazonSearchScraperTool"]


class OxylabsAmazonSearchScraperArgs(BaseModel):
    query: str = Field(description="Amazon search term")


class OxylabsAmazonSearchScraperConfig(BaseModel):
    """Amazon Search Scraper configuration options:
    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/amazon/search.
    """

    domain: str | None = Field(
        None, description="The domain to limit the search results to."
    )
    start_page: int | None = Field(None, description="The starting page number.")
    pages: int | None = Field(None, description="The number of pages to scrape.")
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


class OxylabsAmazonSearchScraperTool(OxylabsBaseTool):
    """Scrape Amazon search results with OxylabsAmazonSearchScraperTool.

    Get Oxylabs account:
    https://dashboard.oxylabs.io/en

    Args:
        username (str): Oxylabs username.
        password (str): Oxylabs password.
        config: Configuration options. See ``OxylabsAmazonSearchScraperConfig``
    """

    name: str = "Oxylabs Amazon Search Scraper tool"
    description: str = "Scrape Amazon search results with Oxylabs Amazon Search Scraper"
    args_schema: type[BaseModel] = OxylabsAmazonSearchScraperArgs

    config: OxylabsAmazonSearchScraperConfig

    def _run(self, query: str) -> str | ToolFailure:
        response = self.oxylabs_api.amazon.scrape_search(
            query,
            **self.config.model_dump(exclude_none=True),
        )

        return self._handle_response(response)
