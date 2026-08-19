from typing import Any

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.oxylabs_base_tool.oxylabs_base_tool import OxylabsBaseTool


__all__ = ["OxylabsAmazonProductScraperConfig", "OxylabsAmazonProductScraperTool"]


class OxylabsAmazonProductScraperArgs(BaseModel):
    query: str = Field(description="Amazon product ASIN")


class OxylabsAmazonProductScraperConfig(BaseModel):
    """Amazon Product Scraper configuration options:
    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/amazon/product.
    """

    domain: str | None = Field(
        None, description="The domain to limit the search results to."
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


class OxylabsAmazonProductScraperTool(OxylabsBaseTool):
    """Scrape Amazon product pages with OxylabsAmazonProductScraperTool.

    Get Oxylabs account:
    https://dashboard.oxylabs.io/en

    Args:
        username (str): Oxylabs username.
        password (str): Oxylabs password.
        config: Configuration options. See ``OxylabsAmazonProductScraperConfig``
    """

    name: str = "Oxylabs Amazon Product Scraper tool"
    description: str = "Scrape Amazon product pages with Oxylabs Amazon Product Scraper"
    args_schema: type[BaseModel] = OxylabsAmazonProductScraperArgs

    config: OxylabsAmazonProductScraperConfig

    def _run(self, query: str) -> str | ToolFailure:
        response = self.oxylabs_api.amazon.scrape_product(
            query,
            **self.config.model_dump(exclude_none=True),
        )

        return self._handle_response(response)
