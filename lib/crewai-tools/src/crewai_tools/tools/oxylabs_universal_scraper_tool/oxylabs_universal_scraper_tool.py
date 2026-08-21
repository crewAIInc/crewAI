from typing import Any

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.oxylabs_base_tool.oxylabs_base_tool import OxylabsBaseTool


__all__ = ["OxylabsUniversalScraperConfig", "OxylabsUniversalScraperTool"]


class OxylabsUniversalScraperArgs(BaseModel):
    url: str = Field(description="Website URL")


class OxylabsUniversalScraperConfig(BaseModel):
    """Universal Scraper configuration options:
    https://developers.oxylabs.io/scraper-apis/web-scraper-api/other-websites.
    """

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


class OxylabsUniversalScraperTool(OxylabsBaseTool):
    """Scrape any website with OxylabsUniversalScraperTool.

    Get Oxylabs account:
    https://dashboard.oxylabs.io/en

    Args:
        username (str): Oxylabs username.
        password (str): Oxylabs password.
        config: Configuration options. See ``OxylabsUniversalScraperConfig``
    """

    name: str = "Oxylabs Universal Scraper tool"
    description: str = "Scrape any url with Oxylabs Universal Scraper"
    args_schema: type[BaseModel] = OxylabsUniversalScraperArgs

    config: OxylabsUniversalScraperConfig

    def _run(self, url: str) -> str | ToolFailure:
        response = self.oxylabs_api.universal.scrape_url(
            url,
            **self.config.model_dump(exclude_none=True),
        )

        return self._handle_response(response)
