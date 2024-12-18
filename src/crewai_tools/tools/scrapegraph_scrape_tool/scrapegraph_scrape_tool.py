import os
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from scrapegraph_py import Client
from scrapegraph_py.logger import sgai_logger


class FixedScrapegraphScrapeToolSchema(BaseModel):
    """Input for ScrapegraphScrapeTool when website_url is fixed."""

    pass


class ScrapegraphScrapeToolSchema(FixedScrapegraphScrapeToolSchema):
    """Input for ScrapegraphScrapeTool."""

    website_url: str = Field(..., description="Mandatory website url to scrape")
    user_prompt: str = Field(
        default="Extract the main content of the webpage",
        description="Prompt to guide the extraction of content",
    )


class ScrapegraphScrapeTool(BaseTool):
    name: str = "Scrapegraph website scraper"
    description: str = "A tool that uses Scrapegraph AI to intelligently scrape website content."
    args_schema: Type[BaseModel] = ScrapegraphScrapeToolSchema
    website_url: Optional[str] = None
    user_prompt: Optional[str] = None
    api_key: Optional[str] = None

    def __init__(
        self,
        website_url: Optional[str] = None,
        user_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("SCRAPEGRAPH_API_KEY")
        
        if not self.api_key:
            raise ValueError("Scrapegraph API key is required")

        if website_url is not None:
            self.website_url = website_url
            self.description = f"A tool that uses Scrapegraph AI to intelligently scrape {website_url}'s content."
            self.args_schema = FixedScrapegraphScrapeToolSchema
            
        if user_prompt is not None:
            self.user_prompt = user_prompt

        # Configure logging
        sgai_logger.set_logging(level="INFO")

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        website_url = kwargs.get("website_url", self.website_url)
        user_prompt = kwargs.get("user_prompt", self.user_prompt) or "Extract the main content of the webpage"

        if not website_url:
            raise ValueError("website_url is required")

        # Initialize the client
        sgai_client = Client(api_key=self.api_key)

        try:
            # Make the SmartScraper request
            response = sgai_client.smartscraper(
                website_url=website_url,
                user_prompt=user_prompt,
            )

            # Return the result
            return response["result"]
        finally:
            # Always close the client
            sgai_client.close()
