import logging
from typing import Any, Dict, Literal, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__file__)


class ScrapflyScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(description="Webpage URL")
    scrape_format: Optional[Literal["raw", "markdown", "text"]] = Field(
        default="markdown", description="Webpage extraction format"
    )
    scrape_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Scrapfly request scrape config"
    )
    ignore_scrape_failures: Optional[bool] = Field(
        default=None, description="whether to ignore failures"
    )


class ScrapflyScrapeWebsiteTool(BaseTool):
    name: str = "Scrapfly web scraping API tool"
    description: str = (
        "Scrape a webpage url using Scrapfly and return its content as markdown or text"
    )
    args_schema: Type[BaseModel] = ScrapflyScrapeWebsiteToolSchema
    api_key: str = None
    scrapfly: Optional[Any] = None

    def __init__(self, api_key: str):
        super().__init__()
        try:
            from scrapfly import ScrapflyClient
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'scrapfly-sdk' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "scrapfly-sdk"], check=True)
            else:
                raise ImportError(
                    "`scrapfly-sdk` package not found, please run `uv add scrapfly-sdk`"
                )
        self.scrapfly = ScrapflyClient(key=api_key)

    def _run(
        self,
        url: str,
        scrape_format: str = "markdown",
        scrape_config: Optional[Dict[str, Any]] = None,
        ignore_scrape_failures: Optional[bool] = None,
    ):
        from scrapfly import ScrapeApiResponse, ScrapeConfig

        scrape_config = scrape_config if scrape_config is not None else {}
        try:
            response: ScrapeApiResponse = self.scrapfly.scrape(
                ScrapeConfig(url, format=scrape_format, **scrape_config)
            )
            return response.scrape_result["content"]
        except Exception as e:
            if ignore_scrape_failures:
                logger.error(f"Error fetching data from {url}, exception: {e}")
                return None
            else:
                raise e
