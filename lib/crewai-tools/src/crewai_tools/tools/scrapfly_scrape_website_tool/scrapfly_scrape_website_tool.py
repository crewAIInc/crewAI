import logging
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


logger = logging.getLogger(__file__)


class ScrapflyScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(description="Webpage URL")
    scrape_format: Literal["raw", "markdown", "text"] | None = Field(
        default="markdown", description="Webpage extraction format"
    )
    scrape_config: dict[str, Any] | None = Field(
        default=None, description="Scrapfly request scrape config"
    )
    ignore_scrape_failures: bool | None = Field(
        default=None, description="whether to ignore failures"
    )


class ScrapflyScrapeWebsiteTool(BaseTool):
    name: str = "Scrapfly web scraping API tool"
    description: str = (
        "Scrape a webpage url using Scrapfly and return its content as markdown or text"
    )
    args_schema: type[BaseModel] = ScrapflyScrapeWebsiteToolSchema
    api_key: str | None = None
    scrapfly: Any | None = None
    package_dependencies: list[str] = Field(default_factory=lambda: ["scrapfly-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SCRAPFLY_API_KEY",
                description="API key for Scrapfly",
                required=True,
            ),
        ]
    )

    def __init__(self, api_key: str):
        super().__init__(
            name="Scrapfly web scraping API tool",
            description="Scrape a webpage url using Scrapfly and return its content as markdown or text",
        )
        try:
            from scrapfly import ScrapflyClient  # type: ignore[import-untyped]
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'scrapfly-sdk' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "scrapfly-sdk"], check=True)  # noqa: S607
            else:
                raise ImportError(
                    "`scrapfly-sdk` package not found, please run `uv add scrapfly-sdk`"
                ) from None
        self.scrapfly = ScrapflyClient(key=api_key or os.getenv("SCRAPFLY_API_KEY"))

    def _run(
        self,
        url: str,
        scrape_format: str = "markdown",
        scrape_config: dict[str, Any] | None = None,
        ignore_scrape_failures: bool | None = None,
    ):
        from scrapfly import ScrapeApiResponse, ScrapeConfig

        scrape_config = scrape_config if scrape_config is not None else {}
        try:
            response: ScrapeApiResponse = self.scrapfly.scrape(  # type: ignore[union-attr]
                ScrapeConfig(url, format=scrape_format, **scrape_config)
            )
            return response.scrape_result["content"]
        except Exception as e:
            if ignore_scrape_failures:
                logger.error(f"Error fetching data from {url}, exception: {e}")
                return None
            raise e
