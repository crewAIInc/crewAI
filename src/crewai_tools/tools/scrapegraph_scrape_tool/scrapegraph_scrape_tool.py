import os
from typing import TYPE_CHECKING, Any, Optional, Type
from urllib.parse import urlparse

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Type checking import
if TYPE_CHECKING:
    from scrapegraph_py import Client


class ScrapegraphError(Exception):
    """Base exception for Scrapegraph-related errors"""


class RateLimitError(ScrapegraphError):
    """Raised when API rate limits are exceeded"""


class FixedScrapegraphScrapeToolSchema(BaseModel):
    """Input for ScrapegraphScrapeTool when website_url is fixed."""


class ScrapegraphScrapeToolSchema(FixedScrapegraphScrapeToolSchema):
    """Input for ScrapegraphScrapeTool."""

    website_url: str = Field(..., description="Mandatory website url to scrape")
    user_prompt: str = Field(
        default="Extract the main content of the webpage",
        description="Prompt to guide the extraction of content",
    )

    @field_validator("website_url")
    def validate_url(cls, v):
        """Validate URL format"""
        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError
            return v
        except Exception:
            raise ValueError(
                "Invalid URL format. URL must include scheme (http/https) and domain"
            )


class ScrapegraphScrapeTool(BaseTool):
    """
    A tool that uses Scrapegraph AI to intelligently scrape website content.

    Raises:
        ValueError: If API key is missing or URL format is invalid
        RateLimitError: If API rate limits are exceeded
        RuntimeError: If scraping operation fails
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Scrapegraph website scraper"
    description: str = (
        "A tool that uses Scrapegraph AI to intelligently scrape website content."
    )
    args_schema: Type[BaseModel] = ScrapegraphScrapeToolSchema
    website_url: Optional[str] = None
    user_prompt: Optional[str] = None
    api_key: Optional[str] = None
    enable_logging: bool = False
    _client: Optional["Client"] = None

    def __init__(
        self,
        website_url: Optional[str] = None,
        user_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_logging: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            from scrapegraph_py import Client
            from scrapegraph_py.logger import sgai_logger

        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'scrapegraph-py' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "scrapegraph-py"], check=True)
                from scrapegraph_py import Client
                from scrapegraph_py.logger import sgai_logger

            else:
                raise ImportError(
                    "`scrapegraph-py` package not found, please run `uv add scrapegraph-py`"
                )

        self._client = Client(api_key=api_key)

        self.api_key = api_key or os.getenv("SCRAPEGRAPH_API_KEY")

        if not self.api_key:
            raise ValueError("Scrapegraph API key is required")

        if website_url is not None:
            self._validate_url(website_url)
            self.website_url = website_url
            self.description = f"A tool that uses Scrapegraph AI to intelligently scrape {website_url}'s content."
            self.args_schema = FixedScrapegraphScrapeToolSchema

        if user_prompt is not None:
            self.user_prompt = user_prompt

        # Configure logging only if enabled
        if self.enable_logging:
            sgai_logger.set_logging(level="INFO")

    @staticmethod
    def _validate_url(url: str) -> None:
        """Validate URL format"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError
        except Exception:
            raise ValueError(
                "Invalid URL format. URL must include scheme (http/https) and domain"
            )

    def _handle_api_response(self, response: dict) -> str:
        """Handle and validate API response"""
        if not response:
            raise RuntimeError("Empty response from Scrapegraph API")

        if "error" in response:
            error_msg = response.get("error", {}).get("message", "Unknown error")
            if "rate limit" in error_msg.lower():
                raise RateLimitError(f"Rate limit exceeded: {error_msg}")
            raise RuntimeError(f"API error: {error_msg}")

        if "result" not in response:
            raise RuntimeError("Invalid response format from Scrapegraph API")

        return response["result"]

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        website_url = kwargs.get("website_url", self.website_url)
        user_prompt = (
            kwargs.get("user_prompt", self.user_prompt)
            or "Extract the main content of the webpage"
        )

        if not website_url:
            raise ValueError("website_url is required")

        # Validate URL format
        self._validate_url(website_url)

        try:
            # Make the SmartScraper request
            response = self._client.smartscraper(
                website_url=website_url,
                user_prompt=user_prompt,
            )

            return response

        except RateLimitError:
            raise  # Re-raise rate limit errors
        except Exception as e:
            raise RuntimeError(f"Scraping failed: {str(e)}")
        finally:
            # Always close the client
            self._client.close()
