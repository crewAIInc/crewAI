import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class YouContentsToolSchema(BaseModel):
    """Input for YouContentsTool."""

    urls: list[str] | str = Field(
        ...,
        description="The URL(s) to extract content from. Can be a single URL or a list of URLs.",
    )


class YouContentsTool(BaseTool):
    """A tool that extracts content from web pages using the You.com Contents API."""

    name: str = "You.com Contents Extractor"
    description: str = (
        "Extracts content from one or more web pages using the You.com Contents API. "
        "Returns structured data in markdown, HTML, or metadata format."
    )
    args_schema: type[BaseModel] = YouContentsToolSchema
    contents_url: str = "https://api.ydc-index.io/contents"
    formats: list[Literal["markdown", "html", "metadata"]] = Field(
        default_factory=lambda: ["markdown"],
    )
    crawl_timeout: int | None = 10
    timeout: int = 60
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="YOU_API_KEY",
                description="API key for You.com contents extraction service",
                required=True,
            ),
        ],
    )

    def __init__(self, *args, **kwargs):
        """Initialize the YouContentsTool.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)
        if "YOU_API_KEY" not in os.environ:
            raise ValueError(
                "YOU_API_KEY environment variable is required for YouContentsTool. "
                "Get your API key at https://you.com/platform/api-keys",
            )

    def _run(
        self,
        urls: list[str] | str,
    ) -> str:
        """Extract content from the given URL(s).

        Args:
            urls: The URL(s) to extract content from.

        Returns:
            JSON string containing the extracted content.
        """
        try:
            # Normalize urls to list
            url_list = [urls] if isinstance(urls, str) else urls

            if not url_list:
                raise ValueError("At least one URL is required")

            # Build request payload
            payload: dict[str, Any] = {
                "urls": url_list,
                "formats": self.formats,
            }

            # Add crawl timeout if specified (must be between 1-60)
            if self.crawl_timeout is not None:
                payload["crawl_timeout"] = min(max(self.crawl_timeout, 1), 60)

            # Setup request headers
            headers = {
                "X-API-Key": os.environ["YOU_API_KEY"],
                "Content-Type": "application/json",
            }

            # Make API request
            response = requests.post(
                self.contents_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json()

            return json.dumps(results, indent=2)

        except requests.RequestException as e:
            return f"Error extracting content: {e!s}"
        except ValueError as e:
            return f"Invalid parameters: {e!s}"
        except KeyError as e:
            return f"Error parsing API response: {e!s}"
