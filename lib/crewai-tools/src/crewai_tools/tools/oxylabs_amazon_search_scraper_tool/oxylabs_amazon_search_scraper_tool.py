from importlib.metadata import version
import json
import os
from platform import architecture, python_version
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


try:
    from oxylabs import RealtimeClient  # type: ignore[import-untyped]
    from oxylabs.sources.response import (  # type: ignore[import-untyped]
        Response as OxylabsResponse,
    )

    OXYLABS_AVAILABLE = True
except ImportError:
    RealtimeClient = Any
    OxylabsResponse = Any

    OXYLABS_AVAILABLE = False


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
    context: list | None = Field(
        None,
        description="Additional advanced settings and controls for specialized requirements.",
    )
    parse: bool | None = Field(None, description="True will return structured data.")
    parsing_instructions: dict | None = Field(
        None, description="Instructions for parsing the results."
    )


class OxylabsAmazonSearchScraperTool(BaseTool):
    """Scrape Amazon search results with OxylabsAmazonSearchScraperTool.

    Get Oxylabs account:
    https://dashboard.oxylabs.io/en

    Args:
        username (str): Oxylabs username.
        password (str): Oxylabs password.
        config: Configuration options. See ``OxylabsAmazonSearchScraperConfig``
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    name: str = "Oxylabs Amazon Search Scraper tool"
    description: str = "Scrape Amazon search results with Oxylabs Amazon Search Scraper"
    args_schema: type[BaseModel] = OxylabsAmazonSearchScraperArgs

    oxylabs_api: RealtimeClient
    config: OxylabsAmazonSearchScraperConfig
    package_dependencies: list[str] = Field(default_factory=lambda: ["oxylabs"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OXYLABS_USERNAME",
                description="Username for Oxylabs",
                required=True,
            ),
            EnvVar(
                name="OXYLABS_PASSWORD",
                description="Password for Oxylabs",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        config: OxylabsAmazonSearchScraperConfig | dict | None = None,
        **kwargs,
    ):
        bits, _ = architecture()
        sdk_type = (
            f"oxylabs-crewai-sdk-python/"
            f"{version('crewai')} "
            f"({python_version()}; {bits})"
        )

        if username is None or password is None:
            username, password = self._get_credentials_from_env()

        if OXYLABS_AVAILABLE:
            # import RealtimeClient to make it accessible for the current scope
            from oxylabs import RealtimeClient

            kwargs["oxylabs_api"] = RealtimeClient(
                username=username,
                password=password,
                sdk_type=sdk_type,
            )
        else:
            import click

            if click.confirm(
                "You are missing the 'oxylabs' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "oxylabs"], check=True)  # noqa: S607
                    from oxylabs import RealtimeClient

                    kwargs["oxylabs_api"] = RealtimeClient(
                        username=username,
                        password=password,
                        sdk_type=sdk_type,
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install oxylabs package") from e
            else:
                raise ImportError(
                    "`oxylabs` package not found, please run `uv add oxylabs`"
                )

        if config is None:
            config = OxylabsAmazonSearchScraperConfig()
        super().__init__(config=config, **kwargs)

    def _get_credentials_from_env(self) -> tuple[str, str]:
        username = os.environ.get("OXYLABS_USERNAME")
        password = os.environ.get("OXYLABS_PASSWORD")
        if not username or not password:
            raise ValueError(
                "You must pass oxylabs username and password when instantiating the tool "
                "or specify OXYLABS_USERNAME and OXYLABS_PASSWORD environment variables"
            )
        return username, password

    def _run(self, query: str) -> str:
        response = self.oxylabs_api.amazon.scrape_search(
            query,
            **self.config.model_dump(exclude_none=True),
        )

        content = response.results[0].content

        if isinstance(content, dict):
            return json.dumps(content)

        return content
