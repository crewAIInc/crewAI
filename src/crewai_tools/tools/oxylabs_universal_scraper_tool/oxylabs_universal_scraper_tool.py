import json
import os
from importlib.metadata import version
from platform import architecture, python_version
from typing import Any, List, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

try:
    from oxylabs import RealtimeClient
    from oxylabs.sources.response import Response as OxylabsResponse

    OXYLABS_AVAILABLE = True
except ImportError:
    RealtimeClient = Any
    OxylabsResponse = Any

    OXYLABS_AVAILABLE = False

__all__ = ["OxylabsUniversalScraperTool", "OxylabsUniversalScraperConfig"]


class OxylabsUniversalScraperArgs(BaseModel):
    url: str = Field(description="Website URL")


class OxylabsUniversalScraperConfig(BaseModel):
    """
    Universal Scraper configuration options:
    https://developers.oxylabs.io/scraper-apis/web-scraper-api/other-websites
    """

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


class OxylabsUniversalScraperTool(BaseTool):
    """
    Scrape any website with OxylabsUniversalScraperTool.

    Get Oxylabs account:
    https://dashboard.oxylabs.io/en

    Args:
        username (str): Oxylabs username.
        password (str): Oxylabs password.
        config: Configuration options. See ``OxylabsUniversalScraperConfig``
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    name: str = "Oxylabs Universal Scraper tool"
    description: str = "Scrape any url with Oxylabs Universal Scraper"
    args_schema: Type[BaseModel] = OxylabsUniversalScraperArgs

    oxylabs_api: RealtimeClient
    config: OxylabsUniversalScraperConfig
    package_dependencies: List[str] = ["oxylabs"]

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        config: OxylabsUniversalScraperConfig | dict = OxylabsUniversalScraperConfig(),
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
                    subprocess.run(["uv", "add", "oxylabs"], check=True)
                    from oxylabs import RealtimeClient

                    kwargs["oxylabs_api"] = RealtimeClient(
                        username=username,
                        password=password,
                        sdk_type=sdk_type,
                    )
                except subprocess.CalledProcessError:
                    raise ImportError("Failed to install oxylabs package")
            else:
                raise ImportError(
                    "`oxylabs` package not found, please run `uv add oxylabs`"
                )

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

    def _run(self, url: str) -> str:
        response = self.oxylabs_api.universal.scrape_url(
            url,
            **self.config.model_dump(exclude_none=True),
        )

        content = response.results[0].content

        if isinstance(content, dict):
            return json.dumps(content)

        return content
