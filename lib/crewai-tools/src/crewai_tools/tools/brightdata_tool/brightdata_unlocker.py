import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class BrightDataConfig(BaseModel):
    API_URL: str = "https://api.brightdata.com/request"

    @classmethod
    def from_env(cls):
        return cls(
            API_URL=os.environ.get(
                "BRIGHTDATA_API_URL", "https://api.brightdata.com/request"
            )
        )


class BrightDataUnlockerToolSchema(BaseModel):
    """Pydantic schema for input parameters used by the BrightDataWebUnlockerTool.

    This schema defines the structure and validation for parameters passed when performing
    a web scraping request using Bright Data's Web Unlocker.

    Attributes:
        url (str): The target URL to scrape.
        format (Optional[str]): Format of the response returned by Bright Data. Default 'raw' format.
        data_format (Optional[str]): Response data format (html by default). markdown is one more option.
    """

    url: str = Field(..., description="URL to perform the web scraping")
    format: str | None = Field(
        default="raw", description="Response format (raw is standard)"
    )
    data_format: str | None = Field(
        default="markdown", description="Response data format (html by default)"
    )


class BrightDataWebUnlockerTool(BaseTool):
    """A tool for performing web scraping using the Bright Data Web Unlocker API.

    This tool allows automated and programmatic access to web pages by routing requests
    through Bright Data's unlocking and proxy infrastructure, which can bypass bot
    protection mechanisms like CAPTCHA, geo-restrictions, and anti-bot detection.

    Attributes:
        name (str): Name of the tool.
        description (str): Description of what the tool does.
        args_schema (Type[BaseModel]): Pydantic model schema for expected input arguments.
        base_url (str): Base URL of the Bright Data Web Unlocker API.
        api_key (str): Bright Data API key (must be set in the BRIGHT_DATA_API_KEY environment variable).
        zone (str): Bright Data zone identifier (must be set in the BRIGHT_DATA_ZONE environment variable).

    Methods:
        _run(**kwargs: Any) -> Any:
            Sends a scraping request to Bright Data's Web Unlocker API and returns the result.
    """

    name: str = "Bright Data Web Unlocker Scraping"
    description: str = "Tool to perform web scraping using Bright Data Web Unlocker"
    args_schema: type[BaseModel] = BrightDataUnlockerToolSchema
    _config = BrightDataConfig.from_env()
    base_url: str = ""
    api_key: str = ""
    zone: str = ""
    url: str | None = None
    format: str = "raw"
    data_format: str = "markdown"
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BRIGHT_DATA_API_KEY",
                description="API key for Bright Data",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        url: str | None = None,
        format: str = "raw",
        data_format: str = "markdown",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.base_url = self._config.API_URL
        self.url = url
        self.format = format
        self.data_format = data_format

        self.api_key = os.getenv("BRIGHT_DATA_API_KEY") or ""
        self.zone = os.getenv("BRIGHT_DATA_ZONE") or ""
        if not self.api_key:
            raise ValueError("BRIGHT_DATA_API_KEY environment variable is required.")
        if not self.zone:
            raise ValueError("BRIGHT_DATA_ZONE environment variable is required.")

    def _run(
        self,
        url: str | None = None,
        format: str | None = None,
        data_format: str | None = None,
        **kwargs: Any,
    ) -> Any:
        url = url or self.url
        format = format or self.format
        data_format = data_format or self.data_format

        if not url:
            raise ValueError("url is required either in constructor or method call")

        payload = {
            "url": url,
            "zone": self.zone,
            "format": format,
        }
        valid_data_formats = {"html", "markdown"}
        if data_format not in valid_data_formats:
            raise ValueError(
                f"Unsupported data format: {data_format}. Must be one of {', '.join(valid_data_formats)}."
            )

        if data_format == "markdown":
            payload["data_format"] = "markdown"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.base_url, json=payload, headers=headers, timeout=30
            )
            response.raise_for_status()

            return response.text

        except requests.RequestException as e:
            return f"HTTP Error performing BrightData Web Unlocker Scrape: {e}\nResponse: {getattr(e.response, 'text', '')}"
        except Exception as e:
            return f"Error fetching results: {e!s}"
