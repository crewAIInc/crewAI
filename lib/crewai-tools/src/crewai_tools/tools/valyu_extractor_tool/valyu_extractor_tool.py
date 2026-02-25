import asyncio
import json
import os
from typing import Any, Literal, Union

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()
try:
    from valyu import Valyu  # type: ignore[import-untyped]

    VALYU_AVAILABLE = True
except ImportError:
    VALYU_AVAILABLE = False
    Valyu = Any


class ValyuExtractorToolSchema(BaseModel):
    """Input schema for ValyuExtractorTool."""

    urls: list[str] | str = Field(
        ...,
        description="The URL(s) to extract data from. Can be a single URL or a list of URLs (max 10).",
    )


class ValyuExtractorTool(BaseTool):
    """Tool that uses the Valyu API to extract clean, structured content from web pages.

    Attributes:
        client: An instance of Valyu client.
        name: The name of the tool.
        description: The description of the tool.
        args_schema: The schema for the tool's arguments.
        api_key: The Valyu API key.
        response_length: Content length per result.
        extract_effort: Processing quality level.
        screenshot: Whether to request page screenshots.
        summary: Enable AI-powered summarization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any | None = None
    name: str = "Valyu Extractor"
    description: str = (
        "Extracts clean, structured content from one or more web pages using the Valyu API. "
        "Returns structured data including title, content, and metadata."
    )
    args_schema: type[BaseModel] = ValyuExtractorToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("VALYU_API_KEY"),
        description="The Valyu API key. If not provided, it will be loaded from the environment variable VALYU_API_KEY.",
    )
    response_length: Literal["short", "medium", "large", "max"] = Field(
        default="short",
        description="Content length per result. 'short' (25K chars), 'medium' (50K), 'large' (100K), or 'max' (unlimited).",
    )
    extract_effort: Literal["normal", "high", "auto"] = Field(
        default="normal",
        description="Processing quality level. 'normal' for fastest, 'high' for better quality, 'auto' for automatic selection.",
    )
    screenshot: bool = Field(
        default=False,
        description="Whether to request page screenshots as pre-signed URLs.",
    )
    summary: Union[bool, str] = Field(
        default=False,
        description="Enable AI-powered summarization. Pass True for default summary, or a string with custom instructions.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["valyu"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="VALYU_API_KEY",
                description="API key for Valyu extraction service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        """Initializes the ValyuExtractorTool.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if VALYU_AVAILABLE:
            self.client = Valyu(api_key=self.api_key)
        else:
            try:
                import subprocess

                import click
            except ImportError:
                raise ImportError(
                    "The 'valyu' package is required. 'click' and 'subprocess' are also needed to assist with installation if the package is missing. "
                    "Please install 'valyu' manually (e.g., 'pip install valyu') and ensure 'click' and 'subprocess' are available."
                ) from None

            if click.confirm(
                "You are missing the 'valyu' package, which is required for ValyuExtractorTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "valyu"], check=True)  # noqa: S607
                    raise ImportError(
                        "'valyu' has been installed. Please restart your Python application to use the ValyuExtractorTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'valyu' but failed: {e}. "
                        f"Please install it manually to use the ValyuExtractorTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'valyu' package is required to use the ValyuExtractorTool. "
                    "Please install it with: pip install valyu"
                )

    def _run(
        self,
        urls: list[str] | str,
    ) -> str:
        """Synchronously extracts content from the given URL(s).

        Args:
            urls: The URL(s) to extract data from. Maximum 10 URLs.

        Returns:
            A JSON string containing the extracted data.
        """
        if not self.client:
            raise ValueError(
                "Valyu client is not initialized. Ensure 'valyu' is installed and API key is set."
            )

        # Ensure urls is a list
        if isinstance(urls, str):
            urls = [urls]

        # Build contents parameters
        contents_params: dict[str, Any] = {
            "urls": urls,
            "response_length": self.response_length,
            "extract_effort": self.extract_effort,
            "screenshot": self.screenshot,
        }

        # Add summary if set
        if self.summary:
            contents_params["summary"] = self.summary

        raw_results = self.client.contents(**contents_params)

        # Convert response to dict if it's a Pydantic model
        if hasattr(raw_results, "model_dump"):
            raw_results = raw_results.model_dump()
        elif hasattr(raw_results, "dict"):
            raw_results = raw_results.dict()

        return json.dumps(raw_results, indent=2)

    async def _arun(
        self,
        urls: list[str] | str,
    ) -> str:
        """Asynchronously extracts content from the given URL(s).

        Note: The Valyu SDK currently uses synchronous calls, so this method
        runs the synchronous implementation in a thread pool to avoid blocking
        the event loop.

        Args:
            urls: The URL(s) to extract data from. Maximum 10 URLs.

        Returns:
            A JSON string containing the extracted data.
        """
        return await asyncio.to_thread(self._run, urls)
