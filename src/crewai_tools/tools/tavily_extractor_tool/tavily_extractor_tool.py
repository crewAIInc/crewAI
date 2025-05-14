from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type, Any, Union, List, Literal
from dotenv import load_dotenv
import os
import json

load_dotenv()
try:
    from tavily import TavilyClient, AsyncTavilyClient

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = Any
    AsyncTavilyClient = Any


class TavilyExtractorToolSchema(BaseModel):
    """Input schema for TavilyExtractorTool."""

    urls: Union[List[str], str] = Field(
        ...,
        description="The URL(s) to extract data from. Can be a single URL or a list of URLs.",
    )


class TavilyExtractorTool(BaseTool):
    """
    Tool that uses the Tavily API to extract content from web pages.

    Attributes:
        client: Synchronous Tavily client.
        async_client: Asynchronous Tavily client.
        name: The name of the tool.
        description: The description of the tool.
        args_schema: The schema for the tool's arguments.
        api_key: The Tavily API key.
        proxies: Optional proxies for the API requests.
        include_images: Whether to include images in the extraction.
        extract_depth: The depth of extraction.
        timeout: The timeout for the extraction request in seconds.
    """

    model_config = {"arbitrary_types_allowed": True}
    client: Optional[TavilyClient] = None
    async_client: Optional[AsyncTavilyClient] = None
    name: str = "TavilyExtractorTool"
    description: str = "Extracts content from one or more web pages using the Tavily API. Returns structured data."
    args_schema: Type[BaseModel] = TavilyExtractorToolSchema
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"),
        description="The Tavily API key. If not provided, it will be loaded from the environment variable TAVILY_API_KEY.",
    )
    proxies: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional proxies to use for the Tavily API requests.",
    )
    include_images: bool = Field(
        default=False,
        description="Whether to include images in the extraction.",
    )
    extract_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="The depth of extraction. 'basic' for basic extraction, 'advanced' for advanced extraction.",
    )
    timeout: int = Field(
        default=60,
        description="The timeout for the extraction request in seconds.",
    )

    def __init__(self, **kwargs: Any):
        """
        Initializes the TavilyExtractorTool.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if TAVILY_AVAILABLE:
            self.client = TavilyClient(api_key=self.api_key, proxies=self.proxies)
            self.async_client = AsyncTavilyClient(
                api_key=self.api_key, proxies=self.proxies
            )
        else:
            try:
                import click
                import subprocess
            except ImportError:
                raise ImportError(
                    "The 'tavily-python' package is required. 'click' and 'subprocess' are also needed to assist with installation if the package is missing. "
                    "Please install 'tavily-python' manually (e.g., 'uv add tavily-python') and ensure 'click' and 'subprocess' are available."
                )

            if click.confirm(
                "You are missing the 'tavily-python' package, which is required for TavilyExtractorTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["pip", "install", "tavily-python"], check=True)
                    raise ImportError(
                        "'tavily-python' has been installed. Please restart your Python application to use the TavilyExtractorTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'tavily-python' but failed: {e}. "
                        f"Please install it manually to use the TavilyExtractorTool."
                    )
            else:
                raise ImportError(
                    "The 'tavily-python' package is required to use the TavilyExtractorTool. "
                    "Please install it with: uv add tavily-python"
                )

    def _run(
        self,
        urls: Union[List[str], str],
    ) -> str:
        """
        Synchronously extracts content from the given URL(s).

        Args:
            urls: The URL(s) to extract data from.

        Returns:
            A JSON string containing the extracted data.
        """
        if not self.client:
            raise ValueError(
                "Tavily client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        return json.dumps(
            self.client.extract(
                urls=urls,
                extract_depth=self.extract_depth,
                include_images=self.include_images,
                timeout=self.timeout,
            ),
            indent=2,
        )

    async def _arun(
        self,
        urls: Union[List[str], str],
    ) -> str:
        """
        Asynchronously extracts content from the given URL(s).

        Args:
            urls: The URL(s) to extract data from.

        Returns:
            A JSON string containing the extracted data.
        """
        if not self.async_client:
            raise ValueError(
                "Tavily async client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        results = await self.async_client.extract(
            urls=urls,
            extract_depth=self.extract_depth,
            include_images=self.include_images,
            timeout=self.timeout,
        )
        return json.dumps(results, indent=2)
