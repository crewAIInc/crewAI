import json
import os
from collections.abc import Sequence
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()
try:
    from tavily import AsyncTavilyClient, TavilyClient  # type: ignore[import-untyped]

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = Any
    AsyncTavilyClient = Any


class TavilyMapToolSchema(BaseModel):
    """Input schema for TavilyMapTool."""

    url: str = Field(
        ...,
        description="The base URL to start mapping from.",
    )
    max_depth: int | None = Field(
        default=None,
        description="Maximum depth to map (number of link hops from the base URL, 1-5).",
        ge=1,
        le=5,
    )
    instructions: str | None = Field(
        default=None,
        description="Natural language instructions to guide the mapping (e.g., 'focus on documentation pages', 'skip API references').",
    )
    allow_external: bool | None = Field(
        default=None,
        description="Whether to allow mapping external domains.",
    )


class TavilyMapTool(BaseTool):
    """Tool that uses the Tavily API to map website structure starting from a base URL.

    Attributes:
        client: Synchronous Tavily client.
        async_client: Asynchronous Tavily client.
        name: The name of the tool.
        description: The description of the tool.
        args_schema: The schema for the tool's arguments.
        api_key: The Tavily API key.
        proxies: Optional proxies for the API requests.
        max_breadth: Maximum number of links to follow per page.
        limit: Total number of links to process before stopping.
        select_paths: Regex patterns to select specific paths.
        select_domains: Regex patterns to select specific domains.
        exclude_paths: Regex patterns to exclude specific paths.
        exclude_domains: Regex patterns to exclude specific domains.
        include_images: Whether to include images in the map results.
        timeout: The timeout for the map request in seconds.
        include_usage: Whether to include credit usage information.
        extra_kwargs: Additional kwargs passed directly to tavily-python's map() method.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: TavilyClient | None = None
    async_client: AsyncTavilyClient | None = None
    name: str = "Tavily Map"
    description: str = (
        "Map the structure of a website starting from a base URL. "
        "Discovers pages, links, and site hierarchy without extracting full content. "
        "Ideal for understanding site architecture."
    )
    args_schema: type[BaseModel] = TavilyMapToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"),
        description="The Tavily API key. If not provided, it will be loaded from the environment variable TAVILY_API_KEY.",
    )
    proxies: dict[str, str] | None = Field(
        default=None,
        description="Optional proxies to use for the Tavily API requests.",
    )
    max_breadth: int | None = Field(
        default=None,
        description="Maximum number of links to follow per page.",
    )
    limit: int | None = Field(
        default=None,
        description="Total number of links to process before stopping.",
    )
    select_paths: Sequence[str] | None = Field(
        default=None,
        description="Regex patterns to select only URLs with specific path patterns.",
    )
    select_domains: Sequence[str] | None = Field(
        default=None,
        description="Regex patterns to select mapping to specific domains or subdomains.",
    )
    exclude_paths: Sequence[str] | None = Field(
        default=None,
        description="Regex patterns to exclude URLs with specific path patterns.",
    )
    exclude_domains: Sequence[str] | None = Field(
        default=None,
        description="Regex patterns to exclude specific domains or subdomains.",
    )
    include_images: bool | None = Field(
        default=None,
        description="Whether to include images in the map results.",
    )
    timeout: float = Field(
        default=150,
        description="The timeout for the map request in seconds.",
    )
    include_usage: bool | None = Field(
        default=None,
        description="Whether to include credit usage information in the response.",
    )
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass directly to tavily-python's map() method. "
        "Use this for new tavily-python parameters not yet explicitly supported.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["tavily-python"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TAVILY_API_KEY",
                description="API key for Tavily map service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        """Initializes the TavilyMapTool.

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
                import subprocess

                import click
            except ImportError:
                raise ImportError(
                    "The 'tavily-python' package is required. 'click' and 'subprocess' are also needed to assist with installation if the package is missing. "
                    "Please install 'tavily-python' manually (e.g., 'uv add tavily-python') and ensure 'click' and 'subprocess' are available."
                ) from None

            if click.confirm(
                "You are missing the 'tavily-python' package, which is required for TavilyMapTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "tavily-python"], check=True)  # noqa: S607
                    raise ImportError(
                        "'tavily-python' has been installed. Please restart your Python application to use the TavilyMapTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'tavily-python' but failed: {e}. "
                        f"Please install it manually to use the TavilyMapTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'tavily-python' package is required to use the TavilyMapTool. "
                    "Please install it with: uv add tavily-python"
                )

    def _run(
        self,
        url: str,
        max_depth: int | None = None,
        instructions: str | None = None,
        allow_external: bool | None = None,
    ) -> str:
        """Synchronously maps a website structure starting from the given URL.

        Args:
            url: The base URL to start mapping from.
            max_depth: Maximum depth to map (overrides class default if provided).
            instructions: Natural language instructions to guide the mapping.
            allow_external: Whether to allow mapping external domains.

        Returns:
            A JSON string containing the map results.
        """
        if not self.client:
            raise ValueError(
                "Tavily client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        return json.dumps(
            self.client.map(
                url=url,
                max_depth=max_depth,
                max_breadth=self.max_breadth,
                limit=self.limit,
                instructions=instructions,
                select_paths=self.select_paths,
                select_domains=self.select_domains,
                exclude_paths=self.exclude_paths,
                exclude_domains=self.exclude_domains,
                allow_external=allow_external,
                include_images=self.include_images,
                timeout=self.timeout,
                include_usage=self.include_usage,
                **self.extra_kwargs,
            ),
            indent=2,
        )

    async def _arun(
        self,
        url: str,
        max_depth: int | None = None,
        instructions: str | None = None,
        allow_external: bool | None = None,
    ) -> str:
        """Asynchronously maps a website structure starting from the given URL.

        Args:
            url: The base URL to start mapping from.
            max_depth: Maximum depth to map (overrides class default if provided).
            instructions: Natural language instructions to guide the mapping.
            allow_external: Whether to allow mapping external domains.

        Returns:
            A JSON string containing the map results.
        """
        if not self.async_client:
            raise ValueError(
                "Tavily async client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        results = await self.async_client.map(
            url=url,
            max_depth=max_depth,
            max_breadth=self.max_breadth,
            limit=self.limit,
            instructions=instructions,
            select_paths=self.select_paths,
            select_domains=self.select_domains,
            exclude_paths=self.exclude_paths,
            exclude_domains=self.exclude_domains,
            allow_external=allow_external,
            include_images=self.include_images,
            timeout=self.timeout,
            include_usage=self.include_usage,
            **self.extra_kwargs,
        )
        return json.dumps(results, indent=2)

