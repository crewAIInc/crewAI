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


class TavilyCrawlToolSchema(BaseModel):
    """Input schema for TavilyCrawlTool."""

    url: str = Field(
        ...,
        description="The base URL to start crawling from.",
    )
    max_depth: int | None = Field(
        default=None,
        description="Maximum depth to crawl (number of link hops from the base URL, 1-5).",
        ge=1,
        le=5,
    )
    extract_depth: Literal["basic", "advanced"] | None = Field(
        default=None,
        description="Extraction depth for page content - 'basic' for main content, 'advanced' for comprehensive extraction.",
    )
    instructions: str | None = Field(
        default=None,
        description="Natural language instructions to guide the crawler (e.g., 'only crawl blog posts', 'focus on product pages').",
    )
    allow_external: bool | None = Field(
        default=None,
        description="Whether to allow crawling external domains.",
    )


class TavilyCrawlTool(BaseTool):
    """Tool that uses the Tavily API to crawl websites starting from a base URL.

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
        include_images: Whether to include images in the crawl results.
        format: The format of the extracted content.
        timeout: The timeout for the crawl request in seconds.
        include_favicon: Whether to include favicon URLs.
        include_usage: Whether to include credit usage information.
        chunks_per_source: Maximum number of content chunks per source.
        extra_kwargs: Additional kwargs passed directly to tavily-python's crawl() method.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: TavilyClient | None = None
    async_client: AsyncTavilyClient | None = None
    name: str = "Tavily Crawl"
    description: str = (
        "Crawl a website starting from a base URL to discover and extract content from multiple pages. "
        "Intelligently traverses links and extracts structured data at scale."
    )
    args_schema: type[BaseModel] = TavilyCrawlToolSchema
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
        description="Regex patterns to select crawling to specific domains or subdomains.",
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
        description="Whether to include images in the crawl results.",
    )
    format: Literal["markdown", "text"] | None = Field(
        default=None,
        description="The format of the extracted web page content.",
    )
    timeout: float = Field(
        default=150,
        description="The timeout for the crawl request in seconds.",
    )
    include_favicon: bool | None = Field(
        default=None,
        description="Whether to include favicon URLs in the crawl results.",
    )
    include_usage: bool | None = Field(
        default=None,
        description="Whether to include credit usage information in the response.",
    )
    chunks_per_source: int | None = Field(
        default=None,
        description="Maximum number of content chunks per source (1-5). Only used when instructions are provided.",
    )
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass directly to tavily-python's crawl() method. "
        "Use this for new tavily-python parameters not yet explicitly supported.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["tavily-python"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TAVILY_API_KEY",
                description="API key for Tavily crawl service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        """Initializes the TavilyCrawlTool.

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
                "You are missing the 'tavily-python' package, which is required for TavilyCrawlTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "tavily-python"], check=True)  # noqa: S607
                    raise ImportError(
                        "'tavily-python' has been installed. Please restart your Python application to use the TavilyCrawlTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'tavily-python' but failed: {e}. "
                        f"Please install it manually to use the TavilyCrawlTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'tavily-python' package is required to use the TavilyCrawlTool. "
                    "Please install it with: uv add tavily-python"
                )

    def _run(
        self,
        url: str,
        max_depth: int | None = None,
        extract_depth: Literal["basic", "advanced"] | None = None,
        instructions: str | None = None,
        allow_external: bool | None = None,
    ) -> str:
        """Synchronously crawls a website starting from the given URL.

        Args:
            url: The base URL to start crawling from.
            max_depth: Maximum depth to crawl (overrides class default if provided).
            extract_depth: Extraction depth (overrides class default if provided).
            instructions: Natural language instructions to guide the crawler.
            allow_external: Whether to allow crawling external domains.

        Returns:
            A JSON string containing the crawl results.
        """
        if not self.client:
            raise ValueError(
                "Tavily client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        return json.dumps(
            self.client.crawl(
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
                extract_depth=extract_depth,
                format=self.format,
                timeout=self.timeout,
                include_favicon=self.include_favicon,
                include_usage=self.include_usage,
                chunks_per_source=self.chunks_per_source,
                **self.extra_kwargs,
            ),
            indent=2,
        )

    async def _arun(
        self,
        url: str,
        max_depth: int | None = None,
        extract_depth: Literal["basic", "advanced"] | None = None,
        instructions: str | None = None,
        allow_external: bool | None = None,
    ) -> str:
        """Asynchronously crawls a website starting from the given URL.

        Args:
            url: The base URL to start crawling from.
            max_depth: Maximum depth to crawl (overrides class default if provided).
            extract_depth: Extraction depth (overrides class default if provided).
            instructions: Natural language instructions to guide the crawler.
            allow_external: Whether to allow crawling external domains.

        Returns:
            A JSON string containing the crawl results.
        """
        if not self.async_client:
            raise ValueError(
                "Tavily async client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        results = await self.async_client.crawl(
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
            extract_depth=extract_depth,
            format=self.format,
            timeout=self.timeout,
            include_favicon=self.include_favicon,
            include_usage=self.include_usage,
            chunks_per_source=self.chunks_per_source,
            **self.extra_kwargs,
        )
        return json.dumps(results, indent=2)

