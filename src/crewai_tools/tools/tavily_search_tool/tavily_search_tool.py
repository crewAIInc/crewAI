from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type, Any, Union, Literal, Sequence
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


class TavilySearchToolSchema(BaseModel):
    """Input schema for TavilySearchTool."""

    query: str = Field(..., description="The search query string.")


class TavilySearchTool(BaseTool):
    """
    Tool that uses the Tavily Search API to perform web searches.

    Attributes:
        client: An instance of TavilyClient.
        async_client: An instance of AsyncTavilyClient.
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The Tavily API key.
        proxies: Optional proxies for the API requests.
        search_depth: The depth of the search.
        topic: The topic to focus the search on.
        time_range: The time range for the search.
        days: The number of days to search back.
        max_results: The maximum number of results to return.
        include_domains: A list of domains to include in the search.
        exclude_domains: A list of domains to exclude from the search.
        include_answer: Whether to include a direct answer to the query.
        include_raw_content: Whether to include the raw content of the search results.
        include_images: Whether to include images in the search results.
        timeout: The timeout for the search request in seconds.
        max_content_length_per_result: Maximum length for the 'content' of each search result.
    """

    model_config = {"arbitrary_types_allowed": True}
    client: Optional[TavilyClient] = None
    async_client: Optional[AsyncTavilyClient] = None
    name: str = "Tavily Search"
    description: str = (
        "A tool that performs web searches using the Tavily Search API. "
        "It returns a JSON object containing the search results."
    )
    args_schema: Type[BaseModel] = TavilySearchToolSchema
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"),
        description="The Tavily API key. If not provided, it will be loaded from the environment variable TAVILY_API_KEY.",
    )
    proxies: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional proxies to use for the Tavily API requests.",
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic", description="The depth of the search."
    )
    topic: Literal["general", "news", "finance"] = Field(
        default="general", description="The topic to focus the search on."
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None, description="The time range for the search."
    )
    days: int = Field(default=7, description="The number of days to search back.")
    max_results: int = Field(
        default=5, description="The maximum number of results to return."
    )
    include_domains: Optional[Sequence[str]] = Field(
        default=None, description="A list of domains to include in the search."
    )
    exclude_domains: Optional[Sequence[str]] = Field(
        default=None, description="A list of domains to exclude from the search."
    )
    include_answer: Union[bool, Literal["basic", "advanced"]] = Field(
        default=False, description="Whether to include a direct answer to the query."
    )
    include_raw_content: bool = Field(
        default=False,
        description="Whether to include the raw content of the search results.",
    )
    include_images: bool = Field(
        default=False, description="Whether to include images in the search results."
    )
    timeout: int = Field(
        default=60, description="The timeout for the search request in seconds."
    )
    max_content_length_per_result: int = Field(
        default=1000,
        description="Maximum length for the 'content' of each search result to avoid context window issues.",
    )

    def __init__(self, **kwargs: Any):
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
                    "Please install 'tavily-python' manually (e.g., 'pip install tavily-python') and ensure 'click' and 'subprocess' are available."
                )

            if click.confirm(
                "You are missing the 'tavily-python' package, which is required for TavilySearchTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "tavily-python"], check=True)
                    raise ImportError(
                        "'tavily-python' has been installed. Please restart your Python application to use the TavilySearchTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'tavily-python' but failed: {e}. "
                        f"Please install it manually to use the TavilySearchTool."
                    )
            else:
                raise ImportError(
                    "The 'tavily-python' package is required to use the TavilySearchTool. "
                    "Please install it with: uv add tavily-python"
                )

    def _run(
        self,
        query: str,
    ) -> str:
        """
        Synchronously performs a search using the Tavily API.
        Content of each result is truncated to `max_content_length_per_result`.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results with truncated content.
        """
        if not self.client:
            raise ValueError(
                "Tavily client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        raw_results = self.client.search(
            query=query,
            search_depth=self.search_depth,
            topic=self.topic,
            time_range=self.time_range,
            days=self.days,
            max_results=self.max_results,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            include_answer=self.include_answer,
            include_raw_content=self.include_raw_content,
            include_images=self.include_images,
            timeout=self.timeout,
        )

        if (
            isinstance(raw_results, dict)
            and "results" in raw_results
            and isinstance(raw_results["results"], list)
        ):
            for item in raw_results["results"]:
                if (
                    isinstance(item, dict)
                    and "content" in item
                    and isinstance(item["content"], str)
                ):
                    if len(item["content"]) > self.max_content_length_per_result:
                        item["content"] = (
                            item["content"][: self.max_content_length_per_result]
                            + "..."
                        )

        return json.dumps(raw_results, indent=2)

    async def _arun(
        self,
        query: str,
    ) -> str:
        """
        Asynchronously performs a search using the Tavily API.
        Content of each result is truncated to `max_content_length_per_result`.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results with truncated content.
        """
        if not self.async_client:
            raise ValueError(
                "Tavily async client is not initialized. Ensure 'tavily-python' is installed and API key is set."
            )

        raw_results = await self.async_client.search(
            query=query,
            search_depth=self.search_depth,
            topic=self.topic,
            time_range=self.time_range,
            days=self.days,
            max_results=self.max_results,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            include_answer=self.include_answer,
            include_raw_content=self.include_raw_content,
            include_images=self.include_images,
            timeout=self.timeout,
        )

        if (
            isinstance(raw_results, dict)
            and "results" in raw_results
            and isinstance(raw_results["results"], list)
        ):
            for item in raw_results["results"]:
                if (
                    isinstance(item, dict)
                    and "content" in item
                    and isinstance(item["content"], str)
                ):
                    if len(item["content"]) > self.max_content_length_per_result:
                        item["content"] = (
                            item["content"][: self.max_content_length_per_result]
                            + "..."
                        )

        return json.dumps(raw_results, indent=2)
