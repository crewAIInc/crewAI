import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()
try:
    from seltz import Seltz  # type: ignore[import-untyped]

    SELTZ_AVAILABLE = True
except ImportError:
    SELTZ_AVAILABLE = False
    Seltz = Any


class SeltzSearchToolSchema(BaseModel):
    """Input schema for SeltzSearchTool."""

    query: str = Field(..., description="The search query string.")


class SeltzSearchTool(BaseTool):
    """Tool that uses the Seltz Web Knowledge API for AI-optimized web search.

    Attributes:
        client: An instance of the Seltz client.
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The Seltz API key.
        endpoint: Optional custom API endpoint.
        insecure: Whether to use an insecure connection.
        max_documents: Maximum number of documents to return.
        context: Background context to refine search results.
        profile: Named search configuration profile.
        max_content_length_per_result: Maximum length for content of each result.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any | None = None
    name: str = "Seltz Web Knowledge Search"
    description: str = (
        "Fast, source-backed web knowledge for AI reasoning. "
        "Returns context-engineered web content with sources optimized for LLMs and AI agents."
    )
    args_schema: type[BaseModel] = SeltzSearchToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SELTZ_API_KEY"),
        description="The Seltz API key. If not provided, it will be loaded from the environment variable SELTZ_API_KEY.",
    )
    endpoint: str | None = Field(
        default=None,
        description="Optional custom Seltz API endpoint.",
    )
    insecure: bool = Field(
        default=False,
        description="Whether to use an insecure connection to the Seltz API.",
    )
    max_documents: int = Field(
        default=5, description="The maximum number of documents to return."
    )
    context: str | None = Field(
        default=None,
        description="Background context to refine search results.",
    )
    profile: str | None = Field(
        default=None,
        description="Named search configuration profile.",
    )
    max_content_length_per_result: int = Field(
        default=1000,
        description="Maximum length for the 'content' of each search result to avoid context window issues.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["seltz"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SELTZ_API_KEY",
                description="API key for Seltz Web Knowledge service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if SELTZ_AVAILABLE:
            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.endpoint is not None:
                client_kwargs["endpoint"] = self.endpoint
            if self.insecure:
                client_kwargs["insecure"] = self.insecure
            self.client = Seltz(**client_kwargs)
        else:
            try:
                import subprocess

                import click
            except ImportError as e:
                raise ImportError(
                    "The 'seltz' package is required. 'click' and 'subprocess' are also needed to assist with installation if the package is missing. "
                    "Please install 'seltz' manually (e.g., 'pip install seltz') and ensure 'click' and 'subprocess' are available."
                ) from e

            if click.confirm(
                "You are missing the 'seltz' package, which is required for SeltzSearchTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "seltz"], check=True)  # noqa: S607
                    raise ImportError(
                        "'seltz' has been installed. Please restart your Python application to use the SeltzSearchTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'seltz' but failed: {e}. "
                        f"Please install it manually to use the SeltzSearchTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'seltz' package is required to use the SeltzSearchTool. "
                    "Please install it with: uv add seltz"
                )

    def _run(
        self,
        query: str,
    ) -> str:
        """Synchronously performs a search using the Seltz Web Knowledge API.
        Content of each result is truncated to `max_content_length_per_result`.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results with truncated content.
        """
        if not self.client:
            raise ValueError(
                "Seltz client is not initialized. Ensure 'seltz' is installed and API key is set."
            )

        from seltz import Includes  # type: ignore[import-untyped]

        search_kwargs: dict[str, Any] = {
            "query": query,
            "includes": Includes(max_documents=self.max_documents),
        }
        if self.context is not None:
            search_kwargs["context"] = self.context
        if self.profile is not None:
            search_kwargs["profile"] = self.profile

        try:
            response = self.client.search(**search_kwargs)
        except Exception as e:
            return json.dumps({"error": f"Seltz search failed: {e}"})

        results = []
        for doc in response.documents:
            content = doc.content
            if len(content) > self.max_content_length_per_result:
                content = content[: self.max_content_length_per_result] + "..."
            results.append({"url": doc.url, "content": content})

        return json.dumps(results, indent=2)

    async def _arun(
        self,
        query: str,
    ) -> str:
        """Asynchronously performs a search using the Seltz Web Knowledge API.
        Delegates to the synchronous implementation as the Seltz SDK uses gRPC.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results with truncated content.
        """
        import asyncio

        return await asyncio.to_thread(self._run, query=query)
