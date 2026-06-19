from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
import json
import os
from typing import Any, Literal, cast

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


load_dotenv()
try:
    from tavily import (  # type: ignore[import-untyped, import-not-found, unused-ignore]
        AsyncTavilyClient,
        TavilyClient,
    )

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


class TavilyResearchToolSchema(BaseModel):
    """Input schema for TavilyResearchTool."""

    input: str = Field(
        ...,
        description="The research task or question to investigate.",
    )
    model: Literal["mini", "pro", "auto"] = Field(
        default="auto",
        description="The model used by the Tavily research agent.",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema that structures the research output.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream research progress and results as SSE chunks.",
    )
    citation_format: Literal["numbered", "mla", "apa", "chicago"] = Field(
        default="numbered",
        description="Citation format for the research report.",
    )


class TavilyResearchTool(BaseTool):
    """Tool that uses the Tavily Research API to create research tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _client: Any | None = PrivateAttr(default=None)
    _async_client: Any | None = PrivateAttr(default=None)
    name: str = "Tavily Research"
    description: str = (
        "A tool that creates Tavily research tasks and can stream research "
        "progress and results. It returns Tavily responses as JSON or SSE chunks."
    )
    args_schema: type[BaseModel] = TavilyResearchToolSchema
    model: Literal["mini", "pro", "auto"] = Field(
        default="auto",
        description="Default model used for new Tavily research tasks.",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Default JSON Schema used to structure research output.",
    )
    stream: bool = Field(
        default=False,
        description="Whether new Tavily research tasks should stream responses by default.",
    )
    citation_format: Literal["numbered", "mla", "apa", "chicago"] = Field(
        default="numbered",
        description="Default citation format for Tavily research results.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["tavily-python"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TAVILY_API_KEY",
                description="API key for Tavily research service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if TAVILY_AVAILABLE:
            api_key = os.getenv("TAVILY_API_KEY")
            self._client = TavilyClient(api_key=api_key)
            self._async_client = AsyncTavilyClient(api_key=api_key)
        else:
            try:
                import subprocess

                import click
            except ImportError as e:
                raise ImportError(
                    "The 'tavily-python' package is required. 'click' and "
                    "'subprocess' are also needed to assist with installation "
                    "if the package is missing. Please install 'tavily-python' "
                    "manually (e.g., 'pip install tavily-python') and ensure "
                    "'click' and 'subprocess' are available."
                ) from e

            if click.confirm(
                "You are missing the 'tavily-python' package, which is required "
                "for TavilyResearchTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "tavily-python"], check=True)  # noqa: S607
                    raise ImportError(
                        "'tavily-python' has been installed. Please restart your "
                        "Python application to use the TavilyResearchTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'tavily-python' but failed: {e}. "
                        "Please install it manually to use the TavilyResearchTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'tavily-python' package is required to use the "
                    "TavilyResearchTool. Please install it with: uv add tavily-python"
                )

    @staticmethod
    def _stringify_response(response: Any) -> str:
        if isinstance(response, str):
            return response
        return json.dumps(response, indent=2)

    def _run(
        self,
        input: str,
        model: Literal["mini", "pro", "auto"] | None = None,
        output_schema: dict[str, Any] | None = None,
        stream: bool | None = None,
        citation_format: Literal["numbered", "mla", "apa", "chicago"] | None = None,
    ) -> str | Generator[bytes, None, None]:
        """Synchronously creates Tavily research tasks or streams results."""
        if not self._client:
            raise ValueError(
                "Tavily client is not initialized. Ensure 'tavily-python' is "
                "installed and API key is set."
            )

        use_stream = self.stream if stream is None else stream
        result = self._client.research(
            input=input,
            model=self.model if model is None else model,
            output_schema=self.output_schema
            if output_schema is None
            else output_schema,
            stream=use_stream,
            citation_format=(
                self.citation_format if citation_format is None else citation_format
            ),
        )

        if use_stream:
            return cast(Generator[bytes, None, None], result)

        return self._stringify_response(result)

    async def _arun(
        self,
        input: str,
        model: Literal["mini", "pro", "auto"] | None = None,
        output_schema: dict[str, Any] | None = None,
        stream: bool | None = None,
        citation_format: Literal["numbered", "mla", "apa", "chicago"] | None = None,
    ) -> str | AsyncGenerator[bytes, None]:
        """Asynchronously creates Tavily research tasks or streams results."""
        if not self._async_client:
            raise ValueError(
                "Tavily async client is not initialized. Ensure 'tavily-python' is "
                "installed and API key is set."
            )

        use_stream = self.stream if stream is None else stream
        result = await self._async_client.research(
            input=input,
            model=self.model if model is None else model,
            output_schema=self.output_schema
            if output_schema is None
            else output_schema,
            stream=use_stream,
            citation_format=(
                self.citation_format if citation_format is None else citation_format
            ),
        )

        if use_stream:
            return cast(AsyncGenerator[bytes, None], result)

        return self._stringify_response(result)
