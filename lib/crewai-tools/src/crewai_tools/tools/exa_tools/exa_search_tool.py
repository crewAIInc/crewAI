from __future__ import annotations

from builtins import type as type_
import os
from typing import Any, TypedDict
import warnings

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Required


try:
    from exa_py import Exa
except ImportError:
    Exa = None  # type: ignore[assignment,misc]


class SearchParams(TypedDict, total=False):
    """Parameters for Exa search API."""

    type: Required[str | None]
    start_published_date: str
    end_published_date: str
    include_domains: list[str]


class ExaBaseToolSchema(BaseModel):
    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )
    start_published_date: str | None = Field(
        None, description="Start date for the search"
    )
    end_published_date: str | None = Field(None, description="End date for the search")
    include_domains: list[str] | None = Field(
        None, description="List of domains to include in the search"
    )


EXABaseToolSchema = ExaBaseToolSchema


class ExaSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "ExaSearchTool"
    description: str = (
        "Search the web with Exa, the fastest and most accurate web search API."
    )
    args_schema: type_[BaseModel] = ExaBaseToolSchema
    client: Any | None = None
    content: bool | dict[str, Any] | None = False
    summary: bool | dict[str, Any] | None = False
    highlights: bool | dict[str, Any] | None = True
    type: str | None = "auto"
    package_dependencies: list[str] = Field(default_factory=lambda: ["exa_py"])
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("EXA_API_KEY"),
        description="API key for Exa services",
        json_schema_extra={"required": False},
    )
    base_url: str | None = Field(
        default_factory=lambda: os.getenv("EXA_BASE_URL"),
        description="API server url",
        json_schema_extra={"required": False},
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="EXA_API_KEY",
                description="API key for Exa services",
                required=False,
            ),
            EnvVar(
                name="EXA_BASE_URL",
                description="API url for the Exa services",
                required=False,
            ),
        ]
    )

    def __init__(
        self,
        content: bool | dict[str, Any] | None = False,
        summary: bool | dict[str, Any] | None = False,
        highlights: bool | dict[str, Any] | None = True,
        type: str | None = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
        )
        global Exa
        if Exa is None:
            import click

            if click.confirm(
                "You are missing the 'exa_py' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "exa_py"], check=True)  # noqa: S607

                from exa_py import Exa as _Exa

                Exa = _Exa  # type: ignore[misc]
            else:
                raise ImportError(
                    "You are missing the 'exa_py' package. Please install it to use ExaSearchTool."
                )

        client_kwargs: dict[str, str] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = Exa(**client_kwargs)
        self.client.headers["x-exa-integration"] = "crewai"
        self.content = content
        self.summary = summary
        self.highlights = highlights
        self.type = type

    def _run(
        self,
        search_query: str,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        include_domains: list[str] | None = None,
    ) -> Any:
        if self.client is None:
            raise ValueError("Client not initialized")

        search_params: SearchParams = {
            "type": self.type,
        }

        if start_published_date:
            search_params["start_published_date"] = start_published_date
        if end_published_date:
            search_params["end_published_date"] = end_published_date
        if include_domains:
            search_params["include_domains"] = include_domains

        contents_kwargs: dict[str, Any] = {}
        if self.content:
            contents_kwargs["text"] = self.content
        if self.highlights:
            contents_kwargs["highlights"] = self.highlights
        if self.summary:
            contents_kwargs["summary"] = self.summary

        if contents_kwargs:
            return self.client.search_and_contents(
                search_query, **contents_kwargs, **search_params
            )
        return self.client.search(search_query, **search_params)


class EXASearchTool(ExaSearchTool):
    """Deprecated alias for :class:`ExaSearchTool`. Kept for backwards compatibility."""

    name: str = "ExaSearchTool"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "EXASearchTool is deprecated and will be removed in a future release; "
            "use ExaSearchTool instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
