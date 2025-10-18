import os
from typing import Any, Optional

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


try:
    from exa_py import Exa

    EXA_INSTALLED = True
except ImportError:
    Exa = Any
    EXA_INSTALLED = False


class EXABaseToolSchema(BaseModel):
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


class EXASearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "EXASearchTool"
    description: str = "Search the internet using Exa"
    args_schema: type[BaseModel] = EXABaseToolSchema
    client: Optional["Exa"] = None
    content: bool | None = False
    summary: bool | None = False
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
        content: bool | None = False,
        summary: bool | None = False,
        type: str | None = "auto",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        if not EXA_INSTALLED:
            import click

            if click.confirm(
                "You are missing the 'exa_py' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "exa_py"], check=True)  # noqa: S607

            else:
                raise ImportError(
                    "You are missing the 'exa_py' package. Would you like to install it?"
                )
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = Exa(**client_kwargs)
        self.content = content
        self.summary = summary
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

        search_params = {
            "type": self.type,
        }

        if start_published_date:
            search_params["start_published_date"] = start_published_date
        if end_published_date:
            search_params["end_published_date"] = end_published_date
        if include_domains:
            search_params["include_domains"] = include_domains

        if self.content:
            results = self.client.search_and_contents(
                search_query, summary=self.summary, **search_params
            )
        else:
            results = self.client.search(search_query, **search_params)
        return results
