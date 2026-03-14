from datetime import date
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr

from crewai_tools.tools.igpt_email_tool.igpt_email_ask_tool import (
    _stringify_igpt_response,
)


class IgptEmailSearchInput(BaseModel):
    query: str = Field(..., description="Keyword or semantic query for email search.")
    date_from: date | None = Field(
        default=None, description="Start date (YYYY-MM-DD) used to filter messages."
    )
    date_to: date | None = Field(
        default=None, description="End date (YYYY-MM-DD) used to filter messages."
    )
    max_results: int = Field(
        default=20,
        ge=1,
        description="Maximum number of results to return.",
    )


class IgptEmailSearchTool(BaseTool):
    name: str = "iGPT Email Search"
    description: str = (
        "Search across a user's email history using keyword and semantic retrieval "
        "with optional date filtering."
    )
    args_schema: type[BaseModel] = IgptEmailSearchInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["igptai"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="IGPT_API_KEY",
                description="API key for iGPT services",
                required=True,
            ),
            EnvVar(
                name="IGPT_API_USER",
                description="User identifier for iGPT email memory",
                required=True,
            ),
        ]
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("IGPT_API_KEY"),
        description="API key for iGPT services",
    )
    user: str | None = Field(
        default_factory=lambda: os.getenv("IGPT_API_USER"),
        description="User identifier for iGPT email memory",
    )
    quality: str = Field(
        default="cef-1-normal",
        description="Quality preset used by iGPT recall endpoints.",
    )
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError(
                "IGPT API key must be provided via constructor or IGPT_API_KEY environment variable."
            )
        if not self.user:
            raise ValueError(
                "IGPT user must be provided via constructor or IGPT_API_USER environment variable."
            )

        try:
            from igptai import IGPT  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'igptai' package is required to use IgptEmailSearchTool. "
                "Install it with: uv add igptai"
            ) from exc

        self._client = IGPT(api_key=self.api_key, user=self.user)

    def _run(
        self,
        query: str,
        date_from: date | None = None,
        date_to: date | None = None,
        max_results: int = 20,
    ) -> str:
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must be less than or equal to date_to.")

        search_kwargs: dict[str, str | int] = {
            "query": query,
            "quality": self.quality,
            "max_results": max_results,
        }
        if date_from:
            search_kwargs["date_from"] = date_from.isoformat()
        if date_to:
            search_kwargs["date_to"] = date_to.isoformat()

        response = self._client.recall.search(**search_kwargs)
        return _stringify_igpt_response(response)
