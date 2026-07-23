import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field

try:
    from serpdive import AsyncSerpDive, SerpDive  # type: ignore[import-untyped]

    SERPDIVE_AVAILABLE = True
except ImportError:
    SERPDIVE_AVAILABLE = False


class SerpdiveSearchToolSchema(BaseModel):
    """Input schema for SerpdiveSearchTool."""

    query: str = Field(
        ...,
        description="The search query, in any language, phrased like a real web search.",
    )


class SerpdiveSearchTool(BaseTool):
    """Tool that uses the SERPdive Search API to perform web searches.

    Every result carries the actual text of the page (url, title, date,
    content), already extracted and cleaned for LLM consumption, so agents
    can quote and cite straight from the tool output.

    Attributes:
        client: An instance of SerpDive (sync client).
        async_client: An instance of AsyncSerpDive.
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The SERPdive API key.
        model: Retrieval depth, "mako" (key sentences, fast) or "moby"
            (full page text, deep research).
        answer: Whether to also return a direct answer synthesized from
            the sources.
        max_results: Hard cap on delivered results (1-10).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any | None = None
    async_client: Any | None = None
    name: str = "SERPdive Search"
    description: str = (
        "Search the live web and get back answer-ready page content, not a list "
        "of links. Each result carries the actual text of the page (url, title, "
        "date, content), already extracted and cleaned, so facts can be quoted "
        "and cited straight from the response. Use it for anything that needs "
        "current or post-training information: news, prices, releases, docs, "
        "niche facts. Write the query the way a person would type it, in any "
        "language: localization is automatic."
    )
    args_schema: type[BaseModel] = SerpdiveSearchToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SERPDIVE_API_KEY"),
        description="The SERPdive API key. If not provided, it is read from the SERPDIVE_API_KEY environment variable.",
    )
    model: str = Field(
        default="mako",
        description='Retrieval depth: "mako" (default) returns the fact-carrying sentences of each page, fast; "moby" returns the full readable text, for deep research.',
    )
    answer: bool = Field(
        default=False,
        description='When True, the output also carries an "answer" field: a direct answer synthesized from the sources (concise on mako, cited on moby).',
    )
    max_results: int | None = Field(
        default=None,
        description="Hard cap on delivered results (1-10). None lets the engine pick its calibrated mix.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["serpdive"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SERPDIVE_API_KEY",
                description="API key for SERPdive, free at https://serpdive.com/dashboard/keys",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not SERPDIVE_AVAILABLE:
            raise ImportError(
                "The 'serpdive' package is required to use the SerpdiveSearchTool. "
                "Please install it with: uv add serpdive (or pip install serpdive)"
            )
        self.client = SerpDive(api_key=self.api_key)
        self.async_client = AsyncSerpDive(api_key=self.api_key)

    def _search_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model}
        if self.answer:
            kwargs["answer"] = True
        if self.max_results is not None:
            kwargs["max_results"] = self.max_results
        return kwargs

    def _payload(self, response: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": response.query,
            "results": [
                {
                    "url": result.url,
                    "title": result.title,
                    **({"date": result.date} if result.date else {}),
                    "content": result.content,
                }
                for result in response.results
            ],
        }
        if response.answer is not None:
            payload["answer"] = response.answer
        if response.extra_info is not None:
            payload["extra_info"] = response.extra_info
        return payload

    def _run(self, query: str) -> str:
        """Synchronously performs a search using the SERPdive API.

        Args:
            query: The search query string.

        Returns:
            A JSON string with the query, the results (url, title, date,
            content) and, when enabled, the synthesized answer.
        """
        if not self.client:
            raise ValueError(
                "SERPdive client is not initialized. Ensure 'serpdive' is "
                "installed and the API key is set."
            )
        response = self.client.search(query, **self._search_kwargs())
        return json.dumps(self._payload(response), indent=2)

    async def _arun(self, query: str) -> str:
        """Asynchronously performs a search using the SERPdive API.

        Args:
            query: The search query string.

        Returns:
            A JSON string with the query, the results (url, title, date,
            content) and, when enabled, the synthesized answer.
        """
        if not self.async_client:
            raise ValueError(
                "SERPdive async client is not initialized. Ensure 'serpdive' is "
                "installed and the API key is set."
            )
        response = await self.async_client.search(query, **self._search_kwargs())
        return json.dumps(self._payload(response), indent=2)
