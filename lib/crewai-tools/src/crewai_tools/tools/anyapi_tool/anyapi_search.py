from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.anyapi_tool.anyapi_base import AnyApiToolBase


class AnyApiSearchToolSchema(BaseModel):
    """Input for AnyApiSearchTool."""

    query: str = Field(
        ...,
        description=(
            "What the data is, in plain words, for example 'instagram profile' or "
            "'google maps reviews'."
        ),
    )
    category: str | None = Field(
        default=None,
        description="Optional category filter, for example 'social' or 'search'.",
    )
    platform: str | None = Field(
        default=None,
        description=(
            "Optional platform filter, matched against the slug prefix, for example "
            "'instagram', 'tiktok' or 'linkedin'."
        ),
    )
    limit: int | None = Field(
        default=None,
        description="Optional cap on how many matching APIs to return.",
    )


class AnyApiSearchTool(AnyApiToolBase):
    """Search the AnyAPI catalog for the API that answers a question."""

    name: str = "AnyAPI catalog search"
    description: str = (
        "Search the AnyAPI catalog for an API that returns the data you need. AnyAPI is "
        "one gateway to hundreds of scraping and data APIs (social media, search "
        "results, general web data), reached with one key and paid per request in USD. "
        "Use this tool first. It returns matching slugs with a short description and the "
        "USD price per request, but no input schemas. Pick a slug from the results, then "
        "call 'AnyAPI endpoint schema' on it, and only then 'AnyAPI endpoint run'."
    )
    args_schema: type[BaseModel] = AnyApiSearchToolSchema

    def _run(
        self,
        query: str,
        category: str | None = None,
        platform: str | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> str:
        try:
            results = self._client.search(
                query=query,
                category=category,
                platform=platform,
                limit=limit,
            )
        except self._anyapi.AnyAPIError as exc:
            return f"AnyAPI catalog search failed for '{query}': {exc}"

        return self._as_json(results)
