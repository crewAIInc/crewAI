from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crewai_tools.tools.context_dev_tools.base import ContextDevBaseTool, compact


class ContextExtractToolSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(
        min_length=1,
        description="Starting HTTP(S) URL to crawl and extract data from.",
    )
    response_schema: dict[str, Any] = Field(
        alias="schema",
        description="JSON Schema describing the structured data to return.",
    )
    instructions: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional guidance for interpreting and prioritizing fields.",
    )
    fact_check: bool = Field(
        default=False,
        description="Require every value to be grounded directly in page content.",
    )
    follow_subdomains: bool = Field(
        default=False,
        description="Follow relevant links on subdomains of the starting domain.",
    )
    max_pages: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of pages to analyze, from 1 to 50.",
    )
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="Maximum link depth; 0 limits extraction to the starting page.",
    )
    stop_after_ms: int | None = Field(
        default=None,
        ge=10000,
        le=110000,
        description="Soft extraction crawl time budget in milliseconds.",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        le=300000,
        description="Maximum server processing time in milliseconds.",
    )


class ContextExtractTool(ContextDevBaseTool):
    """Extract JSON Schema-constrained web data through Context.dev."""

    name: str = "Context.dev structured web extraction"
    description: str = (
        "Extract structured data from one or more website pages with Context.dev "
        "and return an object matching the supplied JSON Schema."
    )
    args_schema: type[BaseModel] = ContextExtractToolSchema

    def _run(
        self,
        url: str,
        response_schema: dict[str, Any],
        instructions: str | None = None,
        fact_check: bool = False,
        follow_subdomains: bool = False,
        max_pages: int = 5,
        max_depth: int | None = None,
        stop_after_ms: int | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        return self._request(
            "POST",
            "/web/extract",
            json_body=compact(
                {
                    "url": url,
                    "schema": response_schema,
                    "instructions": instructions,
                    "factCheck": fact_check,
                    "followSubdomains": follow_subdomains,
                    "maxPages": max_pages,
                    "maxDepth": max_depth,
                    "stopAfterMs": stop_after_ms,
                    "timeoutMS": timeout_ms,
                }
            ),
        )
