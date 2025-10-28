import os
from typing import Annotated, Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class ParallelSearchInput(BaseModel):
    """Input schema for ParallelSearchTool using the Search API (v1beta).

    At least one of objective or search_queries is required.
    """

    objective: str | None = Field(
        None,
        description="Natural-language goal for the web research (<=5000 chars)",
        max_length=5000,
    )
    search_queries: list[Annotated[str, Field(max_length=200)]] | None = Field(
        default=None,
        description="Optional list of keyword queries (<=5 items, each <=200 chars)",
        min_length=1,
        max_length=5,
    )
    processor: str = Field(
        default="base",
        description="Search processor: 'base' (fast/low cost) or 'pro' (higher quality/freshness)",
        pattern=r"^(base|pro)$",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=40,
        description="Maximum number of search results to return (processor limits apply)",
    )
    max_chars_per_result: int = Field(
        default=6000,
        ge=100,
        description="Maximum characters per result excerpt (values >30000 not guaranteed)",
    )
    source_policy: dict[str, Any] | None = Field(
        default=None, description="Optional source policy configuration"
    )


class ParallelSearchTool(BaseTool):
    name: str = "Parallel Web Search Tool"
    description: str = (
        "Search the web using Parallel's Search API (v1beta). Returns ranked results with "
        "compressed excerpts optimized for LLMs."
    )
    args_schema: type[BaseModel] = ParallelSearchInput

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="PARALLEL_API_KEY",
                description="API key for Parallel",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])

    search_url: str = "https://api.parallel.ai/v1beta/search"

    def _run(
        self,
        objective: str | None = None,
        search_queries: list[str] | None = None,
        processor: str = "base",
        max_results: int = 10,
        max_chars_per_result: int = 6000,
        source_policy: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        api_key = os.environ.get("PARALLEL_API_KEY")
        if not api_key:
            return "Error: PARALLEL_API_KEY environment variable is required"

        if not objective and not search_queries:
            return "Error: Provide at least one of 'objective' or 'search_queries'"

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            payload: dict[str, Any] = {
                "processor": processor,
                "max_results": max_results,
                "max_chars_per_result": max_chars_per_result,
            }
            if objective is not None:
                payload["objective"] = objective
            if search_queries is not None:
                payload["search_queries"] = search_queries
            if source_policy is not None:
                payload["source_policy"] = source_policy

            request_timeout = 90 if processor == "pro" else 30
            resp = requests.post(
                self.search_url, json=payload, headers=headers, timeout=request_timeout
            )
            if resp.status_code >= 300:
                return (
                    f"Parallel Search API error: {resp.status_code} {resp.text[:200]}"
                )
            data = resp.json()
            return self._format_output(data)
        except requests.Timeout:
            return "Parallel Search API timeout. Please try again later."
        except Exception as exc:
            return f"Unexpected error calling Parallel Search API: {exc}"

    def _format_output(self, result: dict[str, Any]) -> str:
        # Return the full JSON payload (search_id + results) as a compact JSON string
        try:
            import json

            return json.dumps(result or {}, ensure_ascii=False)
        except Exception:
            return str(result or {})
