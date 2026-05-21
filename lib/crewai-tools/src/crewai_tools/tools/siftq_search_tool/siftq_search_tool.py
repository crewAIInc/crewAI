import asyncio
import json
import os
from typing import Any, Literal

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


# Intentionally public free-tier key (100 searches/day). Users can set SIFTQ_API_KEY for higher limits.
DEFAULT_SIFTQ_API_KEY = "mk-1D3D81EFC32A25683B0C2C3B315F8579"
SIFTQ_API_URL = "https://api.siftq.com/v1/search"


class SiftqSearchToolSchema(BaseModel):
    query: str = Field(..., description="The search query string.")
    scope: (
        Literal["webpage", "document", "scholar", "image", "video", "podcast"] | None
    ) = Field(
        default=None,
        description="Override the search scope for this query. One of: webpage, document, scholar, image, video, podcast. "
        "Uses the tool's default scope if not provided.",
    )
    include_summary: bool | None = Field(
        default=None,
        description="Override whether to include AI-generated summaries of each result.",
    )
    include_raw_content: bool | None = Field(
        default=None,
        description="Override whether to fetch and include the raw content from each source page.",
    )
    concise_snippet: bool | None = Field(
        default=None,
        description="Override whether to return concise snippets with exact original text matches.",
    )
    max_results: int | None = Field(
        default=None,
        description="Override the maximum number of results to return (1-100).",
        ge=1,
        le=100,
    )


class SiftqSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "SiftQ Search"
    description: str = (
        "A tool that performs neural-ranked web searches using the SiftQ API. "
        "Returns a JSON object containing the search results with titles, URLs, snippets, and more."
    )
    args_schema: type[BaseModel] = SiftqSearchToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SIFTQ_API_KEY"),
        description="The SiftQ API key. Falls back to a public free-tier key if not set. "
        "Set the SIFTQ_API_KEY environment variable for a higher rate limit.",
    )
    scope: Literal["webpage", "document", "scholar", "image", "video", "podcast"] = (
        Field(
            default="webpage",
            description="The search scope. One of: webpage, document, scholar, image, video, podcast.",
        )
    )
    include_summary: bool = Field(
        default=False,
        description="Whether to include AI-generated summaries of each result.",
    )
    include_raw_content: bool = Field(
        default=False,
        description="Whether to fetch and include the raw content from each source page.",
    )
    concise_snippet: bool = Field(
        default=False,
        description="Whether to return concise snippets with exact original text matches.",
    )
    max_results: int = Field(
        default=5,
        description="The maximum number of results to return (1-100).",
        ge=1,
        le=100,
    )
    timeout: int = Field(
        default=60,
        description="The timeout for the search request in seconds.",
    )
    max_content_length_per_result: int = Field(
        default=1000,
        description="Maximum length for the 'content' of each search result to avoid context window issues.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SIFTQ_API_KEY",
                description="API key for SiftQ search service",
                required=False,
            ),
        ]
    )

    def _get_api_key(self) -> str:
        return self.api_key or DEFAULT_SIFTQ_API_KEY

    def _run(
        self,
        query: str,
        scope: Literal["webpage", "document", "scholar", "image", "video", "podcast"]
        | None = None,
        include_summary: bool | None = None,
        include_raw_content: bool | None = None,
        concise_snippet: bool | None = None,
        max_results: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "q": query,
            "scope": scope or self.scope,
            "includeSummary": include_summary if include_summary is not None else self.include_summary,
            "includeRawContent": include_raw_content if include_raw_content is not None else self.include_raw_content,
            "conciseSnippet": concise_snippet if concise_snippet is not None else self.concise_snippet,
            "size": max_results if max_results is not None else self.max_results,
        }

        try:
            response = requests.post(
                SIFTQ_API_URL,
                headers={
                    "Authorization": f"Bearer {self._get_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"SiftQ search timed out after {self.timeout}s. Try again or increase the timeout."
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to the SiftQ API. Check your network connection."
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            raise RuntimeError(
                f"SiftQ API returned HTTP {status}: {e}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"SiftQ search request failed: {e}"
            ) from e

        try:
            raw_results = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"SiftQ returned invalid JSON: {e}"
            ) from e

        result_key = f"{scope or self.scope}s"
        if result_key in raw_results and isinstance(raw_results[result_key], list):
            for item in raw_results[result_key]:
                for field in ("content", "snippet", "summary"):
                    value = item.get(field)
                    if (
                        isinstance(value, str)
                        and len(value) > self.max_content_length_per_result
                    ):
                        item[field] = (
                            value[: self.max_content_length_per_result] + "..."
                        )

        return json.dumps(raw_results, indent=2)

    async def _arun(
        self,
        query: str,
        scope: Literal["webpage", "document", "scholar", "image", "video", "podcast"]
        | None = None,
        include_summary: bool | None = None,
        include_raw_content: bool | None = None,
        concise_snippet: bool | None = None,
        max_results: int | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run, query, scope, include_summary, include_raw_content, concise_snippet, max_results
        )
