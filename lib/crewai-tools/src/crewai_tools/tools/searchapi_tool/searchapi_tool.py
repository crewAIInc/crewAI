"""Search the web through SearchApi (https://www.searchapi.io)."""

import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


DEFAULT_SEARCH_URL = "https://www.searchapi.io/api/v1/search"

_RESULT_LIST_SUFFIX = "_results"


def _is_data_uri(value: Any) -> bool:
    """Return True for an inline ``data:`` URI, which is never useful to an agent."""
    return isinstance(value, str) and value.startswith("data:")


def _sanitize(value: Any, max_string_length: int) -> Any:
    """Drop inline data URIs and truncate long strings, recursively.

    SearchApi ships favicons and thumbnails as ``data:image/png;base64,...``
    strings on most result items. They mean nothing to an agent and each one
    can run to tens of kilobytes, so they are dropped rather than truncated.

    Args:
        value: Any part of a decoded JSON response.
        max_string_length: Longest string kept intact; longer ones are cut.

    Returns:
        The same structure with data URIs removed and long strings truncated.
    """
    if isinstance(value, dict):
        return {
            key: _sanitize(item, max_string_length)
            for key, item in value.items()
            if not _is_data_uri(item)
        }
    if isinstance(value, list):
        return [
            _sanitize(item, max_string_length)
            for item in value
            if not _is_data_uri(item)
        ]
    if isinstance(value, str) and len(value) > max_string_length:
        return value[:max_string_length] + "..."
    return value


def _error_detail(response: requests.Response) -> str:
    """Pull SearchApi's error message out of a failed response.

    Args:
        response: A response whose status code is not a success.

    Returns:
        The API's own error message, or the start of the raw body when the
        response is not the documented ``{"error": ...}`` shape.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, str):
        return error
    return json.dumps(body)[:500]


class SearchApiToolSchema(BaseModel):
    """Input for SearchApiTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class SearchApiTool(BaseTool):
    """Search the internet through SearchApi's unified search endpoint.

    One endpoint fronts many engines, selected with ``engine``, so the same
    tool covers web search, news, scholar, jobs and the rest. The response is
    the engine's own JSON, so its shape follows the engine's documentation at
    https://www.searchapi.io/docs.
    """

    name: str = "Search the internet with SearchApi"
    description: str = (
        "A tool that can be used to search the internet with a search_query using "
        "SearchApi. One endpoint fronts many engines, set with 'engine': 'google' "
        "(default), 'google_news', 'google_scholar', 'google_jobs', 'bing', "
        "'youtube', 'baidu' and others. Returns the engine's structured JSON results."
    )
    args_schema: type[BaseModel] = SearchApiToolSchema
    search_url: str = DEFAULT_SEARCH_URL
    engine: str = "google"
    n_results: int = 10
    country: str | None = None
    locale: str | None = None
    location: str | None = None
    max_string_length: int = 1000
    timeout: int = 30
    api_key: str | None = None
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SEARCHAPI_API_KEY",
                description="API key for SearchApi",
                required=True,
            ),
        ]
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the search operation.

        Args:
            **kwargs: ``search_query`` (or ``query``) to search for, and an
                optional ``engine`` overriding the configured one for this call.

        Returns:
            The engine's JSON response, with data URIs dropped, long strings
            truncated and result lists capped at ``n_results``. A response that
            carries an ``error`` message instead of results, which is how
            SearchApi reports a page with nothing on it, is passed through so
            the agent can read why.

        Raises:
            ValueError: No search query was given, or no API key is configured.
            RuntimeError: SearchApi answered with an error status.
        """
        search_query: str | None = kwargs.get("search_query") or kwargs.get("query")
        if not search_query:
            raise ValueError("search_query is required")

        api_key = self.api_key or os.getenv("SEARCHAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "SEARCHAPI_API_KEY environment variable is required for SearchApiTool"
            )

        params: dict[str, Any] = {
            "engine": kwargs.get("engine", self.engine),
            "q": search_query,
        }
        if self.country:
            params["gl"] = self.country
        if self.locale:
            params["hl"] = self.locale
        if self.location:
            params["location"] = self.location

        # The key travels in the Authorization header rather than the query
        # string, so it stays out of request logs and out of the request_url
        # SearchApi echoes back in search_metadata.
        response = requests.get(
            self.search_url,
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"SearchApi request failed (HTTP {response.status_code}): "
                f"{_error_detail(response)}"
            )

        return self._format_results(response.json())

    def _format_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Cap result lists and strip payload that only burns context.

        Args:
            results: The decoded JSON body of a successful search.

        Returns:
            The same body with every ``*_results`` list capped at
            ``n_results`` and every value sanitized.
        """
        formatted: dict[str, Any] = {}
        for key, value in results.items():
            if key.endswith(_RESULT_LIST_SUFFIX) and isinstance(value, list):
                value = value[: self.n_results]
            formatted[key] = _sanitize(value, self.max_string_length)
        return formatted
