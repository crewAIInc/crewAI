import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


AUTOM_API_URL = "https://api.autom.dev"


def _error_message(response: requests.Response, api_key: str) -> str:
    """A readable error for the agent, without the API key."""
    detail = ""
    try:
        body = response.json()
    except ValueError:
        detail = response.text.strip()
    else:
        if isinstance(body, dict):
            error = body.get("error") or body.get("message") or ""
            if isinstance(error, dict):
                error = error.get("message") or ""
            detail = str(error)
    if api_key:
        detail = detail.replace(api_key, "***")
    detail = detail[:300]
    status = response.status_code
    # Autom answers a rate limit with HTTP 401 and a "Rate limit exceeded" message.
    if status == 429 or (status == 401 and "rate limit" in detail.lower()):
        label = "Autom rate limit exceeded"
    elif status == 401:
        label = "Autom rejected the API key"
    elif status == 402:
        label = "Autom account is out of credits"
    else:
        label = "Autom API error"
    return f"{label} (HTTP {status})" + (f": {detail}" if detail else "")


class AutomSearchToolSchema(BaseModel):
    """Input for AutomSearchTool."""

    search_query: str = Field(
        ...,
        description="The search query. Search operators such as site: are supported.",
    )


class AutomSearchTool(BaseTool):
    """Search the web with Autom, a SERP API for live Google, Bing, Brave, DuckDuckGo and Yahoo results.

    Returns a JSON list of results with ``title``, ``link``, ``snippet`` and ``position``.
    Google News results also carry ``date`` and ``source``.
    """

    name: str = "Search the internet with Autom"
    description: str = (
        "Search the internet with a search_query and get the top results with title, "
        "link and snippet. Use it for current events and facts you do not know."
    )
    args_schema: type[BaseModel] = AutomSearchToolSchema
    engine: Literal["google", "bing", "brave", "duckduckgo", "yahoo"] = "google"
    search_type: Literal["search", "news"] = "search"
    n_results: int = Field(default=10, ge=1, le=100)
    country: str | None = Field(
        default=None, description="Two-letter country code, such as us."
    )
    language: str | None = Field(
        default=None, description="Language of the results, such as en (Google only)."
    )
    location: str | None = Field(
        default=None,
        description="City or region to search from (Google web search only).",
    )
    timeout: float = 60.0
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="AUTOM_API_KEY",
                description="API key for Autom",
                required=True,
            ),
        ]
    )

    def _request(self, query: str) -> tuple[str, dict[str, Any]]:
        """Pick the endpoint and query parameters for this search."""
        if self.search_type == "news":
            # Autom has news results for Google only.
            path, engine = "/v1/google/news", "google"
        else:
            path, engine = f"/v1/{self.engine}/search", self.engine
        params: dict[str, Any] = {"query": query}
        if self.country:
            # Bing takes the country as cc; the other engines take gl.
            params["cc" if engine == "bing" else "gl"] = self.country.strip().lower()
        if engine == "google":
            if self.language:
                params["hl"] = self.language.strip()
            if self.location and self.search_type == "search":
                params["location"] = self.location.strip()
        return path, params

    def _run(self, search_query: str, **kwargs: Any) -> str:
        query = (search_query or "").strip()
        if not query:
            return "The search query is empty."
        api_key = (os.environ.get("AUTOM_API_KEY") or "").strip()
        if not api_key:
            return "AUTOM_API_KEY is not set. Get a key at https://app.autom.dev"

        path, params = self._request(query)
        try:
            response = requests.get(
                f"{AUTOM_API_URL}{path}",
                params=params,
                headers={"x-api-key": api_key, "Accept": "application/json"},
                timeout=self.timeout,
            )
        except requests.Timeout:
            return f"Autom did not answer within {self.timeout:g} seconds."
        except requests.RequestException as e:
            return f"Could not reach Autom: {str(e).replace(api_key, '***')}"
        if not response.ok:
            return _error_message(response, api_key)

        try:
            body = response.json()
        except ValueError:
            return "Autom returned a response that is not JSON."
        rows = body.get("organic_results") if isinstance(body, dict) else None
        if rows is None and isinstance(body, dict):
            rows = []
        if not isinstance(rows, list):
            return "Autom returned an unexpected response."

        results: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            result = {
                "title": row.get("title") or "",
                "link": row.get("link") or "",
                "snippet": row.get("snippet") or "",
            }
            for key in ("position", "date", "source"):
                if row.get(key) not in (None, ""):
                    result[key] = row[key]
            results.append(result)
            if len(results) >= self.n_results:
                break
        if not results:
            return f"No results for: {query}"
        return json.dumps(results, ensure_ascii=False)
