import json
import os
import re
from typing import Any, Literal

from bs4 import BeautifulSoup
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


PILOTERR_API_URL = "https://api.piloterr.com"


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
    # Piloterr answers a rate limit with HTTP 401 and a "Rate limit exceeded" message.
    if status == 429 or (status == 401 and "rate limit" in detail.lower()):
        label = "Piloterr rate limit exceeded"
    elif status == 401:
        label = "Piloterr rejected the API key"
    elif status == 402:
        label = "Piloterr account is out of credits"
    else:
        label = "Piloterr API error"
    return f"{label} (HTTP {status})" + (f": {detail}" if detail else "")


class _PiloterrTool(BaseTool):
    """Shared request and error handling for the Piloterr tools."""

    timeout: float = 60.0
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="PILOTERR_API_KEY",
                description="API key for Piloterr",
                required=True,
            ),
        ]
    )

    def _get(self, path: str, params: dict[str, Any]) -> requests.Response | str:
        """GET a Piloterr endpoint, or return an error message for the agent."""
        api_key = (os.environ.get("PILOTERR_API_KEY") or "").strip()
        if not api_key:
            return "PILOTERR_API_KEY is not set. Get a key at https://app.piloterr.com"
        try:
            response = requests.get(
                f"{PILOTERR_API_URL}{path}",
                params=params,
                headers={"x-api-key": api_key, "Accept": "application/json"},
                timeout=self.timeout,
            )
        except requests.Timeout:
            return f"Piloterr did not answer within {self.timeout:g} seconds."
        except requests.RequestException as e:
            return f"Could not reach Piloterr: {str(e).replace(api_key, '***')}"
        if not response.ok:
            return _error_message(response, api_key)
        return response


class PiloterrGoogleSearchToolSchema(BaseModel):
    """Input for PiloterrGoogleSearchTool."""

    search_query: str = Field(
        ...,
        description="The Google search query. Operators such as site: are supported.",
    )


class PiloterrGoogleSearchTool(_PiloterrTool):
    """Search Google with Piloterr.

    Returns a JSON list of results with ``title``, ``link``, ``snippet`` and ``position``.
    """

    name: str = "Search Google with Piloterr"
    description: str = (
        "Search Google with a search_query and get the top results with title, "
        "link and snippet. Use it for current events and facts you do not know."
    )
    args_schema: type[BaseModel] = PiloterrGoogleSearchToolSchema
    n_results: int = Field(default=10, ge=1, le=100)
    country: str | None = Field(
        default=None, description="Two-letter country code, such as us."
    )
    language: str | None = Field(
        default=None, description="Language of the results, such as en."
    )
    location: str | None = Field(
        default=None, description="City or region to search from."
    )

    def _run(self, search_query: str, **kwargs: Any) -> str:
        query = (search_query or "").strip()
        if not query:
            return "The search query is empty."
        params: dict[str, Any] = {"query": query, "num": self.n_results}
        if self.country:
            params["gl"] = self.country.strip().lower()
        if self.language:
            params["hl"] = self.language.strip()
        if self.location:
            params["location"] = self.location.strip()

        response = self._get("/v2/google/search", params)
        if isinstance(response, str):
            return response
        try:
            body = response.json()
        except ValueError:
            return "Piloterr returned a response that is not JSON."
        rows = body.get("organic_results") if isinstance(body, dict) else None
        if rows is None and isinstance(body, dict):
            rows = []
        if not isinstance(rows, list):
            return "Piloterr returned an unexpected response."

        results: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            result = {
                "title": row.get("title") or "",
                "link": row.get("link") or "",
                "snippet": row.get("snippet") or "",
            }
            if row.get("position") is not None:
                result["position"] = row["position"]
            results.append(result)
            if len(results) >= self.n_results:
                break
        if not results:
            return f"No results for: {query}"
        return json.dumps(results, ensure_ascii=False)


class PiloterrScrapeWebsiteToolSchema(BaseModel):
    """Input for PiloterrScrapeWebsiteTool."""

    website_url: str = Field(
        ..., description="The absolute http(s) URL of the page to read."
    )


class PiloterrScrapeWebsiteTool(_PiloterrTool):
    """Read a web page through Piloterr and return its title and readable text.

    ``engine`` picks how Piloterr loads the page: ``crawler`` (a plain HTTP request,
    the default), ``rendering`` (a headless browser, for pages that need JavaScript)
    or ``webunlocker`` (for pages behind anti-bot protection).
    """

    name: str = "Read a website with Piloterr"
    description: str = "Read the content of a web page from its website_url and get its title and text."
    args_schema: type[BaseModel] = PiloterrScrapeWebsiteToolSchema
    engine: Literal["crawler", "rendering", "webunlocker"] = "crawler"
    wait_in_seconds: int | None = Field(
        default=None,
        ge=0,
        le=30,
        description="Rendering only: extra seconds to wait after the page loads.",
    )
    max_characters: int | None = Field(default=20_000, ge=1)
    timeout: float = 120.0

    def _run(self, website_url: str, **kwargs: Any) -> str:
        url = (website_url or "").strip()
        if not re.match(r"^https?://[^/\s]+", url, re.IGNORECASE):
            return f"Use an absolute http(s) URL, such as https://example.com; got {url!r}."
        params: dict[str, Any] = {"query": url}
        if self.engine == "rendering" and self.wait_in_seconds:
            params["wait_in_seconds"] = self.wait_in_seconds

        response = self._get(f"/v2/website/{self.engine}", params)
        if isinstance(response, str):
            return response
        try:
            html = response.json()
        except ValueError:
            html = response.text
        if not isinstance(html, str):
            return "Piloterr returned an unexpected response; expected the page HTML."

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for tag in soup(["script", "style", "noscript", "template", "svg", "head"]):
            tag.decompose()
        lines = (line.strip() for line in soup.get_text("\n").splitlines())
        text = "\n".join(re.sub(r"\s+", " ", line) for line in lines if line)
        if self.max_characters is not None:
            text = text[: self.max_characters]
        return f"# {title or url}\n\n{text or '(The page has no readable text.)'}"
