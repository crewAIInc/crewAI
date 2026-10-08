import json
import logging
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests

from crewai_tools.security.safe_path import validate_url


logger = logging.getLogger(__name__)

STRING_API_BASE_URL = "https://request.usestring.ai/v1"
_DEFAULT_TIMEOUT = 120


def _api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("STRING_API_KEY")
    if not key:
        raise ValueError(
            "STRING_API_KEY is not set. Get a key at https://usestring.ai and set it as "
            "STRING_API_KEY, or pass api_key= to the tool."
        )
    return key


def _post(
    path: str, payload: dict[str, Any], api_key: str, timeout: int
) -> requests.Response:
    response = requests.post(
        f"{STRING_API_BASE_URL}{path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


class StringWebAccessScrapeToolSchema(BaseModel):
    url: str = Field(description="The http/https URL to fetch")
    format: Literal["markdown", "raw", "json"] | None = Field(
        default="markdown",
        description="Response format: markdown (default), raw for the verbatim body, json for a "
        "{statusCode, headers, data} envelope",
    )
    main_content_only: bool | None = Field(
        default=None,
        description="Strip page chrome (navigation, footers) from the returned Markdown",
    )
    execute_js: bool | None = Field(
        default=None,
        description="Render the page in a browser first. Use when a fetch comes back empty on a "
        "JavaScript-rendered site",
    )
    country_code: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code to route the request through, e.g. 'US'",
    )


class StringWebAccessScrapeTool(BaseTool):
    """Fetch any URL through String Web Access and return it as clean, LLM-ready Markdown."""

    name: str = "String Web Access scrape tool"
    description: str = (
        "Fetch any URL and get clean, LLM-ready Markdown back. Proxy rotation, anti-bot handling, "
        "CAPTCHA solving and JavaScript rendering happen server-side, so use this for pages that "
        "rate-limit, geo-gate or block automated traffic and return a block screen to an ordinary "
        "HTTP request."
    )
    args_schema: type[BaseModel] = StringWebAccessScrapeToolSchema
    api_key: str | None = None
    timeout: int = _DEFAULT_TIMEOUT
    ignore_failures: bool = False
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="STRING_API_KEY",
                description="API key for String Web Access",
                required=True,
            ),
        ]
    )

    def _run(
        self,
        url: str,
        format: str = "markdown",
        main_content_only: bool | None = None,
        execute_js: bool | None = None,
        country_code: str | None = None,
    ) -> str | None:
        try:
            url = validate_url(url)
            payload: dict[str, Any] = {"url": url, "format": format}
            if main_content_only is not None:
                payload["mainContentOnly"] = main_content_only
            if execute_js is not None:
                payload["executeJS"] = execute_js
            if country_code is not None:
                payload["countryCode"] = country_code

            response = _post("/fetch", payload, _api_key(self.api_key), self.timeout)
            if format == "json":
                return json.dumps(response.json(), indent=2)
            return response.text
        except Exception as e:
            if self.ignore_failures:
                logger.error(f"Error fetching {url} through String Web Access: {e}")
                return None
            raise


class StringWebAccessSearchToolSchema(BaseModel):
    query: str = Field(description="The search query to run")
    engine: Literal["google", "duckduckgo", "brave", "mojeek"] | None = Field(
        default="google", description="Search engine to query"
    )
    country: str | None = Field(
        default="US",
        description="ISO 3166-1 alpha-2 country code used to localize the results",
    )
    language: str | None = Field(
        default=None,
        description="Language tag for the results, such as 'en' or 'pt-br'",
    )
    max_results: int | None = Field(
        default=10, gt=0, description="Maximum number of organic results to return"
    )


class StringWebAccessSearchTool(BaseTool):
    """Search the web through String Web Access and return the organic results."""

    name: str = "String Web Access search tool"
    description: str = (
        "Search the web and get the organic results back as structured JSON — position, title, "
        "url, displayUrl and snippet for each. Supports Google, DuckDuckGo, Brave and Mojeek, "
        "localized by country and language."
    )
    args_schema: type[BaseModel] = StringWebAccessSearchToolSchema
    api_key: str | None = None
    timeout: int = _DEFAULT_TIMEOUT
    ignore_failures: bool = False
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="STRING_API_KEY",
                description="API key for String Web Access",
                required=True,
            ),
        ]
    )

    def _run(
        self,
        query: str,
        engine: str = "google",
        country: str = "US",
        language: str | None = None,
        max_results: int = 10,
    ) -> str | None:
        payload: dict[str, Any] = {"query": query, "engine": engine, "country": country}
        if language is not None:
            payload["language"] = language

        try:
            response = _post("/search", payload, _api_key(self.api_key), self.timeout)
            results = response.json().get("results", [])[:max_results]
        except Exception as e:
            if self.ignore_failures:
                logger.error(f"Error searching String Web Access for '{query}': {e}")
                return None
            raise

        if not results:
            return f"No results found for '{query}'."
        return json.dumps(results, indent=2)
