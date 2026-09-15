import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator
import requests


load_dotenv()

ANYSEARCH_SEARCH_URL = "https://api.anysearch.com/v1/search"
MIN_RESULTS = 1
MAX_RESULTS = 10


class AnySearchToolSchema(BaseModel):
    """Input schema for AnySearchTool."""

    query: str = Field(..., description="The search query string.")


class AnySearchTool(BaseTool):
    """Tool that uses the AnySearch API to perform web searches.

    AnySearch accepts anonymous requests: when no API key is configured the
    request is sent without an ``Authorization`` header. Setting
    ``ANYSEARCH_API_KEY`` sends authenticated requests instead; the applicable
    quota and permissions are determined by the AnySearch service.

    Attributes:
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        search_url: The AnySearch search endpoint.
        api_key: Optional AnySearch API key.
        max_results: The maximum number of results to return (1-10).
        result_format: The format requested from the API.
        timeout: The timeout for the search request in seconds.
        max_content_length_per_result: Maximum length for the 'content' of each result.
    """

    name: str = "AnySearch Web Search"
    description: str = (
        "A tool that performs a web search using the AnySearch API and returns "
        "the most relevant results as a JSON string. Useful to look up current "
        "information on the internet. Works out of the box without an API key."
    )
    args_schema: type[BaseModel] = AnySearchToolSchema
    search_url: str = Field(
        default=ANYSEARCH_SEARCH_URL,
        description="The AnySearch search endpoint.",
    )
    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(key)
            if (key := os.getenv("ANYSEARCH_API_KEY", "").strip())
            else None
        ),
        description=(
            "The AnySearch API key. If not provided, it is loaded from the "
            "ANYSEARCH_API_KEY environment variable. When it is unset or blank "
            "the tool sends anonymous requests. Stored as a SecretStr so it is "
            "masked in repr()/model_dump()."
        ),
    )
    max_results: int = Field(
        default=10,
        ge=MIN_RESULTS,
        le=MAX_RESULTS,
        description="The maximum number of results to return (1-10).",
    )
    result_format: Literal["json", "markdown"] = Field(
        default="json",
        description="The content format requested from the AnySearch API.",
    )
    timeout: int = Field(
        default=30,
        description="The timeout for the search request in seconds.",
    )
    max_content_length_per_result: int = Field(
        default=1000,
        description=(
            "Maximum length for the 'content' of each search result to avoid "
            "context window issues."
        ),
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="ANYSEARCH_API_KEY",
                description=(
                    "Optional API key for AnySearch. Leave it unset to send "
                    "anonymous requests."
                ),
                required=False,
            ),
        ]
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def _normalize_api_key(cls, value: Any) -> Any:
        """Normalizes a blank API key to None so no empty credential is sent.

        Args:
            value: The raw api_key input, either a string or a SecretStr.

        Returns:
            None when the key is blank, otherwise the stripped key.
        """
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            return value.strip() or None
        return value

    def _build_headers(self) -> dict[str, str]:
        """Builds the request headers, omitting auth when running anonymously.

        Returns:
            The headers to send with the search request.
        """
        headers = {"Content-Type": "application/json"}
        # Anonymous request: omit the auth header unless a non-blank key is set.
        api_key = self.api_key.get_secret_value().strip() if self.api_key else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _run(self, query: str) -> str:
        """Performs a search using the AnySearch API.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results.

        Raises:
            RuntimeError: If the AnySearch API reports a non-zero status code or
                returns a body that does not match the documented envelope.
        """
        # Clamp max_results to the API-supported range (1-10) at runtime.
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max(MIN_RESULTS, min(self.max_results, MAX_RESULTS)),
            "format": self.result_format,
        }

        # Transport-level failures (HTTP errors, timeouts) raise here.
        response = requests.post(
            self.search_url,
            headers=self._build_headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        try:
            body: Any = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "AnySearch API returned a malformed response: "
                "the body is not valid JSON"
            ) from exc

        if not isinstance(body, dict):
            raise RuntimeError(
                "AnySearch API returned a malformed response: "
                f"the body must be an object, got {type(body).__name__}"
            )

        code = body.get("code")
        # bool subclasses int, so reject it explicitly: `code: false` is not a success.
        if isinstance(code, bool) or not isinstance(code, int):
            raise RuntimeError(
                "AnySearch API returned a malformed response: "
                f"'code' must be an integer, got {type(code).__name__}"
            )
        # Business-level errors are reported in the response body with a non-zero code.
        if code != 0:
            message = body.get("message", "unknown error")
            raise RuntimeError(
                f"AnySearch API returned an error (code={code}): {message}"
            )

        data = body.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(
                "AnySearch API returned a malformed response: "
                f"'data' must be an object, got {type(data).__name__}"
            )
        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError(
                "AnySearch API returned a malformed response: "
                f"'data.results' must be a list, got {type(raw_results).__name__}"
            )

        results: list[dict[str, Any]] = []
        # Reject malformed entries instead of degrading them to empty results.
        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                raise RuntimeError(
                    "AnySearch API returned a malformed response: "
                    f"result #{index} must be an object, got {type(item).__name__}"
                )
            url = item.get("url")
            if not isinstance(url, str) or not url.strip():
                raise RuntimeError(
                    "AnySearch API returned a malformed response: "
                    f"result #{index} is missing a valid 'url'"
                )
            content = str(item.get("content") or "")
            if len(content) > self.max_content_length_per_result:
                content = content[: self.max_content_length_per_result] + "..."
            results.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": url,
                    "snippet": str(item.get("snippet") or ""),
                    "content": content,
                }
            )

        return json.dumps({"query": query, "results": results}, indent=2)
