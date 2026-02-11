from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging
import os
import time
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


logger = logging.getLogger(__name__)

# Brave API error codes that indicate non-retryable quota/usage exhaustion.
_QUOTA_CODES = frozenset({"QUOTA_LIMITED", "USAGE_LIMIT_EXCEEDED"})


def _save_results_to_file(content: str) -> None:
    """Saves the search results to a file."""
    filename = f"search_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(filename, "w") as file:
        file.write(content)


def _parse_error_body(resp: requests.Response) -> dict[str, Any] | None:
    """Extract the structured "error" object from a Brave API error response."""
    try:
        body = resp.json()
        error = body.get("error")
        return error if isinstance(error, dict) else None
    except (ValueError, KeyError):
        return None


def _raise_for_error(resp: requests.Response) -> None:
    """Brave Search API error responses contain helpful JSON payloads"""
    status = resp.status_code
    try:
        body = json.dumps(resp.json())
    except (ValueError, KeyError):
        body = resp.text[:500]

    raise RuntimeError(f"Brave Search API error (HTTP {status}): {body}")


def _is_retryable(resp: requests.Response) -> bool:
    """Return True for transient failures that are worth retrying.

    * 429 + RATE_LIMITED — the per-second sliding window is full.
    * 5xx — transient server-side errors.

    Quota exhaustion (QUOTA_LIMITED, USAGE_LIMIT_EXCEEDED) is
    explicitly excluded: retrying will never succeed until the billing
    period resets.
    """
    if resp.status_code == 429:
        error = _parse_error_body(resp) or {}
        return error.get("code") not in _QUOTA_CODES
    return 500 <= resp.status_code < 600


def _retry_delay(resp: requests.Response, attempt: int) -> float:
    """Compute wait time before the next retry attempt.

    Prefers the server-supplied Retry-After header when available;
    falls back to exponential backoff (1s, 2s, 4s, ...).
    """
    retry_after = resp.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except (ValueError, TypeError):
            pass
    return float(2**attempt)


class BraveSearchToolBase(BaseTool, ABC):
    """
    Base class for Brave Search API interactions.

    Individual tool subclasses must provide the following:
      - search_url
      - header_schema (pydantic model)
      - args_schema (pydantic model)
      - _refine_payload() -> dict[str, Any]
    """

    search_url: str
    raw: bool = False
    args_schema: type[BaseModel]
    header_schema: type[BaseModel]

    # Rate limiting parameters
    _last_request_time: ClassVar[float] = 0

    # Tool options (legacy parameters)
    country: str | None = None
    save_file: bool = False
    n_results: int = 10

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BRAVE_API_KEY",
                description="API key for Brave Search",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        headers: dict[str, Any] | None = None,
        requests_per_second: float = 1.0,
        save_file: bool = False,
        raw: bool = False,
        timeout: int = 30,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self._api_key:
            raise ValueError("BRAVE_API_KEY environment variable is required")

        self.raw = bool(raw)
        self._timeout = int(timeout)
        self.save_file = bool(save_file)
        self._requests_per_second = float(requests_per_second)
        self._headers = self._build_and_validate_headers(headers or {})

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def headers(self) -> dict[str, Any]:
        return self._headers

    def set_headers(self, headers: dict[str, Any]) -> BraveSearchToolBase:
        merged = {**self._headers, **{k.lower(): v for k, v in headers.items()}}
        self._headers = self._build_and_validate_headers(merged)
        return self

    def _build_and_validate_headers(self, headers: dict[str, Any]) -> dict[str, Any]:
        normalized = {k.lower(): v for k, v in headers.items()}
        normalized.setdefault("x-subscription-token", self._api_key)
        normalized.setdefault("accept", "application/json")

        try:
            self.header_schema(**normalized)
        except Exception as e:
            raise ValueError(f"Invalid headers: {e}") from e

        return normalized

    def _rate_limit(self) -> None:
        if self._requests_per_second <= 0:
            return

        now = time.time()
        min_interval = 1.0 / self._requests_per_second
        elapsed = now - BraveSearchToolBase._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _make_request(
        self, params: dict[str, Any], *, _max_retries: int = 3
    ) -> dict[str, Any]:
        """Execute an HTTP GET against the Brave Search API with retry logic."""
        last_resp: requests.Response | None = None

        # Retry the request up to _max_retries times
        for attempt in range(_max_retries):
            self._rate_limit()

            # Make the request
            try:
                resp = requests.get(
                    self.search_url,
                    headers=self._headers,
                    params=params,
                    timeout=self._timeout,
                )
            except requests.ConnectionError as exc:
                raise RuntimeError(
                    f"Brave Search API connection failed: {exc}"
                ) from exc
            except requests.Timeout as exc:
                raise RuntimeError(
                    f"Brave Search API request timed out after {self._timeout}s: {exc}"
                ) from exc

            # Update the last request time
            BraveSearchToolBase._last_request_time = time.time()

            # Log the rate limit headers and request details
            logger.debug(
                "Brave Search API request: %s %s -> %d",
                "GET",
                resp.url,
                resp.status_code,
            )

            # Response was OK, return the JSON body
            if resp.ok:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise RuntimeError(
                        f"Brave Search API returned invalid JSON (HTTP {resp.status_code}): {exc}"
                    ) from exc

            # Response was not OK, but is retryable
            # (e.g., 429 Too Many Requests, 500 Internal Server Error)
            if _is_retryable(resp) and attempt < _max_retries - 1:
                delay = _retry_delay(resp, attempt)
                logger.warning(
                    "Brave Search API returned %d. Retrying in %.1fs (attempt %d/%d)",
                    resp.status_code,
                    delay,
                    attempt + 1,
                    _max_retries,
                )
                time.sleep(delay)
                last_resp = resp
                continue

            # Response was not OK, nor was it retryable
            # (e.g., 422 Unprocessable Entity, 400 Bad Request (OPTION_NOT_IN_PLAN))
            _raise_for_error(resp)

        # All retries exhausted
        _raise_for_error(last_resp or resp)  # type: ignore[possibly-undefined]
        return {}  # unreachable (here to satisfy the type checker and linter)

    def _run(self, q: str | None = None, **params: Any) -> Any:
        # Allow positional usage: tool.run("latest Brave browser features")
        if q is not None:
            params["q"] = q

        params = self._common_payload_refinement(params)

        # Validate only schema fields
        schema_keys = self.args_schema.model_fields
        payload_in = {k: v for k, v in params.items() if k in schema_keys}

        try:
            validated = self.args_schema(**payload_in)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

        # The subclass may have additional refinements to apply to the payload, such as goggles or other parameters
        payload = self._refine_request_payload(validated.model_dump(exclude_none=True))
        response = self._make_request(payload)

        if not self.raw:
            response = self._refine_response(response)

        if self.save_file:
            _save_results_to_file(json.dumps(response, indent=2))

        return response

    @abstractmethod
    def _refine_request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        """Subclass must implement: transform validated params dict into API request params."""
        raise NotImplementedError

    @abstractmethod
    def _refine_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Subclass must implement: transform response dict into a more useful format."""
        raise NotImplementedError

    _EMPTY_VALUES: ClassVar[tuple[None, str, str, list[Any]]] = (None, "", "null", [])

    def _common_payload_refinement(self, params: dict[str, Any]) -> dict[str, Any]:
        """Common payload refinement for all tools."""
        # crewAI's schema pipeline (ensure_all_properties_required in
        # pydantic_schema_utils.py) marks every property as required so
        # that OpenAI strict-mode structured outputs work correctly.
        # The side-effect is that the LLM fills in *every* parameter —
        # even truly optional ones — using placeholder values such as
        # None, "", "null", or [].  Only optional fields are affected,
        # so we limit the check to those.
        fields = self.args_schema.model_fields
        params = {
            k: v
            for k, v in params.items()
            # Permit custom and required fields, and fields with non-empty values
            if k not in fields or fields[k].is_required() or v not in self._EMPTY_VALUES
        }

        # Make sure params has "q" for query instead of "query" or "search_query"
        query = params.get("query") or params.get("search_query")
        if query is not None and "q" not in params:
            params["q"] = query
        params.pop("query", None)
        params.pop("search_query", None)

        # If "count" was not explicitly provided, use n_results
        # (only when the schema actually supports a "count" field)
        if "count" in self.args_schema.model_fields:
            if "count" not in params and self.n_results is not None:
                params["count"] = self.n_results

        # If "country" was not explicitly provided, but self.country is set, use it
        # (only when the schema actually supports a "country" field)
        if "country" in self.args_schema.model_fields:
            if "country" not in params and self.country is not None:
                params["country"] = self.country

        return params
