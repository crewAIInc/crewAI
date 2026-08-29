from __future__ import annotations

from collections.abc import Mapping
import os
import time
from typing import Any
from urllib.parse import urlparse

from crewai.tools import BaseTool, EnvVar
from pydantic import Field, SecretStr
import requests


DEFAULT_CONTEXT_API_BASE = "https://api.context.dev/v1"
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 180.0
MAX_CONTEXT_REQUEST_ATTEMPTS = 3
MAX_CONTEXT_RETRY_DELAY_SECONDS = 10.0
OMITTED_RESPONSE_FIELDS = {"debug", "key_metadata", "request_id", "trace_id"}


def compact(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy without entries whose value is None."""
    return {name: value for name, value in values.items() if value is not None}


def _query_values(name: str, value: Any) -> list[tuple[str, str]]:
    """Serialize nested query values using repeated keys and bracket notation."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [pair for item in value for pair in _query_values(name, item)]
    if isinstance(value, Mapping):
        return [
            pair
            for key, item in value.items()
            for pair in _query_values(f"{name}[{key}]", item)
        ]
    if isinstance(value, bool):
        return [(name, "true" if value else "false")]
    return [(name, str(value))]


def _error_message(payload: Any, fallback: str) -> str:
    """Extract an agent-readable API error message from a response payload."""
    if isinstance(payload, Mapping):
        for key in ("message", "error", "error_description"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return fallback or "Request failed"


class ContextDevBaseTool(BaseTool):
    """Shared authenticated transport for Context.dev CrewAI tools."""

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(value)
            if (value := os.getenv("CONTEXT_API_KEY", "").strip())
            else None
        ),
        exclude=True,
        repr=False,
    )
    api_base: str = Field(
        default_factory=lambda: (
            os.getenv("CONTEXT_API_BASE", "").strip() or DEFAULT_CONTEXT_API_BASE
        ).rstrip("/"),
        description="Context.dev API base URL.",
    )
    timeout: float = Field(
        default=DEFAULT_CONTEXT_TIMEOUT_SECONDS,
        gt=0,
        description="HTTP request timeout in seconds.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="CONTEXT_API_KEY",
                description="API key for Context.dev",
                required=True,
            ),
            EnvVar(
                name="CONTEXT_API_BASE",
                description="Optional Context.dev API base URL",
                required=False,
            ),
        ]
    )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        content: bytes | None = None,
    ) -> Any:
        """Send one authenticated request and return the public response payload."""
        if self.api_key is None:
            raise ValueError(
                "Context.dev API key missing. Pass api_key or set CONTEXT_API_KEY."
            )
        parsed_api_base = urlparse(self.api_base)
        if (
            parsed_api_base.scheme not in {"http", "https"}
            or not parsed_api_base.hostname
        ):
            raise ValueError("Context.dev API base must be an HTTP(S) URL.")
        if parsed_api_base.scheme == "http" and parsed_api_base.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError(
                "Context.dev API base must use HTTPS outside local development."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "User-Agent": "crewai-tools/context-dev",
        }
        if content is not None:
            headers["Content-Type"] = "application/octet-stream"

        request_kwargs: dict[str, Any] = {
            "method": method,
            "url": f"{self.api_base}{path}",
            "headers": headers,
            "params": [
                pair
                for name, value in (params or {}).items()
                for pair in _query_values(name, value)
            ],
        }
        if json_body is not None:
            request_kwargs["json"] = dict(json_body)
        if content is not None:
            request_kwargs["data"] = content

        response = self._send_with_retries(request_kwargs)

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not response.ok:
            message = _error_message(payload, response.text.strip())
            raise RuntimeError(f"Context.dev API {response.status_code}: {message}")
        if isinstance(payload, Mapping):
            return {
                name: value
                for name, value in payload.items()
                if name not in OMITTED_RESPONSE_FIELDS
            }
        return payload if payload is not None else response.text

    def _send_with_retries(
        self, request_kwargs: Mapping[str, Any]
    ) -> requests.Response:
        for attempt in range(MAX_CONTEXT_REQUEST_ATTEMPTS):
            try:
                response = requests.request(timeout=self.timeout, **request_kwargs)
            except requests.RequestException as exc:
                raise RuntimeError(f"Failed to reach Context.dev: {exc}") from exc

            if response.status_code != 429 and response.status_code < 500:
                return response
            if attempt == MAX_CONTEXT_REQUEST_ATTEMPTS - 1:
                return response

            time.sleep(_retry_delay_seconds(response, attempt))

        raise AssertionError("Context.dev retry loop exited unexpectedly")


def _retry_delay_seconds(response: requests.Response, attempt: int) -> float:
    """Return a bounded Retry-After or exponential-backoff delay."""
    retry_after_value = response.headers.get("Retry-After")
    if retry_after_value is not None:
        try:
            retry_after = float(str(retry_after_value))
            return min(max(retry_after, 0.0), MAX_CONTEXT_RETRY_DELAY_SECONDS)
        except (TypeError, ValueError):
            pass
    return float(min(0.5 * (2**attempt), MAX_CONTEXT_RETRY_DELAY_SECONDS))
