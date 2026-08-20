from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field, SecretStr
import requests


DEFAULT_CONTEXT_API_BASE = "https://api.context.dev/v1"
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 180.0
OMITTED_RESPONSE_FIELDS = {"debug", "key_metadata", "request_id", "trace_id"}


def compact(values: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in values.items() if value is not None}


def _query_values(name: str, value: Any) -> list[tuple[str, str]]:
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
    if isinstance(payload, Mapping):
        for key in ("message", "error", "error_description"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return fallback or "Request failed"


class ContextDevBaseTool(BaseTool):
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
        default_factory=lambda: os.getenv(
            "CONTEXT_API_BASE", DEFAULT_CONTEXT_API_BASE
        ).rstrip("/"),
        exclude=True,
    )
    timeout: float = Field(
        default=DEFAULT_CONTEXT_TIMEOUT_SECONDS,
        gt=0,
        exclude=True,
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
        if self.api_key is None:
            raise ValueError(
                "Context.dev API key missing. Pass api_key or set CONTEXT_API_KEY."
            )
        if not self.api_base.startswith(("http://", "https://")):
            raise ValueError("Context.dev API base must be an HTTP(S) URL.")

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

        try:
            response = requests.request(timeout=self.timeout, **request_kwargs)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reach Context.dev: {exc}") from exc

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
