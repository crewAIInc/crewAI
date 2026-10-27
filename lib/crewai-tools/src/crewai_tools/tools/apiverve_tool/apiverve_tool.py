"""Tools for the APIVerve catalog of REST APIs.

Requests go only to ``https://api.apiverve.com`` with the key in the ``x-api-key``
header. Redirects are not followed, so the key is never forwarded to another host,
and the key is stripped from anything returned to the agent. The public catalog is
read from ``https://assets.apiverve.com`` and cached for an hour.
"""

import json
import os
import re
import threading
import time
from typing import Any
from urllib.parse import quote

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


API_BASE = "https://api.apiverve.com"
CATALOG_URL = "https://assets.apiverve.com/cli-manifest.json"
CATALOG_TTL_SECONDS = 3600.0
REQUEST_TIMEOUT_SECONDS = 60

_catalog: dict[str, dict[str, Any]] | None = None
_catalog_at = 0.0
_catalog_lock = threading.Lock()


def _redact(value: Any, secret: str) -> Any:
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "[redacted]")
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(k, secret): _redact(v, secret) for k, v in value.items()}
    return value


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_query_value(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _load_catalog() -> dict[str, dict[str, Any]]:
    """Return ``{api_id: definition}`` from the public catalog, cached for an hour."""
    global _catalog, _catalog_at
    with _catalog_lock:
        if (
            _catalog is not None
            and time.monotonic() - _catalog_at < CATALOG_TTL_SECONDS
        ):
            return _catalog
        try:
            response = requests.get(
                CATALOG_URL,
                headers={"accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False,
            )
            response.raise_for_status()
            apis = response.json()["apis"]
            if not isinstance(apis, dict):
                raise TypeError("Catalog 'apis' must be an object.")
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            if _catalog is not None:
                # Serve stale data and wait a full TTL before retrying.
                _catalog_at = time.monotonic()
                return _catalog
            raise RuntimeError(
                "The APIVerve catalog could not be loaded. Retry in a moment."
            ) from exc
        _catalog, _catalog_at = apis, time.monotonic()
        return apis


def _describe(api_id: str, api: dict[str, Any]) -> dict[str, Any]:
    return {
        "api": api_id,
        "title": api.get("title"),
        "category": api.get("category"),
        "description": api.get("description"),
        "parameters": [
            {
                "name": p.get("name"),
                "type": p.get("type"),
                "required": bool(p.get("required")),
                "description": p.get("description"),
                "example": p.get("example"),
            }
            for p in api.get("params", [])
        ],
    }


def _error_message(status: int, payload: Any) -> str:
    detail = ""
    if isinstance(payload, dict):
        detail = str(payload.get("error") or payload.get("message") or "")
    if status == 401:
        return "APIVerve rejected the API key."
    if status == 403:
        return f"This API key is not allowed to make this call{': ' + detail if detail else '.'}"
    if status == 429:
        return (
            "Rate limit reached or monthly credits used up. Retry shortly, "
            "or upgrade at https://dashboard.apiverve.com."
        )
    if detail:
        return f"APIVerve returned HTTP {status}: {detail}"
    return f"APIVerve returned HTTP {status}."


class ApiverveFindApisToolSchema(BaseModel):
    """Input for ApiverveFindApisTool."""

    query: str = Field(
        ...,
        description="What you need, in a few words, for example 'validate email' or 'vat rates'.",
    )


class ApiverveFindApisTool(BaseTool):
    """Search the APIVerve catalog for the API that fits a task."""

    name: str = "Find APIVerve APIs"
    description: str = (
        "Search the APIVerve catalog of about 380 APIs (validation, IP and DNS lookups, "
        "currency and tax rates, documents, images and more). Returns matching API ids "
        "with their parameters, to pass to the Run APIVerve API tool."
    )
    args_schema: type[BaseModel] = ApiverveFindApisToolSchema
    max_results: int = 10
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])

    def _run(self, query: str, **_: Any) -> str:
        try:
            apis = _load_catalog()
        except RuntimeError as exc:
            return f"Error: {exc}"
        words = [w for w in (query or "").lower().split() if w]
        scored = []
        for api_id, api in apis.items():
            title = (api.get("title") or "").lower()
            category = (api.get("category") or "").lower()
            description = (api.get("description") or "").lower()
            score = 0
            for word in words:
                # Word-start matches, so "vat" finds VAT Rates but not Gravatar.
                starts = re.compile(r"\b" + re.escape(word))
                if api_id.startswith(word) or (len(word) >= 4 and word in api_id):
                    score += 4
                if starts.search(title):
                    score += 4
                if starts.search(category):
                    score += 2
                if starts.search(description):
                    score += 1
            if score or not words:
                scored.append((score, api_id, api))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return json.dumps(
            [_describe(api_id, api) for _, api_id, api in scored[: self.max_results]]
        )


class ApiverveRunApiToolSchema(BaseModel):
    """Input for ApiverveRunApiTool."""

    api: str = Field(
        ...,
        description="The APIVerve API id, for example emailvalidator, iplookup or vindecoder.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description='The API parameters, for example {"email": "jane@example.com"}.',
    )


class ApiverveRunApiTool(BaseTool):
    """Call one APIVerve API by id and return its response data."""

    name: str = "Run APIVerve API"
    description: str = (
        "Call any APIVerve API by id with its parameters and get the result as JSON: "
        "validate emails and phone numbers, look up IPs, DNS and SSL certificates, "
        "get exchange rates, decode VINs, and more. Use Find APIVerve APIs to look up ids."
    )
    args_schema: type[BaseModel] = ApiverveRunApiToolSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="APIVERVE_API_KEY",
                description="API key for APIVerve, from https://dashboard.apiverve.com",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])

    def _run(self, api: str, parameters: dict[str, Any] | None = None, **_: Any) -> str:
        api_key = os.environ.get("APIVERVE_API_KEY", "").strip()
        if not api_key:
            return "Error: APIVERVE_API_KEY environment variable is required"
        try:
            apis = _load_catalog()
        except RuntimeError as exc:
            return f"Error: {exc}"

        api_id = (api or "").strip().lower()
        definition = apis.get(api_id)
        if definition is None:
            return f'Error: Unknown APIVerve API "{api_id}". Use Find APIVerve APIs to look up the id.'
        params = {
            k: v for k, v in (parameters or {}).items() if v is not None and v != ""
        }
        declared = {p["name"]: p for p in definition.get("params", [])}
        missing = [
            n for n, p in declared.items() if p.get("required") and n not in params
        ]
        if missing:
            return f"Error: {api_id} needs: {', '.join(missing)}."
        path = definition.get("path") or ""
        if not path.startswith("/v1/"):
            return f'Error: The catalog entry for "{api_id}" is invalid.'

        method = (
            "POST" if (definition.get("method") or "GET").upper() == "POST" else "GET"
        )
        query: dict[str, str] = {}
        body: dict[str, Any] = {}
        for name, value in params.items():
            location = declared.get(name, {}).get("in")
            if location == "query" or (location is None and method == "GET"):
                query[name] = _query_value(value)
            else:
                body[name] = value

        try:
            response = requests.request(
                method,
                API_BASE + quote(path, safe="/"),
                params=query,
                json=body if method == "POST" else None,
                headers={"x-api-key": api_key, "accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False,
            )
        except requests.Timeout:
            return "Error: APIVerve did not respond in time. Retry in a moment."
        except requests.RequestException:
            return "Error: APIVerve could not be reached. Retry in a moment."

        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not response.ok or (
            isinstance(payload, dict) and payload.get("status") == "error"
        ):
            message = _error_message(response.status_code, payload)
            return "Error: " + message.replace(api_key, "[redacted]")
        if payload is None:
            return "Error: APIVerve returned a non-JSON response."
        result = payload.get("data", payload) if isinstance(payload, dict) else payload
        return json.dumps(_redact(result, api_key), ensure_ascii=False)
