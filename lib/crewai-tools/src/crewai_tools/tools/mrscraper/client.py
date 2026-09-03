"""Shared, secret-safe HTTP transport for MrScraper tools."""

import json
import re
from typing import Any, Literal

import requests


Origin = Literal["primary", "serp", "rendered"]

_ORIGINS: dict[Origin, str] = {
    "primary": "https://api.app.mrscraper.com",
    "serp": "https://sync.scraper.mrscraper.com",
    "rendered": "https://api.mrscraper.com",
}
_TIMEOUT = (10, 660)
_ERROR_BODY_LIMIT = 1000
_TOKEN_QUERY_RE = re.compile(r"([?&]token=)[^&\s]+", re.IGNORECASE)


class MrScraperClient:
    """Make requests only to MrScraper's fixed API origins."""

    def __init__(self, token: str, *, session: requests.Session | None = None) -> None:
        """Initialize a client with a nonblank token and optional HTTP session."""
        if not token.strip():
            raise ValueError("MRSCRAPER_API_TOKEN must be a nonblank value")
        self._token = token
        self._session = session or requests.Session()

    def request(
        self,
        method: Literal["GET", "POST"],
        origin: Origin,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        force_text: bool = False,
    ) -> str:
        """Send one request and return deterministic JSON text or exact response text."""
        url = f"{_ORIGINS[origin]}{path}"
        headers = self._headers(origin)
        if origin == "rendered":
            params = {
                "token": self._token,
                "browserRendering": "true",
                **(params or {}),
            }
        try:
            response = self._session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=_TIMEOUT,
            )
        except requests.RequestException as exc:
            detail = self._sanitize(str(exc))
            raise RuntimeError(f"MrScraper request failed: {detail}") from None

        if not 200 <= response.status_code < 300:
            sanitized = self._sanitize(response.text)
            body = sanitized[:_ERROR_BODY_LIMIT]
            suffix = "…" if len(sanitized) > _ERROR_BODY_LIMIT else ""
            raise RuntimeError(
                f"MrScraper API error (HTTP {response.status_code}): {body}{suffix}"
            )

        if force_text:
            return self._sanitize(response.text)

        content_type = response.headers.get("Content-Type", "").lower()
        if "json" not in content_type:
            return self._sanitize(response.text)
        try:
            value = response.json()
        except requests.JSONDecodeError:
            return self._sanitize(response.text)
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return self._sanitize(serialized)

    def _headers(self, origin: Origin) -> dict[str, str]:
        """Build the authentication headers required by an API origin."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if origin == "primary":
            headers["x-api-token"] = self._token
        elif origin == "serp":
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _sanitize(self, value: str) -> str:
        """Redact the configured token from response and error text."""
        redacted = value.replace(self._token, "[REDACTED]")
        return _TOKEN_QUERY_RE.sub(r"\1[REDACTED]", redacted)


__all__ = ["MrScraperClient"]
