"""HTTP helpers that preserve crewai-tools URL safety checks."""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from crewai_tools.security.safe_path import validate_url


_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_SENSITIVE_HEADER_NAMES = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
}
_SENSITIVE_HEADER_FRAGMENTS = ("api-key", "apikey", "secret", "token")


def _same_origin(previous_url: str, next_url: str) -> bool:
    previous = urlparse(previous_url)
    next_ = urlparse(next_url)
    return (previous.scheme, previous.netloc) == (next_.scheme, next_.netloc)


def _is_sensitive_header(header: str) -> bool:
    normalized = header.lower()
    return (
        normalized in _SENSITIVE_HEADER_NAMES
        or normalized.startswith("authorization-")
        or any(fragment in normalized for fragment in _SENSITIVE_HEADER_FRAGMENTS)
    )


def _strip_cross_origin_credentials(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    sanitized = {**request_kwargs}
    headers = sanitized.get("headers")
    if headers:
        sanitized["headers"] = {
            key: value
            for key, value in headers.items()
            if not _is_sensitive_header(str(key))
        }
    sanitized.pop("cookies", None)
    return sanitized


def safe_get(url: str, *, max_redirects: int = 10, **kwargs: Any) -> requests.Response:
    """GET a URL while validating each redirect target before following it."""
    current_url = validate_url(url)
    request_kwargs = {**kwargs, "allow_redirects": False}
    timeout = request_kwargs.pop("timeout", 30)
    history: list[requests.Response] = []
    redirects_followed = 0

    while True:
        response = requests.get(current_url, timeout=timeout, **request_kwargs)
        if (
            response.status_code not in _REDIRECT_STATUS_CODES
            or "Location" not in response.headers
        ):
            response.history = history
            return response

        if redirects_followed >= max_redirects:
            response.close()
            raise ValueError(f"Too many redirects while fetching URL: {url}")

        location = response.headers.get("Location")
        if not location:
            response.history = history
            return response

        try:
            redirect_url = validate_url(urljoin(response.url, location))
        except ValueError:
            response.close()
            raise

        if not _same_origin(current_url, redirect_url):
            request_kwargs = _strip_cross_origin_credentials(request_kwargs)

        history.append(response)
        current_url = redirect_url
        redirects_followed += 1
