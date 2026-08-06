"""HTTP helpers that preserve crewai-tools URL safety checks."""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from crewai_tools.security.safe_path import validate_url


_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_STREAM_CHUNK_SIZE = 65536
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
    """GET a URL while validating each redirect target before following it.

    On success the hops are attached to the returned response's ``history`` and
    are the caller's to close. On failure they are closed here: a caller given
    an exception has no handle on them, and a streamed hop holds its connection
    until its body is read or closed.
    """
    current_url = validate_url(url)
    request_kwargs = {**kwargs, "allow_redirects": False}
    timeout = request_kwargs.pop("timeout", 30)
    history: list[requests.Response] = []
    redirects_followed = 0

    try:
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
    except BaseException:
        for hop in history:
            hop.close()
        raise


def safe_get_bounded(
    url: str,
    *,
    max_bytes: int,
    timeout: float | tuple[float, float] = 30,
    headers: dict[str, str] | None = None,
    max_redirects: int = 10,
) -> tuple[bytes, str, str]:
    """GET a URL through :func:`safe_get`, refusing bodies over *max_bytes*.

    The body is streamed and abandoned as soon as it crosses the limit, so an
    oversized response costs one chunk of memory instead of all of it. The cap
    counts decoded bytes, which is what a compressed response expands into --
    ``Content-Length`` describes the wire size and cannot bound that.

    Args:
        url: The URL to fetch.
        max_bytes: Largest body to accept, in decoded bytes.
        timeout: Request timeout, passed through to requests.
        headers: Request headers.
        max_redirects: Hops to follow before giving up.

    Returns:
        A ``(body, content_type, final_url)`` tuple, where *final_url* is the
        last validated URL in the redirect chain.

    Raises:
        ValueError: If *max_bytes* is not positive, URL validation fails, the
            redirect chain is too long, or the body exceeds *max_bytes*.
        requests.RequestException: If the request fails or returns an error
            status.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}.")

    response = safe_get(
        url,
        max_redirects=max_redirects,
        headers=headers,
        timeout=timeout,
        stream=True,
    )
    try:
        response.raise_for_status()

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=_STREAM_CHUNK_SIZE):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                # Names the URL that served the body, which after a redirect is
                # not the one that was requested.
                raise ValueError(
                    f"Response body from '{response.url}' exceeds the "
                    f"{max_bytes} byte limit."
                )
            chunks.append(chunk)

        return (
            b"".join(chunks),
            response.headers.get("Content-Type", ""),
            response.url,
        )
    finally:
        # Under stream=True each hop holds its connection until the body is read,
        # so the redirects need closing too, not just the response we return.
        for hop in response.history:
            hop.close()
        response.close()
