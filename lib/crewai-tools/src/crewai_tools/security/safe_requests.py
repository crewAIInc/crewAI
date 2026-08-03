"""HTTP helpers that preserve crewai-tools URL safety checks."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from pathlib import Path
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from crewai_tools.security.safe_path import validate_and_resolve


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


@contextlib.contextmanager
def _pin_dns(hostname: str, ip: str) -> Iterator[None]:
    """Force ``socket.getaddrinfo(hostname, ...)`` to return only `ip` for the
    duration of the block, so the connection made inside it cannot be
    redirected to a different address by a DNS response that changes between
    validation and connection time (DNS rebinding).

    This patches ``socket.getaddrinfo`` process-wide rather than through a
    custom transport adapter, to stay compatible with plain ``requests``
    without extra dependencies. The window is only as long as the single
    request this wraps, and the original resolver is always restored in
    ``finally``. A concurrent request to a *different* host on another thread
    during that window is unaffected: the pinned resolver only intercepts
    lookups for `hostname` and delegates everything else to the real one.
    """
    original_getaddrinfo = socket.getaddrinfo
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET

    def pinned_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        if host == hostname:
            sockaddr = (ip, port, 0, 0) if family == socket.AF_INET6 else (ip, port)
            return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]
        return original_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = pinned_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def safe_get(url: str, *, max_redirects: int = 10, **kwargs: Any) -> requests.Response:
    """GET a URL while validating each redirect target before following it.

    Every hop -- the initial request and each redirect -- has its DNS
    resolution pinned to the specific IP address that was validated for that
    URL, closing the gap where a hostname could pass validation against a
    safe IP and then have the HTTP client re-resolve it to a different,
    unsafe address (e.g. a cloud metadata endpoint) at actual connection
    time.
    """
    current_url, pinned_ip = validate_and_resolve(url)
    request_kwargs = {**kwargs, "allow_redirects": False}
    timeout = request_kwargs.pop("timeout", 30)
    history: list[requests.Response] = []
    redirects_followed = 0

    while True:
        hostname = urlparse(current_url).hostname
        pin = _pin_dns(hostname, pinned_ip) if pinned_ip and hostname else contextlib.nullcontext()
        with pin:
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
            redirect_url, redirect_ip = validate_and_resolve(urljoin(response.url, location))
        except ValueError:
            response.close()
            raise

        if not _same_origin(current_url, redirect_url):
            request_kwargs = _strip_cross_origin_credentials(request_kwargs)

        history.append(response)
        current_url, pinned_ip = redirect_url, redirect_ip
        redirects_followed += 1


def safe_download(
    url: str,
    dest_path: str | Path,
    *,
    max_redirects: int = 10,
    chunk_size: int = 65536,
    **kwargs: Any,
) -> None:
    """Download `url` to `dest_path`, applying the exact same validation,
    redirect-chain revalidation, and DNS pinning as :func:`safe_get`, but
    streaming the response to disk instead of buffering it in memory.

    Raises:
        ValueError: If the URL (or any redirect target) fails validation.
        requests.HTTPError: If the final response has an error status code.
    """
    response = safe_get(url, max_redirects=max_redirects, stream=True, **kwargs)
    try:
        response.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
    finally:
        response.close()
