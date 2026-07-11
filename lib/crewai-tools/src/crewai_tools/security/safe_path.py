"""Path and URL validation utilities for crewai-tools.

Provides validation for file paths and URLs to prevent unauthorized
file access and server-side request forgery (SSRF) when tools accept
user-controlled or LLM-controlled inputs at runtime.

Set CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true to bypass validation (not
recommended for production).
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import logging
import os
import socket
from typing import Any
from urllib.parse import urlparse

from requests.adapters import HTTPAdapter


logger = logging.getLogger(__name__)

_UNSAFE_PATHS_ENV = "CREWAI_TOOLS_ALLOW_UNSAFE_PATHS"


def format_path_for_display(path: str, base_dir: str | None = None) -> str:
    """Return a path label that does not expose absolute directory prefixes."""
    if base_dir is None:
        base_dir = os.getcwd()

    try:
        resolved_base = os.path.realpath(base_dir)
        resolved_path = os.path.realpath(
            os.path.join(resolved_base, path) if not os.path.isabs(path) else path
        )
        if os.path.commonpath([resolved_base, resolved_path]) == resolved_base:
            return os.path.relpath(resolved_path, resolved_base)
    except (OSError, ValueError) as exc:
        logger.debug("Falling back to basename for display path formatting: %s", exc)

    return os.path.basename(os.path.realpath(path)) or "[redacted path]"


def format_error_for_display(error: Exception) -> str:
    """Return exception details without OS-added absolute path context."""
    if isinstance(error, OSError):
        return error.strerror or error.__class__.__name__
    return str(error)


def _is_escape_hatch_enabled() -> bool:
    """Check if the unsafe paths escape hatch is enabled."""
    return os.environ.get(_UNSAFE_PATHS_ENV, "").lower() in ("true", "1", "yes")


def validate_file_path(path: str, base_dir: str | None = None) -> str:
    """Validate that a file path is safe to read.

    Resolves symlinks and ``..`` components, then checks that the resolved
    path falls within *base_dir* (defaults to the current working directory).

    Args:
        path: The file path to validate.
        base_dir: Allowed root directory. Defaults to ``os.getcwd()``.

    Returns:
        The resolved, validated absolute path.

    Raises:
        ValueError: If the path escapes the allowed directory.
    """
    if _is_escape_hatch_enabled():
        logger.warning(
            "%s is enabled — skipping file path validation for: %s",
            _UNSAFE_PATHS_ENV,
            path,
        )
        return os.path.realpath(path)

    if base_dir is None:
        base_dir = os.getcwd()

    resolved_base = os.path.realpath(base_dir)
    resolved_path = os.path.realpath(
        os.path.join(resolved_base, path) if not os.path.isabs(path) else path
    )

    prefix = resolved_base if resolved_base.endswith(os.sep) else resolved_base + os.sep
    if not resolved_path.startswith(prefix) and resolved_path != resolved_base:
        raise ValueError(
            f"Path '{format_path_for_display(resolved_path, resolved_base)}' is "
            f"outside the allowed directory. "
            f"Set {_UNSAFE_PATHS_ENV}=true to bypass this check."
        )

    return resolved_path


def validate_directory_path(path: str, base_dir: str | None = None) -> str:
    """Validate that a directory path is safe to read.

    Same as :func:`validate_file_path` but also checks that the path
    is an existing directory.

    Args:
        path: The directory path to validate.
        base_dir: Allowed root directory. Defaults to ``os.getcwd()``.

    Returns:
        The resolved, validated absolute path.

    Raises:
        ValueError: If the path escapes the allowed directory or is not a directory.
    """
    validated = validate_file_path(path, base_dir)
    if not os.path.isdir(validated):
        raise ValueError(f"Path '{validated}' is not a directory.")
    return validated


# Private and reserved IP ranges that should not be accessed
_BLOCKED_IPV4_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / cloud metadata
    ipaddress.ip_network("0.0.0.0/32"),
]

_BLOCKED_IPV6_NETWORKS = [
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("fc00::/7"),  # Unique local addresses
    ipaddress.ip_network("fe80::/10"),  # Link-local IPv6
]


def _is_private_or_reserved(ip_str: str) -> bool:
    """Check if an IP address is private, reserved, or otherwise unsafe."""
    try:
        addr = ipaddress.ip_address(ip_str)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        networks = (
            _BLOCKED_IPV4_NETWORKS
            if isinstance(addr, ipaddress.IPv4Address)
            else _BLOCKED_IPV6_NETWORKS
        )
        return any(addr in network for network in networks)
    except ValueError:
        return True


def _validate_url_common(url: str) -> tuple[str, str, str | None]:
    """Shared validation logic for URL schemes, hostnames, and DNS resolution.

    Returns:
        Tuple of (scheme, hostname, resolved_ip) on success.

    Raises:
        ValueError: If the URL is unsafe.
    """
    if _is_escape_hatch_enabled():
        logger.warning(
            "%s is enabled — skipping URL validation for: %s",
            _UNSAFE_PATHS_ENV,
            url,
        )
        return "", "", None

    parsed = urlparse(url)

    if parsed.scheme == "file":
        raise ValueError(
            f"file:// URLs are not allowed: '{url}'. "
            f"Use a file path instead, or set {_UNSAFE_PATHS_ENV}=true to bypass."
        )

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are supported."
        )

    if not parsed.hostname:
        raise ValueError(f"URL has no hostname: '{url}'")

    try:
        addrinfos = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: '{parsed.hostname}'") from exc

    resolved_ip = None
    for _family, _, _, _, sockaddr in addrinfos:
        ip_str = str(sockaddr[0])
        if _is_private_or_reserved(ip_str):
            raise ValueError(
                f"URL '{url}' resolves to private/reserved IP {ip_str}. "
                f"Access to internal networks is not allowed. "
                f"Set {_UNSAFE_PATHS_ENV}=true to bypass."
            )
        if resolved_ip is None:
            resolved_ip = ip_str

    return parsed.scheme, parsed.hostname, resolved_ip


def validate_url(url: str) -> str:
    """Validate that a URL is safe to fetch.

    Blocks ``file://`` scheme entirely. For ``http``/``https``, resolves
    DNS and checks that the target IP is not private or reserved (prevents
    SSRF to internal services and cloud metadata endpoints).

    Args:
        url: The URL to validate.

    Returns:
        The validated URL string.

    Raises:
        ValueError: If the URL uses a blocked scheme or resolves to a
            private/reserved IP address.
    """
    _validate_url_common(url)
    return url


@dataclass(frozen=True, slots=True)
class ValidatedURL:
    """A validated URL with its pinned resolved IP address.

    The resolved IP is captured during validation and used by
    :class:`PinnedIPAdapter` to eliminate the TOCTOU window between
    DNS resolution and TCP connection.
    """

    url: str
    resolved_ip: str


def validate_url_and_resolve(url: str) -> ValidatedURL:
    """Validate a URL and return its resolved IP for pinning.

    Same security checks as :func:`validate_url`, but additionally returns
    the resolved IP address so the caller can pin the connection to it,
    eliminating the DNS-rebinding TOCTOU window.

    Args:
        url: The URL to validate.

    Returns:
        A :class:`ValidatedURL` with the original URL and the resolved IP.

    Raises:
        ValueError: If the URL is unsafe or unresolvable.
    """
    _, _, resolved_ip = _validate_url_common(url)
    if resolved_ip is None:
        return ValidatedURL(url=url, resolved_ip="")
    return ValidatedURL(url=url, resolved_ip=resolved_ip)


class PinnedIPAdapter(HTTPAdapter):
    """urllib3 adapter that pins connections to a pre-resolved IP.

    This prevents DNS rebinding attacks: the IP is validated once, then
    the connection is made directly to that IP while preserving the
    original hostname for ``Host`` header and TLS SNI.

    Implementation: overrides ``send()`` to rewrite the request URL,
    replacing the hostname with the pinned IP and setting the ``Host``
    header to the original hostname. For HTTPS, ``server_hostname`` is
    set on the SSL context so TLS SNI works correctly.
    """

    def __init__(self, resolved_ip: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._resolved_ip = resolved_ip

    def send(self, request: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Send the request, rewriting the URL to use the pinned IP."""
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(request.url)
        original_host = parsed.hostname
        if not original_host or not self._resolved_ip:
            return super().send(request, **kwargs)

        # Rewrite URL: replace hostname with pinned IP
        pinned_netloc = self._resolved_ip
        if parsed.port:
            pinned_netloc = f"{self._resolved_ip}:{parsed.port}"
        if parsed.username:
            auth = parsed.username
            if parsed.password:
                auth += f":{parsed.password}"
            pinned_netloc = f"{auth}@{pinned_netloc}"

        pinned_url = urlunparse(
            parsed._replace(netloc=pinned_netloc)
        )
        request.url = pinned_url

        # Set Host header to original hostname for virtual hosting
        if request.headers.get("Host") is None:
            request.headers["Host"] = original_host
        if "host" not in {k.lower() for k in request.headers}:
            request.headers["Host"] = original_host

        # For HTTPS, ensure TLS SNI uses the original hostname
        if parsed.scheme == "https":
            kwargs.setdefault("server_hostname", original_host)

        return super().send(request, **kwargs)
