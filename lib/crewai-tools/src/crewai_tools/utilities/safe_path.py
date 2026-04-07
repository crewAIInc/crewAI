"""Path and URL validation utilities for crewai-tools.

Provides validation for file paths and URLs to prevent unauthorized
file access and server-side request forgery (SSRF) when tools accept
user-controlled or LLM-controlled inputs at runtime.

Set CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true to bypass validation (not
recommended for production).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

_UNSAFE_PATHS_ENV = "CREWAI_TOOLS_ALLOW_UNSAFE_PATHS"


def _is_escape_hatch_enabled() -> bool:
    """Check if the unsafe paths escape hatch is enabled."""
    return os.environ.get(_UNSAFE_PATHS_ENV, "").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# File path validation
# ---------------------------------------------------------------------------


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

    # Ensure the resolved path is within the base directory.
    # When resolved_base already ends with a separator (e.g. the filesystem
    # root "/"), appending os.sep would double it ("//"), so use the base
    # as-is in that case.
    prefix = resolved_base if resolved_base.endswith(os.sep) else resolved_base + os.sep
    if not resolved_path.startswith(prefix) and resolved_path != resolved_base:
        raise ValueError(
            f"Path '{path}' resolves to '{resolved_path}' which is outside "
            f"the allowed directory '{resolved_base}'. "
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


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

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
        # Unwrap IPv4-mapped IPv6 addresses (e.g., ::ffff:127.0.0.1) to IPv4
        # so they are only checked against IPv4 networks (avoids TypeError when
        # an IPv4Address is compared against an IPv6Network).
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        networks = (
            _BLOCKED_IPV4_NETWORKS
            if isinstance(addr, ipaddress.IPv4Address)
            else _BLOCKED_IPV6_NETWORKS
        )
        return any(addr in network for network in networks)
    except ValueError:
        return True  # If we can't parse, block it


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
    if _is_escape_hatch_enabled():
        logger.warning(
            "%s is enabled — skipping URL validation for: %s",
            _UNSAFE_PATHS_ENV,
            url,
        )
        return url

    parsed = urlparse(url)

    # Block file:// scheme
    if parsed.scheme == "file":
        raise ValueError(
            f"file:// URLs are not allowed: '{url}'. "
            f"Use a file path instead, or set {_UNSAFE_PATHS_ENV}=true to bypass."
        )

    # Only allow http and https
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are supported."
        )

    if not parsed.hostname:
        raise ValueError(f"URL has no hostname: '{url}'")

    # Resolve DNS and check IPs
    try:
        addrinfos = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: '{parsed.hostname}'") from exc

    for _family, _, _, _, sockaddr in addrinfos:
        ip_str = str(sockaddr[0])
        if _is_private_or_reserved(ip_str):
            raise ValueError(
                f"URL '{url}' resolves to private/reserved IP {ip_str}. "
                f"Access to internal networks is not allowed. "
                f"Set {_UNSAFE_PATHS_ENV}=true to bypass."
            )

    return url
