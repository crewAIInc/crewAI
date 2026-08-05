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
_BYPASS_HINT = f"Set {_UNSAFE_PATHS_ENV}=true to bypass this check."


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


def format_sandbox_error(error: Exception, remedy: str) -> str:
    """Restate a containment rejection with a tool-specific remedy.

    Rejections from :func:`validate_file_path` end by advertising the
    process-wide escape hatch, which also disables the SSRF checks on
    URL-fetching tools. Tools that accept a narrower ``base_dir`` should point
    at that instead, so callers reach for the blunt instrument last.

    Args:
        error: The rejection raised by path validation.
        remedy: Guidance to offer in place of the escape-hatch advice.

    Returns:
        The rejection text with *remedy* substituted for the bypass advice.
    """
    text = str(error)
    if text.endswith(_BYPASS_HINT):
        text = text[: -len(_BYPASS_HINT)].rstrip()
    return f"{text} {remedy}".strip()


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

    # Ensure the resolved path is within the base directory.
    # When resolved_base already ends with a separator (e.g. the filesystem
    # root "/"), appending os.sep would double it ("//"), so use the base
    # as-is in that case.
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


def _is_ipv4(ip_str: str) -> bool:
    try:
        return ipaddress.ip_address(ip_str).version == 4
    except ValueError:
        return False


def validate_and_resolve(url: str) -> tuple[str, str | None]:
    """Validate that a URL is safe to fetch, and return the specific IP address
    checked so a caller can pin its actual connection to it.

    Blocks ``file://`` scheme entirely. For ``http``/``https``, resolves DNS
    and checks that every returned address is not private or reserved
    (prevents SSRF to internal services and cloud metadata endpoints).

    Returning the checked IP alongside the URL matters: validating a hostname
    and then letting the HTTP client re-resolve it later at connection time
    leaves a DNS-rebinding gap -- an attacker with control over DNS (or a
    short-TTL record) can present a safe IP for validation and a private one
    for the real connection moments later. Callers that make the actual
    request should connect to the returned IP directly (see
    ``safe_requests.safe_get``'s use of this) rather than handing the
    hostname back to their HTTP client and trusting it to resolve the same
    way twice.

    Args:
        url: The URL to validate.

    Returns:
        A ``(validated_url, ip)`` tuple. Every resolved address is checked,
        but ``ip`` prefers an IPv4 address if one was returned (falling back
        to the first address otherwise) -- ``getaddrinfo`` on a dual-stack
        host often returns IPv6 first, and pinning the connection to a single
        address (see ``safe_requests.safe_get``) forgoes the normal
        multi-address fallback an unpinned connection attempt would get, so
        picking the address most likely to actually be reachable (many CI
        runners, containers, and networks have working IPv4 but broken or
        absent IPv6 routes) matters more here than it would otherwise.
        ``None`` when validation is bypassed via the escape hatch (nothing
        was resolved, so there is nothing to pin).

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
        return url, None

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

    try:
        addrinfos = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: '{parsed.hostname}'") from exc

    checked_ips: list[str] = []
    for _family, _, _, _, sockaddr in addrinfos:
        ip_str = str(sockaddr[0])
        if _is_private_or_reserved(ip_str):
            raise ValueError(
                f"URL '{url}' resolves to private/reserved IP {ip_str}. "
                f"Access to internal networks is not allowed. "
                f"Set {_UNSAFE_PATHS_ENV}=true to bypass."
            )
        checked_ips.append(ip_str)

    pinned_ip = next((ip for ip in checked_ips if _is_ipv4(ip)), None) or (
        checked_ips[0] if checked_ips else None
    )
    return url, pinned_ip


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
    validated_url, _ip = validate_and_resolve(url)
    return validated_url


def resolve_validated_ip(url: str) -> str | None:
    """Validate `url` (same checks as :func:`validate_url`) and return the
    specific IP address that was checked, for pinning an actual connection to
    it. See :func:`validate_and_resolve` for why this matters. Returns
    ``None`` when validation is bypassed via the escape hatch.
    """
    _url, ip = validate_and_resolve(url)
    return ip
