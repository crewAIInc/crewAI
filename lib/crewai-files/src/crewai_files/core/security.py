"""URL validation utilities for crewai-files.

Provides SSRF protection by resolving DNS and checking that target IPs
are not private or reserved before allowing HTTP requests.

Set CREWAI_FILES_ALLOW_UNSAFE_URLS=true to bypass validation (not
recommended for production).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

_UNSAFE_URLS_ENV = "CREWAI_FILES_ALLOW_UNSAFE_URLS"

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


def _is_escape_hatch_enabled() -> bool:
    """Check if the unsafe URLs escape hatch is enabled."""
    return os.environ.get(_UNSAFE_URLS_ENV, "").lower() in ("true", "1", "yes")


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
        return True  # If we can't parse, block it


def validate_url(url: str) -> str:
    """Validate that a URL is safe to fetch (SSRF protection).

    Blocks ``file://`` scheme entirely. For ``http``/``https``, resolves
    DNS and checks that the target IP is not private or reserved.

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
            "%s is enabled - skipping URL validation for: %s",
            _UNSAFE_URLS_ENV,
            url,
        )
        return url

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {url}. Only http and https are allowed.")

    if not parsed.hostname:
        raise ValueError(f"URL has no hostname: '{url}'")

    try:
        addrinfos = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: '{parsed.hostname}'") from exc

    for _family, _, _, _, sockaddr in addrinfos:
        ip_str = str(sockaddr[0])
        if _is_private_or_reserved(ip_str):
            raise ValueError(
                f"URL '{url}' resolves to private/reserved IP {ip_str}. "
                f"Access to internal networks is not allowed. "
                f"Set {_UNSAFE_URLS_ENV}=true to bypass."
            )

    return url
