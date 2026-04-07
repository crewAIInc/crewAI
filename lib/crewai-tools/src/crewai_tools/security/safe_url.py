"""URL validation to prevent SSRF attacks."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def validate_url(url: str) -> str:
    """Validate that a URL is safe for outbound requests.

    Args:
        url: The URL to validate.

    Returns:
        The validated URL.

    Raises:
        ValueError: If the URL uses a blocked scheme, resolves to a
            private/loopback/link-local IP, or is otherwise invalid.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme {parsed.scheme!r} is not allowed. "
            "Only http and https are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid URL: no hostname found in {url!r}")

    try:
        resolved = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror as err:
        raise ValueError(f"Cannot resolve hostname: {hostname!r}") from err

    for _family, _, _, _, sockaddr in resolved:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(
                f"URL {url!r} resolves to a private/reserved address "
                f"({ip}). Requests to internal networks are blocked."
            )

    return url
