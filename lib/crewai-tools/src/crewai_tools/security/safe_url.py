"""URL validation to prevent SSRF attacks.

Returns a rewritten URL that connects to the resolved IP directly,
preventing DNS rebinding between validation and request time.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse, urlunparse


def validate_url(
    url: str,
    *,
    allow_private: bool = False,
    pin_ip: bool = True,
) -> str:
    """Validate that a URL is safe for outbound requests.

    Resolves the hostname and optionally rewrites the URL to use the
    resolved IP, preventing DNS rebinding attacks.

    Args:
        url: The URL to validate.
        allow_private: If True, skip the private/reserved IP check.
        pin_ip: If True, rewrite the URL to connect to the resolved IP.
            Set to False for tools that delegate to third-party SDKs
            where IP-based URLs would break TLS.

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

    if not resolved:
        raise ValueError(f"No addresses found for hostname: {hostname!r}")

    safe_ip: str | None = None
    for _family, _, _, _, sockaddr in resolved:
        ip = ipaddress.ip_address(sockaddr[0])
        if not allow_private and (
            ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        ):
            raise ValueError(
                f"URL {url!r} resolves to a private/reserved address "
                f"({ip}). Requests to internal networks are blocked. "
                "Pass allow_private=True to override."
            )
        if safe_ip is None:
            safe_ip = str(sockaddr[0])

    if not pin_ip:
        return url

    ip_obj = ipaddress.ip_address(safe_ip)  # type: ignore[arg-type]
    ip_host = f"[{safe_ip}]" if ip_obj.version == 6 else safe_ip
    port_suffix = f":{parsed.port}" if parsed.port else ""
    pinned_netloc = f"{ip_host}{port_suffix}"

    return urlunparse(parsed._replace(netloc=pinned_netloc))
