"""URL checks for endpoints advertised by a remote A2A AgentCard.

An AgentCard is remote-controlled input. Before credentials are sent to a URL
the card names, the URL must stay on the configured endpoint's origin and must
not point at internal infrastructure.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _split(url: str) -> tuple[str, str, int | None]:
    """Return (scheme, host, port) for a URL, tolerating ``host:port`` forms."""
    parts = urlsplit(url if "://" in url else f"//{url}")
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    return scheme, host, port or _DEFAULT_PORTS.get(scheme)


def is_same_origin(url: str, endpoint: str) -> bool:
    """Whether ``url`` stays on the origin of the configured ``endpoint``.

    HTTP(S) URLs must match scheme, host and port. URLs without an HTTP scheme
    (for example a gRPC ``host:port`` target) cannot share the HTTP origin, so
    only the host is compared.
    """
    scheme, host, port = _split(url)
    e_scheme, e_host, e_port = _split(endpoint)
    if not host or host != e_host:
        return False
    if scheme in _DEFAULT_PORTS:
        return scheme == e_scheme and port == e_port
    return True


def _is_internal(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def is_internal_host(host: str) -> bool:
    """Whether ``host`` is, or resolves to, a private/loopback/link-local address."""
    try:
        return _is_internal(ipaddress.ip_address(host))
    except ValueError:
        pass
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return any(_is_internal(ipaddress.ip_address(i[4][0])) for i in infos)


async def validate_card_url(url: str, endpoint: str) -> None:
    """Reject a card-advertised URL that is unsafe to send credentials to.

    Raises:
        ValueError: If the URL leaves the configured endpoint's origin, or, when
            the endpoint itself is public, resolves to an internal address.
    """
    if not is_same_origin(url, endpoint):
        raise ValueError(
            f"AgentCard advertises URL '{url}' which does not match the "
            f"configured endpoint '{endpoint}'. Refusing to send credentials."
        )
    _, host, _ = _split(url)
    _, endpoint_host, _ = _split(endpoint)
    if await is_internal_host(host) and not await is_internal_host(endpoint_host):
        raise ValueError(
            f"AgentCard URL '{url}' resolves to an internal address while the "
            f"configured endpoint '{endpoint}' does not."
        )
