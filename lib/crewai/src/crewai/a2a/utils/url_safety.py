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
_HTTP_SCHEMES = frozenset(_DEFAULT_PORTS)
_TLS_SCHEMES = frozenset({"https", "grpcs"})
_GRPC_SCHEMES = frozenset({"grpc", "grpcs"})


def _split(url: str) -> tuple[str, str, int | None] | None:
    """Return (scheme, host, port) of the host a client would dial, or None.

    Understands the forms the gRPC channel factory accepts: ``http(s)://``,
    ``grpc(s)://``, ``dns:///host:port`` and bare ``host:port`` (scheme ""). Any
    other scheme, userinfo, or a ``dns://`` authority yields None.
    """
    scheme, sep, rest = url.partition("://")
    if not sep:
        scheme, rest = "", url
    scheme = scheme.lower()
    if scheme == "dns":
        rest = rest.removeprefix("/")
        if rest.startswith("/"):
            return None
        rest = f"//{rest}"
    elif scheme in _HTTP_SCHEMES | _GRPC_SCHEMES or scheme == "":
        rest = f"//{rest}"
    else:
        return None
    parts = urlsplit(rest)
    if "@" in parts.netloc or not parts.hostname:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    return scheme, parts.hostname.lower(), port or _DEFAULT_PORTS.get(scheme)


def is_same_origin(url: str, endpoint: str) -> bool:
    """Whether a client dialing ``url`` stays on the configured ``endpoint``.

    HTTP(S) URLs must match scheme, host and port. gRPC targets
    (``grpc(s)://``, ``dns:///``, bare ``host:port``) must match the host, and
    when the endpoint is HTTPS must use TLS (``grpcs://``) so the connection is
    authenticated to that host. The port may differ: gRPC is commonly served on
    its own port.
    """
    target = _split(url)
    origin = _split(endpoint)
    if target is None or origin is None:
        return False
    scheme, host, port = target
    e_scheme, e_host, e_port = origin
    if host != e_host:
        return False
    if scheme in _HTTP_SCHEMES:
        return scheme == e_scheme and port == e_port
    return e_scheme != "https" or scheme in _TLS_SCHEMES


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
    host = _split(url)[1]  # type: ignore[index]
    endpoint_host = _split(endpoint)[1]  # type: ignore[index]
    if await is_internal_host(host) and not await is_internal_host(endpoint_host):
        raise ValueError(
            f"AgentCard URL '{url}' resolves to an internal address while the "
            f"configured endpoint '{endpoint}' does not."
        )
