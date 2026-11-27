"""URL checks for endpoints advertised by a remote A2A AgentCard.

An AgentCard is remote-controlled input. Before credentials are sent to a URL
the card names, the URL must stay on the configured endpoint's origin and must
not point at internal infrastructure.
"""

from __future__ import annotations

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


def validate_card_url(url: str, endpoint: str) -> None:
    """Reject a card-advertised URL that is unsafe to send credentials to.

    The card is untrusted input, so the client may only talk to the endpoint the
    user configured. This does not pin the resolved IP: a hostname that rebinds
    between lookups is out of scope here.

    Raises:
        ValueError: If the URL leaves the configured endpoint's origin.
    """
    if not is_same_origin(url, endpoint):
        raise ValueError(
            f"AgentCard advertises URL '{url}' which does not match the "
            f"configured endpoint '{endpoint}'. Refusing to send credentials."
        )
