"""SSRF-safe HTTP fetching for crewai-tools.

:func:`validate_url` checks the URL it is handed, but it cannot protect a
fetch on its own: ``requests`` re-resolves DNS at connect time and follows
redirects automatically, so a public-looking host that 302-redirects to an
internal address (or that rebinds DNS between validation and connect) reaches
the internal target without ever being re-checked.

This module closes both gaps at the connection layer:

* :class:`SSRFProtectedAdapter` re-runs :func:`validate_url` for every request
  it sends. ``requests.Session.send`` invokes the adapter once per redirect
  hop, so each ``Location`` target is validated before it is followed.
* The adapter's connections validate the *actual* peer IP immediately after
  the socket connects. The IP that was authorised is therefore the IP the
  connection uses, removing the DNS time-of-check/time-of-use gap that
  :func:`validate_url`'s own ``getaddrinfo`` call leaves open.

Use :func:`safe_get` (or :func:`create_safe_session`) instead of calling
``requests.get`` directly from any tool that fetches a user- or
LLM-controlled URL.
"""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import DEFAULT_POOLBLOCK, HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.poolmanager import PoolManager

from crewai_tools.security.safe_path import (
    _is_escape_hatch_enabled,
    _is_private_or_reserved,
    validate_url,
)


def _assert_safe_peer(sock: Any) -> None:
    """Raise if a connected socket's peer is a private/reserved address.

    Validating the real peer (rather than a separately resolved IP) is what
    defeats DNS rebinding: the address we connected to is the address we check.
    """
    if _is_escape_hatch_enabled():
        return
    try:
        peer = sock.getpeername()
    except OSError:
        return
    ip_str = str(peer[0])
    if _is_private_or_reserved(ip_str):
        raise ValueError(
            f"Connection resolved to private/reserved IP {ip_str}. "
            f"Access to internal networks is not allowed (possible SSRF via "
            f"redirect or DNS rebinding)."
        )


class _SafeHTTPConnection(HTTPConnection):
    def connect(self) -> None:
        super().connect()
        _assert_safe_peer(self.sock)


class _SafeHTTPSConnection(HTTPSConnection):
    def connect(self) -> None:
        super().connect()
        _assert_safe_peer(self.sock)


class _SafeHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _SafeHTTPConnection


class _SafeHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _SafeHTTPSConnection


_SAFE_POOL_CLASSES = {
    "http": _SafeHTTPConnectionPool,
    "https": _SafeHTTPSConnectionPool,
}


class _SafePoolManager(PoolManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool_classes_by_scheme = _SAFE_POOL_CLASSES


class SSRFProtectedAdapter(HTTPAdapter):
    """Transport adapter that re-validates every hop and pins the peer IP.

    ``validate_url`` runs on each ``send`` — including every redirect hop
    ``requests`` follows — and the underlying connections reject any socket
    that ends up connected to a private/reserved address.
    """

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = DEFAULT_POOLBLOCK,
        **pool_kwargs: Any,
    ) -> None:
        self.poolmanager = _SafePoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )

    def send(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        # Re-validate the target of every request the session sends. Because
        # Session.send calls this once per redirect hop, each Location is
        # checked before it is followed.
        validate_url(request.url)
        return super().send(request, *args, **kwargs)


def create_safe_session() -> requests.Session:
    """Return a ``requests.Session`` that is hardened against SSRF.

    The session validates every request (and redirect hop) and pins
    connections to the validated peer IP.
    """
    session = requests.Session()
    # Ambient proxy settings bypass the protected pool classes via requests'
    # proxy manager path, so safe fetches must opt out of environment config.
    session.trust_env = False
    adapter = SSRFProtectedAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def safe_get(url: str, **kwargs: Any) -> requests.Response:
    """Perform an SSRF-safe ``GET``.

    Drop-in replacement for ``requests.get`` for tools that fetch a
    user- or LLM-controlled URL. Validates the initial URL and every redirect
    hop, and rejects connections that land on private/reserved addresses.

    Args:
        url: The URL to fetch.
        **kwargs: Forwarded to ``Session.get`` (``headers``, ``cookies``,
            ``timeout``, ...).

    Returns:
        The ``requests.Response``.

    Raises:
        ValueError: If the URL, a redirect target, or the connected peer is
            not allowed.
    """
    validate_url(url)
    with create_safe_session() as session:
        return session.get(url, **kwargs)
