"""urllib3/requests transport that pins TCP to a validated peer IP."""

from __future__ import annotations

import socket
import sys
from typing import Any

import requests
from requests.adapters import DEFAULT_POOLBLOCK, HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import (
    ConnectTimeoutError,
    LocationParseError,
    NameResolutionError,
    NewConnectionError,
)
from urllib3.poolmanager import PoolManager
from urllib3.util.connection import (
    allowed_gai_family,
    create_connection as urllib3_create_connection,
)

from crewai_tools.security.safe_path import (
    _BYPASS_HINT,
    _is_escape_hatch_enabled,
    is_blocked_ip,
)


def _set_socket_options(sock: socket.socket, options: Any) -> None:
    for opt in options or ():
        sock.setsockopt(*opt)


def _connect_timeout(timeout: Any) -> float | None:
    connect = getattr(timeout, "connect_timeout", timeout)
    if connect is None or connect is getattr(timeout, "DEFAULT_TIMEOUT", None):
        return socket.getdefaulttimeout()
    if isinstance(connect, (int, float)):
        return float(connect)
    return socket.getdefaulttimeout()


def _blocked_ip_error(ip_str: str) -> ValueError:
    return ValueError(
        f"Connection resolved to private/reserved IP {ip_str}. "
        f"Access to internal networks is not allowed (possible SSRF via "
        f"redirect or DNS rebinding). {_BYPASS_HINT}"
    )


def _assert_safe_peer(sock: socket.socket) -> None:
    """Raise if a connected socket's peer is a private/reserved address."""
    if _is_escape_hatch_enabled():
        return
    try:
        peer = sock.getpeername()
    except OSError as exc:
        raise ValueError(
            "Unable to determine the connected peer address; blocking "
            f"request to prevent SSRF. {_BYPASS_HINT}"
        ) from exc
    ip_str = str(peer[0])
    if is_blocked_ip(ip_str):
        raise _blocked_ip_error(ip_str)


def create_validated_connection(
    host: str,
    port: int,
    *,
    timeout: Any = None,
    source_address: tuple[str, int] | None = None,
    socket_options: Any = None,
) -> socket.socket:
    """Open a TCP socket to *host* after validating and pinning the peer IP."""
    if _is_escape_hatch_enabled():
        return urllib3_create_connection(
            (host, port),
            timeout=_connect_timeout(timeout),
            source_address=source_address,
            socket_options=socket_options,
        )

    if host.startswith("["):
        host = host.strip("[]")

    try:
        host.encode("idna")
    except UnicodeError:
        raise LocationParseError(f"'{host}', label empty or too long") from None

    try:
        addrinfos = socket.getaddrinfo(
            host, port, allowed_gai_family(), socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: '{host}'") from exc

    for _family, _socktype, _proto, _canonname, sockaddr in addrinfos:
        ip_str = str(sockaddr[0])
        if is_blocked_ip(ip_str):
            raise _blocked_ip_error(ip_str)

    timeout = _connect_timeout(timeout)
    err: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in addrinfos:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(family, socktype, proto)
            _set_socket_options(sock, socket_options)
            sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            peer_validated = False
            try:
                _assert_safe_peer(sock)
                peer_validated = True
            finally:
                if not peer_validated:
                    sock.close()
            return sock
        except OSError as exc:
            err = exc
            if sock is not None:
                sock.close()

    if err is not None:
        raise err
    raise OSError("getaddrinfo returns an empty list")


def _open_validated_socket(conn: HTTPConnection) -> socket.socket:
    port = conn.port
    if port is None:
        port = 443 if isinstance(conn, HTTPSConnection) else 80
    try:
        sock = create_validated_connection(
            conn._dns_host,
            port,
            timeout=conn.timeout,
            source_address=conn.source_address,
            socket_options=conn.socket_options,
        )
    except socket.gaierror as exc:
        raise NameResolutionError(conn.host, conn, exc) from exc
    except socket.timeout as exc:
        raise ConnectTimeoutError(
            conn,
            f"Connection to {conn.host} timed out. (connect timeout={conn.timeout})",
        ) from exc
    except OSError as exc:
        raise NewConnectionError(
            conn, f"Failed to establish a new connection: {exc}"
        ) from exc

    sys.audit("http.client.connect", conn, conn.host, conn.port)
    return sock


class _SafeHTTPConnection(HTTPConnection):
    def _new_conn(self) -> socket.socket:
        return _open_validated_socket(self)


class _SafeHTTPSConnection(HTTPSConnection):
    def _new_conn(self) -> socket.socket:
        return _open_validated_socket(self)


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
    """HTTPAdapter that connects through :func:`create_validated_connection`."""

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = DEFAULT_POOLBLOCK,
        **pool_kwargs: Any,
    ) -> None:
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block
        self.poolmanager = _SafePoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        if not _is_escape_hatch_enabled():
            raise ValueError(
                f"Proxies are not allowed for SSRF-safe requests. {_BYPASS_HINT}"
            )
        return super().proxy_manager_for(proxy, **proxy_kwargs)  # type: ignore[no-untyped-call]

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: Any = None,
        verify: bool | str = True,
        cert: Any = None,
        proxies: Any = None,
    ) -> requests.Response:
        unsafe = _is_escape_hatch_enabled()
        if proxies and not unsafe:
            raise ValueError(
                f"Proxies are not allowed for SSRF-safe requests. {_BYPASS_HINT}"
            )
        return super().send(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies if unsafe else {},
        )
