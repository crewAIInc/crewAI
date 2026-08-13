"""SSRF-safe HTTP fetching for crewai-tools.

:func:`~crewai_tools.security.safe_path.validate_url` checks the URL it is
handed, but it cannot protect a fetch on its own: ``requests`` re-resolves
DNS at connect time and follows redirects automatically, so a public-looking
host that 302-redirects to an internal address (or that rebinds DNS between
validation and connect) reaches the internal target without ever being
re-checked.

This module closes both gaps at the connection layer:

* :class:`SSRFProtectedAdapter` re-runs :func:`validate_url` for every
  request it sends. Combined with :func:`safe_get`'s explicit hop loop,
  each ``Location`` target is validated before it is followed.
* Connections resolve DNS once, refuse any private/reserved address, and
  ``connect()`` to the authorised sockaddr. The IP that was checked is the
  IP the socket uses, so DNS rebinding between validation and connect
  cannot retarget the request.
* After connect, the real peer from ``getpeername()`` is checked again.
* Environment proxies are disabled: a proxy would be the connected peer,
  so the destination IP would never be inspected.

Use :func:`safe_get` (or :func:`create_safe_session`) instead of calling
``requests.get`` directly from any tool that fetches a user- or
LLM-controlled URL.
"""

from __future__ import annotations

import socket
import sys
from typing import Any
from urllib.parse import urljoin, urlparse

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
    _is_escape_hatch_enabled,
    is_blocked_ip,
    validate_url,
)


_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_STREAM_CHUNK_SIZE = 65536
_SENSITIVE_HEADER_NAMES = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
}
_SENSITIVE_HEADER_FRAGMENTS = ("api-key", "apikey", "secret", "token")
_UNSAFE_PATHS_HINT = "Set CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true to bypass this check."


def _same_origin(previous_url: str, next_url: str) -> bool:
    previous = urlparse(previous_url)
    next_ = urlparse(next_url)
    return (previous.scheme, previous.netloc) == (next_.scheme, next_.netloc)


def _is_sensitive_header(header: str) -> bool:
    normalized = header.lower()
    return (
        normalized in _SENSITIVE_HEADER_NAMES
        or normalized.startswith("authorization-")
        or any(fragment in normalized for fragment in _SENSITIVE_HEADER_FRAGMENTS)
    )


def _strip_cross_origin_credentials(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    sanitized = {**request_kwargs}
    headers = sanitized.get("headers")
    if headers:
        sanitized["headers"] = {
            key: value
            for key, value in headers.items()
            if not _is_sensitive_header(str(key))
        }
    sanitized.pop("cookies", None)
    return sanitized


def _set_socket_options(sock: socket.socket, options: Any) -> None:
    if not options:
        return
    for opt in options:
        sock.setsockopt(*opt)


def _connect_timeout(timeout: Any) -> float | None:
    if timeout is None:
        return socket.getdefaulttimeout()
    connect = getattr(timeout, "connect_timeout", timeout)
    default_token = getattr(timeout, "DEFAULT_TIMEOUT", None)
    if connect is None or connect is default_token:
        return socket.getdefaulttimeout()
    if isinstance(connect, (int, float)):
        return float(connect)
    return socket.getdefaulttimeout()


def _blocked_ip_error(ip_str: str) -> ValueError:
    return ValueError(
        f"Connection resolved to private/reserved IP {ip_str}. "
        f"Access to internal networks is not allowed (possible SSRF via "
        f"redirect or DNS rebinding). {_UNSAFE_PATHS_HINT}"
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
            f"request to prevent SSRF. {_UNSAFE_PATHS_HINT}"
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
    """Open a TCP socket to *host* after validating and pinning the peer IP.

    DNS is resolved once. If any address is private or reserved the
    connection is refused. ``connect()`` is then called on a sockaddr from
    that lookup, so a later DNS rebind cannot change the destination.
    """
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

    if not addrinfos:
        raise OSError("getaddrinfo returns an empty list")

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
            try:
                _assert_safe_peer(sock)
            finally:
                if sys.exc_info()[0] is not None:
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
    """Transport adapter that re-validates every hop and pins the peer IP.

    ``validate_url`` runs on each ``send`` — including every redirect hop
    ``requests`` follows — and the underlying connections refuse any socket
    that would land on a private/reserved address.
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

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        if not _is_escape_hatch_enabled():
            raise ValueError(
                f"Proxies are not allowed for SSRF-safe requests. {_UNSAFE_PATHS_HINT}"
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
        validate_url(request.url or "")
        if proxies and not _is_escape_hatch_enabled():
            raise ValueError(
                f"Proxies are not allowed for SSRF-safe requests. {_UNSAFE_PATHS_HINT}"
            )
        return super().send(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies={} if not _is_escape_hatch_enabled() else proxies,
        )


def create_safe_session() -> requests.Session:
    """Return a ``requests.Session`` that is hardened against SSRF.

    The session validates every request (and redirect hop), pins
    connections to a validated peer IP, and ignores environment proxies.
    """
    session = requests.Session()
    session.trust_env = False
    session.proxies = {}
    adapter = SSRFProtectedAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _reject_proxies(kwargs: dict[str, Any]) -> None:
    proxies = kwargs.pop("proxies", None)
    if proxies and not _is_escape_hatch_enabled():
        raise ValueError(f"Proxies are not allowed for safe_get. {_UNSAFE_PATHS_HINT}")
    kwargs["proxies"] = {}


def _raw_get(url: str, **kwargs: Any) -> requests.Response:
    """GET through an SSRF-protected session. Patchable in tests."""
    with create_safe_session() as session:
        return session.get(url, **kwargs)


def safe_get(url: str, *, max_redirects: int = 10, **kwargs: Any) -> requests.Response:
    """GET a URL while validating each redirect target before following it.

    On success the hops are attached to the returned response's ``history`` and
    are the caller's to close. On failure they are closed here: a caller given
    an exception has no handle on them, and a streamed hop holds its connection
    until its body is read or closed.

    Each hop is fetched through :class:`SSRFProtectedAdapter`, which re-runs
    :func:`validate_url` and pins the TCP connection to an authorised IP.
    """
    current_url = validate_url(url)
    _reject_proxies(kwargs)
    request_kwargs = {**kwargs, "allow_redirects": False}
    timeout = request_kwargs.pop("timeout", 30)
    history: list[requests.Response] = []
    redirects_followed = 0

    try:
        while True:
            response = _raw_get(current_url, timeout=timeout, **request_kwargs)
            if (
                response.status_code not in _REDIRECT_STATUS_CODES
                or "Location" not in response.headers
            ):
                response.history = history
                return response

            if redirects_followed >= max_redirects:
                response.close()
                raise ValueError(f"Too many redirects while fetching URL: {url}")

            location = response.headers.get("Location")
            if not location:
                response.history = history
                return response

            try:
                redirect_url = validate_url(urljoin(response.url, location))
            except ValueError:
                response.close()
                raise

            if not _same_origin(current_url, redirect_url):
                request_kwargs = _strip_cross_origin_credentials(request_kwargs)

            history.append(response)
            current_url = redirect_url
            redirects_followed += 1
    except BaseException:
        for hop in history:
            hop.close()
        raise


def safe_get_bounded(
    url: str,
    *,
    max_bytes: int,
    timeout: float | tuple[float, float] = 30,
    headers: dict[str, str] | None = None,
    max_redirects: int = 10,
) -> tuple[bytes, str, str]:
    """GET a URL through :func:`safe_get`, refusing bodies over *max_bytes*.

    The body is streamed and abandoned as soon as it crosses the limit, so an
    oversized response costs one chunk of memory instead of all of it. The cap
    counts decoded bytes, which is what a compressed response expands into --
    ``Content-Length`` describes the wire size and cannot bound that.

    Args:
        url: The URL to fetch.
        max_bytes: Largest body to accept, in decoded bytes.
        timeout: Request timeout, passed through to requests.
        headers: Request headers.
        max_redirects: Hops to follow before giving up.

    Returns:
        A ``(body, content_type, final_url)`` tuple, where *final_url* is the
        last validated URL in the redirect chain.

    Raises:
        ValueError: If *max_bytes* is not positive, URL validation fails, the
            redirect chain is too long, or the body exceeds *max_bytes*.
        requests.RequestException: If the request fails or returns an error
            status.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}.")

    response = safe_get(
        url,
        max_redirects=max_redirects,
        headers=headers,
        timeout=timeout,
        stream=True,
    )
    try:
        response.raise_for_status()

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=_STREAM_CHUNK_SIZE):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                # Names the URL that served the body, which after a redirect is
                # not the one that was requested.
                raise ValueError(
                    f"Response body from '{response.url}' exceeds the "
                    f"{max_bytes} byte limit."
                )
            chunks.append(chunk)

        return (
            b"".join(chunks),
            response.headers.get("Content-Type", ""),
            response.url,
        )
    finally:
        # Under stream=True each hop holds its connection until the body is read,
        # so the redirects need closing too, not just the response we return.
        for hop in response.history:
            hop.close()
        response.close()
