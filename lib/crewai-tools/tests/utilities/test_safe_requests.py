"""Tests for redirect-aware, connection-pinning safe HTTP helpers."""

from __future__ import annotations

import socket
from io import BytesIO
from typing import Any

import pytest
import requests

from crewai_tools.security import safe_requests
from crewai_tools.security.safe_requests import (
    SSRFProtectedAdapter,
    create_safe_session,
    safe_get,
)
from crewai_tools.security.ssrf_adapter import (
    _assert_safe_peer,
    create_validated_connection,
)


def _response(url: str, status_code: int, *, location: str | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = url
    response._content = b"ok"
    response.raw = BytesIO()
    if location is not None:
        response.headers["Location"] = location
    return response


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        if host in {"public.example", "safe.example"}:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("93.184.216.34", port),
                )
            ]
        return original_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def test_safe_get_blocks_direct_internal_url() -> None:
    with pytest.raises(ValueError, match="private/reserved IP"):
        safe_get("http://127.0.0.1/admin", timeout=15)


def _mock_get(monkeypatch: pytest.MonkeyPatch, get_response: Any) -> None:
    monkeypatch.setattr(
        "crewai_tools.security.safe_requests._raw_get",
        get_response,
    )


def test_safe_get_blocks_redirect_to_internal_url(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    requested_urls: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        requested_urls.append(url)
        assert kwargs["allow_redirects"] is False
        return _response(url, 302, location="http://127.0.0.1/admin")

    _mock_get(monkeypatch, fake_get)

    with pytest.raises(ValueError, match="private/reserved IP"):
        safe_get("http://public.example/start", timeout=15)

    assert requested_urls == ["http://public.example/start"]


def test_safe_get_follows_safe_relative_redirect(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    requested_urls: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        requested_urls.append(url)
        assert kwargs["allow_redirects"] is False
        if url == "http://public.example/start":
            return _response(url, 302, location="/final")
        return _response(url, 200)

    _mock_get(monkeypatch, fake_get)

    response = safe_get("http://public.example/start", timeout=15)

    assert response.status_code == 200
    assert response.url == "http://public.example/final"
    assert requested_urls == [
        "http://public.example/start",
        "http://public.example/final",
    ]
    assert len(response.history) == 1


def test_safe_get_fails_closed_after_too_many_redirects(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        return _response(url, 302, location="http://safe.example/again")

    _mock_get(monkeypatch, fake_get)

    with pytest.raises(ValueError, match="Too many redirects"):
        safe_get("http://public.example/start", max_redirects=1, timeout=15)


def _closable_response(
    url: str, status_code: int, *, location: str | None = None, closed: list[str]
) -> requests.Response:
    """Build a response that records its own URL when closed."""
    response = _response(url, status_code, location=location)
    response.close = lambda: closed.append(url)  # type: ignore[method-assign]
    return response


def test_safe_get_closes_earlier_hops_after_too_many_redirects(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    """Hops accumulated before the failure must not be left open.

    Under stream=True each hop holds its connection until its body is read or
    closed, and a caller handed an exception has no handle on them.
    """
    closed: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        return _closable_response(
            url, 302, location="http://safe.example/again", closed=closed
        )

    _mock_get(monkeypatch, fake_get)

    with pytest.raises(ValueError, match="Too many redirects"):
        safe_get("http://public.example/start", max_redirects=2, timeout=15, stream=True)

    assert len(closed) == 3


def test_safe_get_closes_earlier_hops_when_a_redirect_is_rejected(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    """A hop rejected mid-chain still releases the connections already open."""
    closed: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        if url == "http://public.example/start":
            return _closable_response(
                url, 302, location="http://safe.example/next", closed=closed
            )
        return _closable_response(
            url, 302, location="http://169.254.169.254/latest", closed=closed
        )

    _mock_get(monkeypatch, fake_get)

    with pytest.raises(ValueError, match="private/reserved IP"):
        safe_get("http://public.example/start", timeout=15, stream=True)

    assert closed == ["http://safe.example/next", "http://public.example/start"]


def test_safe_get_leaves_hops_open_on_success(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    """On success the hops belong to the caller, via response.history."""
    closed: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        if url == "http://public.example/start":
            return _closable_response(url, 302, location="/final", closed=closed)
        return _closable_response(url, 200, closed=closed)

    _mock_get(monkeypatch, fake_get)

    response = safe_get("http://public.example/start", timeout=15, stream=True)

    assert closed == []
    assert len(response.history) == 1


def test_safe_get_strips_credentials_on_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    requests_made: list[tuple[str, dict[str, Any]]] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        requests_made.append((url, kwargs))
        if url == "http://public.example/start":
            return _response(url, 302, location="http://safe.example/final")
        return _response(url, 200)

    _mock_get(monkeypatch, fake_get)

    response = safe_get(
        "http://public.example/start",
        timeout=15,
        headers={
            "Authorization": "Bearer token",
            "Authorization-Custom": "secret token",
            "Cookie": "session=abc",
            "X-API-Key": "api key",
            "X-CrewAI-Token": "crewai token",
            "User-Agent": "crewai-test",
        },
        cookies={"session": "abc"},
    )

    assert response.status_code == 200
    assert requests_made[0][1]["headers"] == {
        "Authorization": "Bearer token",
        "Authorization-Custom": "secret token",
        "Cookie": "session=abc",
        "X-API-Key": "api key",
        "X-CrewAI-Token": "crewai token",
        "User-Agent": "crewai-test",
    }
    assert requests_made[0][1]["cookies"] == {"session": "abc"}
    assert requests_made[1][1]["headers"] == {"User-Agent": "crewai-test"}
    assert "cookies" not in requests_made[1][1]


def test_safe_get_preserves_credentials_on_same_origin_redirect(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    requests_made: list[tuple[str, dict[str, Any]]] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        requests_made.append((url, kwargs))
        if url == "http://public.example/start":
            return _response(url, 302, location="/final")
        return _response(url, 200)

    _mock_get(monkeypatch, fake_get)

    safe_get(
        "http://public.example/start",
        timeout=15,
        headers={"Authorization": "Bearer token"},
        cookies={"session": "abc"},
    )

    assert requests_made[1][1]["headers"] == {"Authorization": "Bearer token"}
    assert requests_made[1][1]["cookies"] == {"session": "abc"}


def test_safe_get_rejects_proxies(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    _mock_get(monkeypatch, lambda url, **kwargs: _response(url, 200))

    with pytest.raises(ValueError, match="Proxies are not allowed"):
        safe_get(
            "http://public.example/start",
            timeout=15,
            proxies={"http": "http://127.0.0.1:8080"},
        )


def test_session_mounts_protected_adapter_and_ignores_env_proxies() -> None:
    session = create_safe_session()
    assert isinstance(session.get_adapter("http://x"), SSRFProtectedAdapter)
    assert isinstance(session.get_adapter("https://x"), SSRFProtectedAdapter)
    assert session.trust_env is False
    assert session.proxies == {}


def test_safe_session_does_not_follow_redirects_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = create_safe_session()
    urls: list[str] = []

    def fake_send(request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        urls.append(request.url or "")
        response = requests.Response()
        response.status_code = 302
        response.url = request.url
        response.request = request
        response.headers["Location"] = "http://127.0.0.1/admin"
        response._content = b""
        response.raw = BytesIO()
        return response

    adapter = session.get_adapter("http://example.com/")
    monkeypatch.setattr(adapter, "send", fake_send)

    response = session.get("http://example.com/start")

    assert response.status_code == 302
    assert urls == ["http://example.com/start"]


def test_safe_session_follows_redirects_when_caller_opts_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = create_safe_session()
    urls: list[str] = []

    def fake_send(request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        url = request.url or ""
        urls.append(url)
        response = requests.Response()
        response.url = url
        response.request = request
        response._content = b""
        response.raw = BytesIO()
        if url.endswith("/start"):
            response.status_code = 302
            response.headers["Location"] = "http://example.com/final"
        else:
            response.status_code = 200
        return response

    adapter = session.get_adapter("http://example.com/")
    monkeypatch.setattr(adapter, "send", fake_send)

    response = session.get("http://example.com/start", allow_redirects=True)

    assert response.status_code == 200
    assert urls == ["http://example.com/start", "http://example.com/final"]


def test_adapter_rejects_proxies() -> None:
    adapter = SSRFProtectedAdapter()
    req = requests.Request("GET", "http://example.com/").prepare()
    with pytest.raises(ValueError, match="Proxies are not allowed"):
        adapter.send(req, proxies={"http": "http://127.0.0.1:8080"})


class _FakeSock:
    def __init__(self, peer: tuple[str, int]) -> None:
        self._peer = peer

    def getpeername(self) -> tuple[str, int]:
        return self._peer


def test_assert_safe_peer_blocks_private() -> None:
    with pytest.raises(ValueError, match="private/reserved"):
        _assert_safe_peer(_FakeSock(("127.0.0.1", 80)))


def test_assert_safe_peer_blocks_metadata() -> None:
    with pytest.raises(ValueError, match="private/reserved"):
        _assert_safe_peer(_FakeSock(("169.254.169.254", 80)))


def test_assert_safe_peer_allows_public() -> None:
    _assert_safe_peer(_FakeSock(("93.184.216.34", 80)))


def test_assert_safe_peer_respects_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")
    _assert_safe_peer(_FakeSock(("127.0.0.1", 80)))


def test_assert_safe_peer_force_safe_overrides_escape_hatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")
    monkeypatch.setenv("CREWAI_TOOLS_FORCE_SAFE_PATHS", "true")
    with pytest.raises(ValueError, match="private/reserved"):
        _assert_safe_peer(_FakeSock(("127.0.0.1", 80)))


def test_create_validated_connection_pins_resolved_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups = {"n": 0}
    connected_to: list[tuple[str, int]] = []

    def fake_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        lookups["n"] += 1
        ip = "93.184.216.34" if lookups["n"] == 1 else "169.254.169.254"
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 80)),
        ]

    class RecordingSocket:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.peer: tuple[str, int] | None = None

        def setsockopt(self, *args: Any, **kwargs: Any) -> None:
            return None

        def settimeout(self, timeout: Any) -> None:
            return None

        def connect(self, sockaddr: tuple[str, int]) -> None:
            connected_to.append(sockaddr)
            self.peer = sockaddr

        def getpeername(self) -> tuple[str, int]:
            assert self.peer is not None
            return self.peer

        def close(self) -> None:
            return None

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: RecordingSocket())

    sock = create_validated_connection("rebind.example", 80)

    assert connected_to == [("93.184.216.34", 80)]
    assert sock.getpeername() == ("93.184.216.34", 80)


def test_create_validated_connection_blocks_when_any_record_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port or 80),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("169.254.169.254", port or 80),
            ),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ValueError, match="169.254.169.254"):
        create_validated_connection("dual.example", 80)


def test_create_validated_connection_blocks_direct_loopback() -> None:
    with pytest.raises(ValueError, match="private/reserved"):
        create_validated_connection("127.0.0.1", 9)


def test_create_validated_connection_keeps_socket_when_called_from_except(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    def fake_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80)),
        ]

    class RecordingSocket:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.peer: tuple[str, int] | None = None

        def setsockopt(self, *args: Any, **kwargs: Any) -> None:
            return None

        def settimeout(self, timeout: Any) -> None:
            return None

        def connect(self, sockaddr: tuple[str, int]) -> None:
            self.peer = sockaddr

        def getpeername(self) -> tuple[str, int]:
            assert self.peer is not None
            return self.peer

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: RecordingSocket())

    try:
        raise RuntimeError("caller is handling an error")
    except RuntimeError:
        sock = create_validated_connection("public.example", 80)

    assert closed == []
    assert sock.getpeername() == ("93.184.216.34", 80)


def test_create_validated_connection_closes_socket_when_peer_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    def fake_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80)),
        ]

    class RecordingSocket:
        def setsockopt(self, *args: Any, **kwargs: Any) -> None:
            return None

        def settimeout(self, timeout: Any) -> None:
            return None

        def connect(self, sockaddr: tuple[str, int]) -> None:
            return None

        def getpeername(self) -> tuple[str, int]:
            return ("127.0.0.1", 80)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: RecordingSocket())

    with pytest.raises(ValueError, match="private/reserved"):
        create_validated_connection("rebind.example", 80)

    assert closed == [True]


class _TrackingSession:
    """Session stand-in that records whether it was closed too early."""

    def __init__(self) -> None:
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = b"hello" if not kwargs.get("stream") else False
        response.raw = BytesIO()

        def iter_content(
            chunk_size: int = 1, decode_unicode: bool = False
        ) -> Any:
            if self.closed:
                raise RuntimeError("session already closed")
            yield b"hello"

        response.iter_content = iter_content  # type: ignore[method-assign]
        return response

    def close(self) -> None:
        self.closed = True


def test_streamed_raw_get_keeps_session_open_until_response_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _TrackingSession()
    monkeypatch.setattr(safe_requests, "create_safe_session", lambda: session)

    response = safe_requests._raw_get("http://example.com/file", stream=True)

    assert session.closed is False
    assert b"".join(response.iter_content()) == b"hello"
    response.close()
    assert session.closed is True


def test_non_streamed_raw_get_closes_session_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _TrackingSession()
    monkeypatch.setattr(safe_requests, "create_safe_session", lambda: session)

    response = safe_requests._raw_get("http://example.com/file")

    assert session.closed is True
    assert response.content == b"hello"
