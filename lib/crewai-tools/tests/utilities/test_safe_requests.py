"""Tests for redirect-aware safe HTTP helpers."""

from __future__ import annotations

import socket
from io import BytesIO
from typing import Any

import pytest
import requests

from crewai_tools.security.safe_requests import safe_download, safe_get


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
        "crewai_tools.security.safe_requests.requests.get",
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


def test_safe_get_pins_dns_against_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostname that resolves to a safe IP at validation time but would
    resolve to a private IP at connection time must not let the private IP
    actually get connected to -- the connection is pinned to the IP that was
    validated, not left to re-resolve.

    Without DNS pinning, this test fails: validate_and_resolve's lookup
    consumes the "first" (safe) answer, and the mocked `requests.get` below
    -- standing in for what the real HTTP client does when it opens the
    connection -- would independently re-resolve to the "second" (private)
    answer.
    """
    lookups: list[str] = []

    def rebinding_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        if host != "rebind.example":
            raise socket.gaierror(f"unexpected host in test: {host}")
        lookups.append(host)
        # First lookup (validation) returns a safe IP; any further, unpinned
        # lookup returns a private one -- simulating DNS rebinding via a
        # short-TTL record that changes between validation and connection.
        ip = "93.184.216.34" if len(lookups) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)

    connected_ips: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        # Stand-in for what the real HTTP client does at connection time:
        # resolve the hostname again. If DNS pinning is working,
        # socket.getaddrinfo has been overridden for the duration of this
        # call to return only the validated IP, regardless of how many times
        # "rebind.example" is looked up.
        resolved = socket.getaddrinfo("rebind.example", 80)
        connected_ips.append(resolved[0][4][0])
        return _response(url, 200)

    _mock_get(monkeypatch, fake_get)

    response = safe_get("http://rebind.example/", timeout=15)

    assert response.status_code == 200
    assert connected_ips == ["93.184.216.34"]
    assert "127.0.0.1" not in connected_ips


def test_safe_get_restores_real_resolver_after_pinning(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    """DNS pinning must not leak past the request it was applied to."""

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        return _response(url, 200)

    _mock_get(monkeypatch, fake_get)

    original_getaddrinfo = socket.getaddrinfo
    safe_get("http://public.example/", timeout=15)

    assert socket.getaddrinfo is original_getaddrinfo


def test_safe_download_writes_content_to_disk(
    monkeypatch: pytest.MonkeyPatch, public_dns: None, tmp_path: Any
) -> None:
    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        assert kwargs.get("stream") is True
        response = _response(url, 200)
        response._content = b"pdf-bytes-here"
        response.raw = BytesIO(b"pdf-bytes-here")
        response.iter_content = lambda chunk_size=1: [b"pdf-bytes-here"]
        return response

    _mock_get(monkeypatch, fake_get)

    dest = tmp_path / "paper.pdf"
    safe_download("http://public.example/paper.pdf", dest, timeout=15)

    assert dest.read_bytes() == b"pdf-bytes-here"


def test_safe_download_blocks_private_ip() -> None:
    with pytest.raises(ValueError, match="private/reserved IP"):
        safe_download("http://127.0.0.1/malicious.pdf", "/tmp/out.pdf", timeout=15)


def test_safe_download_blocks_redirect_to_internal_url(
    monkeypatch: pytest.MonkeyPatch, public_dns: None, tmp_path: Any
) -> None:
    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        return _response(url, 302, location="http://127.0.0.1/admin")

    _mock_get(monkeypatch, fake_get)

    with pytest.raises(ValueError, match="private/reserved IP"):
        safe_download("http://public.example/start", tmp_path / "out.pdf", timeout=15)
