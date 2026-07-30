"""Tests for redirect-aware safe HTTP helpers."""

from __future__ import annotations

import socket
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools.security.safe_requests import safe_get


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


def _mock_session_get(monkeypatch: pytest.MonkeyPatch, get_response: Any) -> None:
    """Patch _build_pinned_session to return a session with a mocked get."""
    mock_session = MagicMock()
    mock_session.get = get_response
    mock_session.close = MagicMock()
    monkeypatch.setattr(
        "crewai_tools.security.safe_requests._build_pinned_session",
        lambda validated: mock_session,
    )


def test_safe_get_blocks_redirect_to_internal_url(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    requested_urls: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> requests.Response:
        requested_urls.append(url)
        assert kwargs["allow_redirects"] is False
        return _response(url, 302, location="http://127.0.0.1/admin")

    _mock_session_get(monkeypatch, fake_get)

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

    _mock_session_get(monkeypatch, fake_get)

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

    _mock_session_get(monkeypatch, fake_get)

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

    _mock_session_get(monkeypatch, fake_get)

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

    _mock_session_get(monkeypatch, fake_get)

    safe_get(
        "http://public.example/start",
        timeout=15,
        headers={"Authorization": "Bearer token"},
        cookies={"session": "abc"},
    )

    assert requests_made[1][1]["headers"] == {"Authorization": "Bearer token"}
    assert requests_made[1][1]["cookies"] == {"session": "abc"}


def test_safe_get_uses_pinned_ip_adapter(
    monkeypatch: pytest.MonkeyPatch, public_dns: None
) -> None:
    """Verify that safe_get creates a session with the PinnedIPAdapter."""
    from crewai_tools.security.safe_path import ValidatedURL

    captured_validated: list[ValidatedURL] = []

    def tracking_build(validated: ValidatedURL) -> requests.Session:
        captured_validated.append(validated)
        mock_session = MagicMock()
        mock_session.get = lambda url, **kw: _response(url, 200)
        mock_session.close = MagicMock()
        return mock_session

    monkeypatch.setattr(
        "crewai_tools.security.safe_requests._build_pinned_session",
        tracking_build,
    )

    safe_get("http://public.example/data", timeout=15)

    assert len(captured_validated) == 1
    assert captured_validated[0].resolved_ip == "93.184.216.34"
    assert captured_validated[0].url == "http://public.example/data"


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_pinned_adapter_rewrites_url(
    scheme: str, public_dns: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavioral test: PinnedIPAdapter.send() rewrites URL and sets Host header."""
    from crewai_tools.security.safe_path import PinnedIPAdapter

    adapter = PinnedIPAdapter(resolved_ip="93.184.216.34")

    captured_requests: list[Any] = []

    def fake_send(self: Any, request: Any, **kwargs: Any) -> MagicMock:
        captured_requests.append(request)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    monkeypatch.setattr(
        "requests.adapters.HTTPAdapter.send", fake_send,
    )

    req = requests.Request("GET", f"{scheme}://public.example/data")
    prepared = req.prepare()
    adapter.send(prepared)

    assert len(captured_requests) == 1
    sent = captured_requests[0]
    assert "93.184.216.34" in sent.url
    assert sent.headers["Host"] == "public.example"
