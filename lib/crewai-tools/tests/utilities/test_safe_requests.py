"""Tests for SSRF-safe HTTP fetching (redirect + DNS-rebinding protection)."""

from __future__ import annotations

import http.server
import socketserver
import threading

import pytest
import requests

from crewai_tools.security import safe_requests
from crewai_tools.security.safe_requests import (
    SSRFProtectedAdapter,
    create_safe_session,
    safe_get,
)


INTERNAL_BODY = b"INTERNAL-ONLY-SECRET"


class _InternalHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(INTERNAL_BODY)

    def log_message(self, *args):  # silence
        pass


def _serve(handler):
    """Start a localhost server on an ephemeral port; return (server, port)."""
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


class TestRedirectRevalidation:
    """Layer 1: validate_url runs on every send, including each redirect hop.

    ``requests.Session.send`` calls ``adapter.send`` once per redirect hop, so
    re-validating in ``send`` is what blocks a 302 to an internal target.
    """

    def test_adapter_revalidates_before_any_network_call(self, monkeypatch):
        calls: list[str] = []

        def spy(url: str) -> str:
            calls.append(url)
            if "internal.target" in url:
                raise ValueError("URL resolves to private/reserved IP")
            return url

        monkeypatch.setattr(safe_requests, "validate_url", spy)

        adapter = SSRFProtectedAdapter()
        # Internal redirect target: send() must reject it before ever calling
        # the real transport (super().send is never reached).
        req = requests.Request("GET", "http://internal.target/").prepare()
        with pytest.raises(ValueError, match="private/reserved"):
            adapter.send(req)
        assert calls == ["http://internal.target/"]

    def test_session_mounts_protected_adapter(self):
        session = create_safe_session()
        assert isinstance(session.get_adapter("http://x"), SSRFProtectedAdapter)
        assert isinstance(session.get_adapter("https://x"), SSRFProtectedAdapter)


class _FakeSock:
    def __init__(self, peer):
        self._peer = peer

    def getpeername(self):
        return self._peer


class TestConnectionPeerGuard:
    """Layer 2: the connection rejects an internal peer IP at connect time.

    This is what closes the validate-then-connect DNS-rebinding gap — the IP
    the socket actually connected to is the IP that gets checked, so a host
    that resolved public at validation time but connects internal is blocked.
    """

    def test_safe_get_blocks_direct_internal(self):
        # No network: validate_url rejects 127.0.0.1 at the URL layer first.
        with pytest.raises(ValueError, match="private/reserved"):
            safe_get("http://127.0.0.1:9/", timeout=10)

    def test_assert_safe_peer_blocks_private(self):
        with pytest.raises(ValueError, match="private/reserved"):
            safe_requests._assert_safe_peer(_FakeSock(("127.0.0.1", 80)))

    def test_assert_safe_peer_blocks_metadata(self):
        with pytest.raises(ValueError, match="private/reserved"):
            safe_requests._assert_safe_peer(_FakeSock(("169.254.169.254", 80)))

    def test_assert_safe_peer_allows_public(self):
        # A public IP must not raise.
        safe_requests._assert_safe_peer(_FakeSock(("93.184.216.34", 80)))

    def test_assert_safe_peer_respects_escape_hatch(self, monkeypatch):
        monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")
        # No raise even for a private peer when the escape hatch is on.
        safe_requests._assert_safe_peer(_FakeSock(("127.0.0.1", 80)))

    def test_connection_validates_peer_after_connect(self, monkeypatch):
        """_SafeHTTPConnection.connect runs the peer guard after connecting."""
        conn = safe_requests._SafeHTTPConnection("example.com")

        def fake_super_connect(self):
            # Simulate a rebind: we connected to an internal address.
            self.sock = _FakeSock(("127.0.0.1", 80))

        monkeypatch.setattr(
            safe_requests.HTTPConnection, "connect", fake_super_connect
        )
        with pytest.raises(ValueError, match="private/reserved"):
            conn.connect()
