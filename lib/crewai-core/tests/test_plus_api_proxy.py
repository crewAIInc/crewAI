"""PlusAPI reaches AMP through the proxy the environment names.

Corporate networks and coding-agent sandboxes (Claude Code, Codex) only allow
outbound traffic through a proxy announced in HTTP(S)_PROXY. Two local servers
stand in for AMP and for that proxy, so the tests show which one a request reached.
"""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import NamedTuple, cast
import zipfile

from crewai_core.plus_api import PlusAPI
import pytest


PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


class Seen(NamedTuple):
    method: str
    target: str
    authorization: str | None
    body: bytes


class _RecordingServer(ThreadingHTTPServer):
    def __init__(self, name: str) -> None:
        super().__init__(("127.0.0.1", 0), _Recorder)
        self.name = name
        self.seen: list[Seen] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Recorder(BaseHTTPRequestHandler):
    def _answer(self) -> None:
        server = cast(_RecordingServer, self.server)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        server.seen.append(
            Seen(self.command, self.path, self.headers.get("Authorization"), body)
        )
        payload = json.dumps({"served_by": server.name}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._answer()

    def do_POST(self) -> None:
        self._answer()

    def log_message(self, format: str, *args: object) -> None:
        pass


def _serve(name: str) -> Iterator[_RecordingServer]:
    server = _RecordingServer(name)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def proxy() -> Iterator[_RecordingServer]:
    yield from _serve("proxy")


@pytest.fixture
def amp() -> Iterator[_RecordingServer]:
    yield from _serve("amp")


@pytest.fixture(autouse=True)
def no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture
def project_zip(tmp_path: Path) -> Path:
    path = tmp_path / "project.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pyproject.toml", "[project]\nname = 'demo'\n")
    return path


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_requests_go_through_the_proxy_named_in_the_environment(
    proxy: _RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTP_PROXY", proxy.url)

    response = PlusAPI("amp-token", base_url="http://amp.test").get_agent("researcher")

    assert response.json() == {"served_by": "proxy"}
    assert proxy.seen == [
        Seen(
            "GET",
            "http://amp.test/crewai_plus/api/v1/agents/researcher",
            "Bearer amp-token",
            b"",
        )
    ]


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_hosts_in_no_proxy_are_reached_directly(
    proxy: _RecordingServer, amp: _RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTP_PROXY", proxy.url)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")

    response = PlusAPI("amp-token", base_url=amp.url).get_agent("researcher")

    assert response.json() == {"served_by": "amp"}
    assert [seen.target for seen in amp.seen] == [
        "/crewai_plus/api/v1/agents/researcher"
    ]
    assert proxy.seen == []


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_without_proxy_variables_requests_go_directly(
    proxy: _RecordingServer, amp: _RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With no variables set, httpx falls back to the macOS or Windows system proxy
    # settings (urllib's getproxies); model a machine with none configured.
    monkeypatch.setattr("httpx._utils.getproxies", dict)

    response = PlusAPI("amp-token", base_url=amp.url).get_agent("researcher")

    assert response.json() == {"served_by": "amp"}
    assert amp.seen[0].authorization == "Bearer amp-token"
    assert proxy.seen == []


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_zip_deploy_uploads_go_through_the_proxy(
    proxy: _RecordingServer, monkeypatch: pytest.MonkeyPatch, project_zip: Path
) -> None:
    monkeypatch.setenv("HTTP_PROXY", proxy.url)

    response = PlusAPI("amp-token", base_url="http://amp.test").create_crew_from_zip(
        project_zip, name="demo"
    )

    assert response.json() == {"served_by": "proxy"}
    (seen,) = proxy.seen
    assert (seen.method, seen.target) == (
        "POST",
        "http://amp.test/crewai_plus/api/v1/crews/zip",
    )
    assert b'filename="project.zip"' in seen.body
    assert seen.authorization == "Bearer amp-token"
