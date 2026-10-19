"""PlusAPI reaches AMP through the proxy the machine names.

Corporate networks and coding-agent sandboxes (Claude Code, Codex) only allow
outbound traffic through a proxy announced in HTTP(S)_PROXY. Two local servers
stand in for AMP and for that proxy, so the tests show which one a request reached.

The machine's settings are the variables the CLI pinned at startup and the OS
proxy settings. A variable that appears later, as a project's .env loaded by
`crewai run` or by `crewai.llm` on import puts it there, does not count.
"""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import NamedTuple, cast
import urllib.request
import zipfile

from crewai_core import plus_api
from crewai_core.plus_api import PlusAPI
from dotenv import load_dotenv
import pytest


PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
TRANSPORT_VARS = (
    *PROXY_VARS,
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUEST_METHOD",
    "UV_ENV_FILE",
)


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
def machine(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The variables pinned at startup: none, on a machine with no OS proxy.

    The environment starts without transport variables too, and anything a test
    loads into it is removed afterwards.
    """
    pinned: dict[str, str] = {}
    monkeypatch.setattr(plus_api, "_MACHINE_TRANSPORT_ENV", pinned)
    monkeypatch.setattr(plus_api, "_os_proxies", dict)
    for name in TRANSPORT_VARS:
        for variant in {name, name.lower()}:
            monkeypatch.setenv(variant, "")  # recorded, so teardown removes it
            monkeypatch.delenv(variant)
    return pinned


@pytest.fixture
def project_zip(tmp_path: Path) -> Path:
    path = tmp_path / "project.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pyproject.toml", "[project]\nname = 'demo'\n")
    return path


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_requests_go_through_the_proxy_the_machine_names(
    proxy: _RecordingServer, machine: dict[str, str]
) -> None:
    machine["HTTP_PROXY"] = proxy.url

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
def test_in_a_cgi_process_an_uppercase_http_proxy_is_ignored(
    proxy: _RecordingServer, amp: _RecordingServer, machine: dict[str, str]
) -> None:
    """A CGI client's Proxy header arrives as HTTP_PROXY ("httpoxy"); the login must not follow it."""
    machine["HTTP_PROXY"] = proxy.url
    machine["REQUEST_METHOD"] = "GET"

    response = PlusAPI("amp-token", base_url=amp.url).get_agent("researcher")

    assert response.json() == {"served_by": "amp"}
    assert proxy.seen == []


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_in_a_cgi_process_a_lowercase_http_proxy_still_applies(
    proxy: _RecordingServer, machine: dict[str, str]
) -> None:
    machine["http_proxy"] = proxy.url
    machine["REQUEST_METHOD"] = "GET"

    response = PlusAPI("amp-token", base_url="http://amp.test").get_agent("researcher")

    assert response.json() == {"served_by": "proxy"}
    assert proxy.seen[0].authorization == "Bearer amp-token"


@pytest.mark.parametrize(
    "variables",
    [
        {"HTTP_PROXY": "http://upper.test:1"},
        {"HTTP_PROXY": "http://upper.test:1", "REQUEST_METHOD": "GET"},
        {
            "HTTP_PROXY": "http://upper.test:1",
            "http_proxy": "http://lower.test:1",
            "REQUEST_METHOD": "POST",
        },
        {"HTTPS_PROXY": "http://secure.test:1", "REQUEST_METHOD": "GET"},
        {"HTTP_PROXY": "http://upper.test:1", "http_proxy": ""},
    ],
)
def test_proxies_are_read_as_urllib_reads_them(
    variables: dict[str, str], machine: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in variables.items():
        machine[name] = value
        monkeypatch.setenv(name, value)

    assert plus_api._machine_proxies() == urllib.request.getproxies_environment()


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_hosts_in_no_proxy_are_reached_directly(
    proxy: _RecordingServer, amp: _RecordingServer, machine: dict[str, str]
) -> None:
    machine["HTTP_PROXY"] = proxy.url
    machine["NO_PROXY"] = "127.0.0.1"

    response = PlusAPI("amp-token", base_url=amp.url).get_agent("researcher")

    assert response.json() == {"served_by": "amp"}
    assert [seen.target for seen in amp.seen] == [
        "/crewai_plus/api/v1/agents/researcher"
    ]
    assert proxy.seen == []


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_without_proxy_settings_requests_go_directly(
    proxy: _RecordingServer, amp: _RecordingServer
) -> None:
    response = PlusAPI("amp-token", base_url=amp.url).get_agent("researcher")

    assert response.json() == {"served_by": "amp"}
    assert amp.seen[0].authorization == "Bearer amp-token"
    assert proxy.seen == []


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_with_no_proxy_variable_the_os_proxy_settings_apply(
    proxy: _RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plus_api, "_os_proxies", lambda: {"http": proxy.url})

    response = PlusAPI("amp-token", base_url="http://amp.test").get_agent("researcher")

    assert response.json() == {"served_by": "proxy"}
    assert proxy.seen[0].authorization == "Bearer amp-token"


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_a_proxy_a_project_env_sets_after_startup_is_not_used(
    proxy: _RecordingServer, amp: _RecordingServer, tmp_path: Path
) -> None:
    """`crewai run` on a JSON crew or a declarative flow, and `crewai.llm` on
    import, load the project's .env into the process. A cloned project must not
    route the saved login through its own proxy or have AMP vouched for by its
    own CA: a CA file that does not exist would fail the request if it were read."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"HTTP_PROXY={proxy.url}\n"
        f"https_proxy={proxy.url}\n"
        f"SSL_CERT_FILE={tmp_path / 'project-ca.pem'}\n"
    )
    load_dotenv(env_file, override=True)

    response = PlusAPI("amp-token", base_url=amp.url).get_agent("researcher")

    assert response.json() == {"served_by": "amp"}
    assert amp.seen[0].authorization == "Bearer amp-token"
    assert proxy.seen == []


def test_the_ca_bundle_the_machine_names_is_the_one_trusted(
    machine: dict[str, str], tmp_path: Path
) -> None:
    machine["SSL_CERT_FILE"] = str(tmp_path / "missing-ca.pem")

    with pytest.raises(FileNotFoundError):
        PlusAPI("amp-token", base_url="https://amp.test").get_agent("researcher")


@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
def test_zip_deploy_uploads_go_through_the_proxy(
    proxy: _RecordingServer, machine: dict[str, str], project_zip: Path
) -> None:
    machine["HTTP_PROXY"] = proxy.url

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


def test_restoring_the_machine_transport_undoes_what_a_project_env_set(
    machine: dict[str, str],
) -> None:
    machine["HTTPS_PROXY"] = "http://machine-proxy.test:3128"
    environ = {
        "HTTPS_PROXY": "http://project-proxy.test:8080",
        "all_proxy": "socks5://project-proxy.test:1080",
        "NO_PROXY": "*",
        "SSL_CERT_FILE": "/project/ca.pem",
        "REQUEST_METHOD": "GET",
        "UV_ENV_FILE": ".env",
        "OPENAI_API_KEY": "project-key",
    }

    plus_api.restore_machine_transport(environ)

    assert environ == {
        "HTTPS_PROXY": "http://machine-proxy.test:3128",
        "OPENAI_API_KEY": "project-key",
    }
