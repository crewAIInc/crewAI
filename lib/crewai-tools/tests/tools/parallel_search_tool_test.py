import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch
from urllib.parse import urlparse

import pytest
import requests

from crewai_tools.tools.parallel_tools.parallel_search_tool import (
    ParallelSearchTool,
)


def test_requires_env_var(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    tool = ParallelSearchTool()
    result = tool.run(objective="test")
    assert "PARALLEL_API_KEY" in result


@patch("crewai_tools.tools.parallel_tools.parallel_search_tool.requests.post")
def test_happy_path(mock_post, monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "test")

    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "search_id": "search_123",
        "results": [
            {
                "url": "https://www.un.org/en/about-us/history-of-the-un",
                "title": "History of the United Nations",
                "excerpts": [
                    "Four months after the San Francisco Conference ended, the United Nations officially began, on 24 October 1945..."
                ],
            }
        ],
    }

    tool = ParallelSearchTool()
    result = tool.run(
        objective="When was the UN established?", search_queries=["Founding year UN"]
    )
    data = json.loads(result)
    assert "search_id" in data
    urls = [r.get("url", "") for r in data.get("results", [])]
    # Validate host against allowed set instead of substring matching
    allowed_hosts = {"www.un.org", "un.org"}
    assert any(urlparse(u).netloc in allowed_hosts for u in urls)


@pytest.mark.parametrize("version", ["1.2.3", "unknown"])
@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1"])
def test_user_agent_reaches_search_requests(monkeypatch, version):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(
                (
                    self.path,
                    dict(self.headers),
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            body = b'{"search_id":"local","results":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("PARALLEL_API_KEY", "local-test-key")
    monkeypatch.setattr(
        "crewai_tools.tools.parallel_tools.parallel_search_tool.get_crewai_version",
        lambda: version,
    )
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1beta/search"
        tool = ParallelSearchTool(search_url=url)
        for objective in ["first search", "subsequent search"]:
            assert json.loads(tool.run(objective=objective))["search_id"] == "local"

        # Attribution must not change Requests defaults or other callers' headers.
        requests.post(url, json={}, timeout=5)
        requests.post(
            url,
            json={},
            headers={"User-Agent": "caller-client/2", "Authorization": "Bearer local"},
            timeout=5,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    project_agent = f"crewai/{version}" if version != "unknown" else "crewai"
    for (path, headers, payload), objective in zip(
        captured[:2], ["first search", "subsequent search"]
    ):
        assert path == "/v1beta/search"
        assert headers["User-Agent"] == (
            f"{project_agent} {requests.utils.default_user_agent()}"
        )
        assert headers["x-api-key"] == "local-test-key"
        assert headers["Content-Type"] == "application/json"
        assert payload["objective"] == objective
    assert captured[2][1]["User-Agent"] == requests.utils.default_user_agent()
    assert captured[3][1]["User-Agent"] == "caller-client/2"
    assert captured[3][1]["Authorization"] == "Bearer local"
