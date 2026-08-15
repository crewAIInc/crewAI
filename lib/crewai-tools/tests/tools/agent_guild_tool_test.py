"""Focused tests for the Agent Guild tools."""

import json
from unittest.mock import patch

from crewai_tools.tools.agent_guild_tool.agent_guild_tool import (
    DEFAULT_AGENT_GUILD_BASE_URL,
    _base_url,
    _request,
)


def test_base_url_defaults_to_hosted_service(monkeypatch):
    monkeypatch.delenv("AGENT_GUILD_BASE_URL", raising=False)

    assert _base_url() == DEFAULT_AGENT_GUILD_BASE_URL


def test_base_url_accepts_http_and_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("AGENT_GUILD_BASE_URL", "http://localhost:8000/")

    assert _base_url() == "http://localhost:8000"


def test_request_rejects_non_http_base_url_before_transport(monkeypatch):
    monkeypatch.setenv("AGENT_GUILD_BASE_URL", "file:///etc/passwd")

    with patch("urllib.request.urlopen") as urlopen:
        result = json.loads(_request("/check?capability=code-review"))

    assert result == {
        "error": "agent_guild_invalid_base_url",
        "detail": "AGENT_GUILD_BASE_URL must be an absolute http(s) URL",
    }
    urlopen.assert_not_called()
