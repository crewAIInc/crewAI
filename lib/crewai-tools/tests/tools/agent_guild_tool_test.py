"""Focused tests for the Agent Guild tools."""

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.agent_guild_tool.agent_guild_tool import (
    DEFAULT_AGENT_GUILD_BASE_URL,
    AgentGuildCheckTool,
    AgentGuildRiskScoreTool,
    AgentGuildVerifyPassportTool,
    _base_url,
    _request,
)


@pytest.fixture(autouse=True)
def clear_agent_guild_environment(monkeypatch):
    monkeypatch.delenv("AGENT_GUILD_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_GUILD_BASE_URL", raising=False)


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


def _response(body: bytes) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = body
    return response


def test_check_tool_encodes_capability_and_returns_api_json(monkeypatch):
    monkeypatch.delenv("AGENT_GUILD_BASE_URL", raising=False)
    response = _response(b'{"verdict":"hire"}')

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        result = AgentGuildCheckTool()._run("code review/analysis")

    request = urlopen.call_args.args[0]
    assert request.method == "GET"
    assert request.full_url == (
        f"{DEFAULT_AGENT_GUILD_BASE_URL}/check?"
        "capability=code%20review%2Fanalysis"
    )
    assert request.data is None
    assert json.loads(result) == {"verdict": "hire"}


def test_risk_score_tool_encodes_agent_id_and_returns_api_json(monkeypatch):
    monkeypatch.delenv("AGENT_GUILD_BASE_URL", raising=False)
    response = _response(b'{"verdict":"caution"}')

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        result = AgentGuildRiskScoreTool()._run("agent/id 1")

    request = urlopen.call_args.args[0]
    assert request.method == "GET"
    assert request.full_url == (
        f"{DEFAULT_AGENT_GUILD_BASE_URL}/agents/agent%2Fid%201/risk-score"
    )
    assert request.data is None
    assert json.loads(result) == {"verdict": "caution"}


def test_verify_passport_tool_posts_credential_and_returns_api_json(monkeypatch):
    monkeypatch.delenv("AGENT_GUILD_BASE_URL", raising=False)
    credential = '{"type":["VerifiableCredential"]}'
    response = _response(b'{"valid":true}')

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        result = AgentGuildVerifyPassportTool()._run(credential)

    request = urlopen.call_args.args[0]
    assert request.method == "POST"
    assert request.full_url == (
        f"{DEFAULT_AGENT_GUILD_BASE_URL}/credentials/verify"
    )
    assert request.data == credential.encode("utf-8")
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(result) == {"valid": True}


def test_request_sends_configured_api_key(monkeypatch):
    monkeypatch.setenv("AGENT_GUILD_API_KEY", "ak_test")
    response = _response(b'{"verdict":"hire"}')

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        _request("/check?capability=code-review")

    request = urlopen.call_args.args[0]
    assert request.get_header("X-api-key") == "ak_test"


def test_http_error_returns_api_body_unchanged():
    error = urllib.error.HTTPError(
        f"{DEFAULT_AGENT_GUILD_BASE_URL}/check",
        402,
        "Payment Required",
        {},
        io.BytesIO(b'{"detail":"payment required"}'),
    )

    with patch("urllib.request.urlopen", side_effect=error):
        result = _request("/check?capability=code-review")

    assert result == '{"detail":"payment required"}'


def test_http_error_without_body_includes_status():
    error = urllib.error.HTTPError(
        f"{DEFAULT_AGENT_GUILD_BASE_URL}/check",
        503,
        "Service Unavailable",
        {},
        io.BytesIO(b""),
    )

    with patch("urllib.request.urlopen", side_effect=error):
        result = json.loads(_request("/check?capability=code-review"))

    assert result["error"] == "agent_guild_http_error"
    assert result["status"] == 503


def test_transport_error_redacts_configured_url_secrets(monkeypatch):
    monkeypatch.setenv(
        "AGENT_GUILD_BASE_URL",
        "https://user:secret@example.com:8443/api?token=hidden#fragment",
    )

    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        result = json.loads(_request("/check?capability=code-review"))

    assert result["error"] == "agent_guild_unreachable"
    assert result["endpoint"] == "https://example.com:8443/api"
    assert "secret" not in json.dumps(result)
    assert "hidden" not in json.dumps(result)
