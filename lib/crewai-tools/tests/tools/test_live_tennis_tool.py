"""Tests for LiveTennisTool.

All HTTP traffic is mocked — no test touches the network and no real
Live Tennis API key is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from crewai_tools.tools.live_tennis_tool.live_tennis_tool import (
    API_KEY_ENV_VAR,
    LiveTennisTool,
)
from crewai_tools.tools.live_tennis_tool.schemas import LiveTennisToolSchema


REQUESTS_GET = "crewai_tools.tools.live_tennis_tool.live_tennis_tool.requests.get"


def _response(status_code=200, payload=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    if payload is not None:
        response.json.return_value = payload
        response.text = json.dumps(payload)
    else:
        response.json.side_effect = ValueError("no json")
        response.text = text
    return response


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    return LiveTennisTool()


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_schema_requires_search_for_player_search():
    with pytest.raises(ValidationError, match="search"):
        LiveTennisToolSchema(action="search_players")


def test_schema_requires_player_id_for_profile():
    with pytest.raises(ValidationError, match="player_id"):
        LiveTennisToolSchema(action="player_profile")


def test_schema_requires_system_for_rankings():
    with pytest.raises(ValidationError, match="system"):
        LiveTennisToolSchema(action="rankings")


def test_schema_rejects_unknown_action():
    with pytest.raises(ValidationError):
        LiveTennisToolSchema(action="win_probability")


# ---------------------------------------------------------------------------
# Missing / invalid credentials
# ---------------------------------------------------------------------------


def test_missing_api_key_returns_helpful_message(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    result = LiveTennisTool()._run(action="live_matches")
    assert API_KEY_ENV_VAR in result
    assert "livetennisapi.com" in result


def test_invalid_key_maps_401_to_message(tool):
    with patch(REQUESTS_GET, return_value=_response(401, text="unauthorized")):
        result = tool._run(action="live_matches")
    assert "401" in result


def test_upgrade_required_maps_403_to_message(tool):
    body = {"error": "upgrade_required"}
    with patch(REQUESTS_GET, return_value=_response(403, payload=body)):
        result = tool._run(action="rankings", system="atp")
    assert "403" in result
    assert "upgrade_required" in result


def test_rate_limit_maps_429_to_message(tool):
    body = {"error": "rate_limited"}
    with patch(REQUESTS_GET, return_value=_response(429, payload=body)):
        result = tool._run(action="live_matches")
    assert "429" in result
    assert "rate_limited" in result


def test_request_exception_is_caught(tool):
    import requests

    with patch(REQUESTS_GET, side_effect=requests.ConnectionError("boom")):
        result = tool._run(action="live_matches")
    assert "request failed" in result
    assert "boom" in result


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def test_live_matches_request(tool):
    payload = {"data": [], "meta": {"count": 0}}
    with patch(REQUESTS_GET, return_value=_response(200, payload=payload)) as mock_get:
        result = tool._run(action="live_matches", tour="atp", limit=5)

    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.livetennisapi.com/api/public/v1/matches"
    assert kwargs["params"] == {"limit": 5, "status": "live", "tour": "atp"}
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert kwargs["timeout"] == tool.request_timeout
    assert json.loads(result) == payload


def test_upcoming_matches_request(tool):
    with patch(REQUESTS_GET, return_value=_response(200, payload={"data": []})) as mock_get:
        tool._run(action="upcoming_matches")
    assert mock_get.call_args.kwargs["params"] == {"status": "upcoming"}


def test_fixtures_request(tool):
    with patch(REQUESTS_GET, return_value=_response(200, payload={"data": []})) as mock_get:
        tool._run(action="fixtures", tour="wta", offset=10)
    args, kwargs = mock_get.call_args
    assert args[0].endswith("/fixtures")
    assert kwargs["params"] == {"offset": 10, "tour": "wta"}


def test_search_players_request(tool):
    with patch(REQUESTS_GET, return_value=_response(200, payload={"data": []})) as mock_get:
        tool._run(action="search_players", search="alcaraz")
    args, kwargs = mock_get.call_args
    assert args[0].endswith("/players")
    assert kwargs["params"] == {"search": "alcaraz"}


def test_player_profile_request_quotes_id(tool):
    with patch(REQUESTS_GET, return_value=_response(200, payload={"id": "p 1"})) as mock_get:
        tool._run(action="player_profile", player_id="p 1")
    args, kwargs = mock_get.call_args
    assert args[0].endswith("/players/p%201")
    assert kwargs["params"] == {}


def test_rankings_request(tool):
    with patch(REQUESTS_GET, return_value=_response(200, payload={"data": []})) as mock_get:
        tool._run(action="rankings", system="wta", limit=10)
    args, kwargs = mock_get.call_args
    assert args[0].endswith("/rankings")
    assert kwargs["params"] == {"limit": 10, "system": "wta"}


def test_usage_request(tool):
    with patch(REQUESTS_GET, return_value=_response(200, payload={"tier": "FREE"})) as mock_get:
        result = tool._run(action="usage")
    args, kwargs = mock_get.call_args
    assert args[0].endswith("/usage")
    assert kwargs["params"] == {}
    assert json.loads(result) == {"tier": "FREE"}


def test_invalid_arguments_return_message_not_raise(tool):
    with patch(REQUESTS_GET) as mock_get:
        result = tool._run(action="search_players")
    mock_get.assert_not_called()
    assert "Invalid arguments" in result


def test_unexpected_status_is_reported(tool):
    with patch(REQUESTS_GET, return_value=_response(500, text="server error")):
        result = tool._run(action="live_matches")
    assert "500" in result
    assert "server error" in result
