import os
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.ejentum_reasoning_harness_tool.ejentum_reasoning_harness_tool import (
    EjentumHarnessTool,
)


def _mock_response(status_code: int = 200, json_data=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or (str(json_data) if json_data else "")
    resp.json.return_value = json_data if json_data is not None else []
    return resp


def test_missing_api_key_returns_actionable_error(monkeypatch):
    monkeypatch.delenv("EJENTUM_API_KEY", raising=False)
    tool = EjentumHarnessTool()
    result = tool._run(query="diagnose 503s under load", mode="reasoning")
    assert "EJENTUM_API_KEY" in result
    assert "ejentum.com/pricing" in result


def test_invalid_mode_returns_validation_error():
    tool = EjentumHarnessTool()
    result = tool._run(query="anything", mode="not-a-mode")
    assert "mode" in result
    assert "reasoning" in result and "code" in result


def test_missing_query_returns_validation_error():
    tool = EjentumHarnessTool()
    result = tool._run(query="", mode="reasoning")
    assert "query" in result


def test_whitespace_only_query_returns_validation_error():
    """Whitespace-only input must NOT trigger a paid external request."""
    tool = EjentumHarnessTool()
    result = tool._run(query="   \t\n  ", mode="reasoning")
    assert "query" in result.lower()
    assert "required" in result.lower()


@patch(
    "crewai_tools.tools.ejentum_reasoning_harness_tool.ejentum_reasoning_harness_tool.requests.post"
)
def test_successful_reasoning_call_returns_scaffold(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    mock_post.return_value = _mock_response(
        status_code=200,
        json_data=[{"reasoning": "[NEGATIVE GATE]\n... [PROCEDURE]\n..."}],
    )

    tool = EjentumHarnessTool()
    result = tool._run(query="diagnose 503s under load", mode="reasoning")

    assert "[NEGATIVE GATE]" in result
    assert "[PROCEDURE]" in result
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"] == {
        "query": "diagnose 503s under load",
        "mode": "reasoning",
    }


@patch(
    "crewai_tools.tools.ejentum_reasoning_harness_tool.ejentum_reasoning_harness_tool.requests.post"
)
def test_401_returns_actionable_error(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "bad-key")
    mock_post.return_value = _mock_response(status_code=401, text="Unauthorized")

    tool = EjentumHarnessTool()
    result = tool._run(query="anything", mode="anti-deception")

    assert "401" in result
    assert "EJENTUM_API_KEY" in result


@patch(
    "crewai_tools.tools.ejentum_reasoning_harness_tool.ejentum_reasoning_harness_tool.requests.post"
)
def test_unexpected_response_shape_is_handled(mock_post, monkeypatch):
    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    mock_post.return_value = _mock_response(status_code=200, json_data={"wrong": "shape"})

    tool = EjentumHarnessTool()
    result = tool._run(query="anything", mode="code")

    assert "unexpected response shape" in result.lower()


@patch(
    "crewai_tools.tools.ejentum_reasoning_harness_tool.ejentum_reasoning_harness_tool.requests.post"
)
def test_network_error_is_caught(mock_post, monkeypatch):
    import requests

    monkeypatch.setenv("EJENTUM_API_KEY", "test-key")
    mock_post.side_effect = requests.ConnectionError("simulated")

    tool = EjentumHarnessTool()
    result = tool._run(query="anything", mode="memory")

    assert "network error" in result.lower()
    assert "simulated" in result
