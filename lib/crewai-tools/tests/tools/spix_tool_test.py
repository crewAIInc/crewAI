"""Unit tests for Spix tools (mocked httpx — no live API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.spix_tool.spix_tool import (
    SpixCallTool,
    SpixEmailTool,
    SpixSMSTool,
    _get_api_key,
)


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_env_key")
    assert _get_api_key(None) == "sk_env_key"


def test_get_api_key_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_env_key")
    assert _get_api_key("sk_explicit") == "sk_explicit"


def test_get_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SPIX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPIX_API_KEY"):
        _get_api_key(None)


# ---------------------------------------------------------------------------
# SpixCallTool
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.spix_tool.spix_tool.httpx.post")
def test_spix_call_tool_success(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "data": {"session_id": "sess_abc123", "status": "initiated"},
    }
    mock_post.return_value = mock_response

    tool = SpixCallTool()
    result = tool._run(
        to="+19175550123",
        playbook_id="cmp_call_abc123",
        sender="+14155550101",
    )

    assert "sess_abc123" in result
    assert "initiated" in result
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["to"] == "+19175550123"
    assert call_kwargs[1]["json"]["playbook_id"] == "cmp_call_abc123"


@patch("crewai_tools.tools.spix_tool.spix_tool.httpx.post")
def test_spix_call_tool_api_error(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": False,
        "error": {"code": "insufficient_credits", "message": "Not enough credits"},
    }
    mock_post.return_value = mock_response

    tool = SpixCallTool()
    with pytest.raises(RuntimeError, match="insufficient_credits"):
        tool._run(
            to="+19175550123",
            playbook_id="cmp_call_abc123",
            sender="+14155550101",
        )


def test_spix_call_tool_missing_api_key(monkeypatch):
    monkeypatch.delenv("SPIX_API_KEY", raising=False)
    tool = SpixCallTool()
    with pytest.raises(ValueError, match="SPIX_API_KEY"):
        tool._run(
            to="+19175550123",
            playbook_id="cmp_call_abc123",
            sender="+14155550101",
        )


# ---------------------------------------------------------------------------
# SpixSMSTool
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.spix_tool.spix_tool.httpx.post")
def test_spix_sms_tool_success(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "data": {"message_id": "msg_xyz789", "segments": 1, "credits_used": 1},
    }
    mock_post.return_value = mock_response

    tool = SpixSMSTool()
    result = tool._run(
        to="+19175550123",
        sender="+14155550101",
        body="Your appointment is confirmed.",
    )

    assert "msg_xyz789" in result
    assert "Segments: 1" in result
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["body"] == "Your appointment is confirmed."


@patch("crewai_tools.tools.spix_tool.spix_tool.httpx.post")
def test_spix_sms_tool_with_playbook_id(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "data": {"message_id": "msg_123", "segments": 1, "credits_used": 1},
    }
    mock_post.return_value = mock_response

    tool = SpixSMSTool()
    tool._run(
        to="+19175550123",
        sender="+14155550101",
        body="Hi there!",
        playbook_id="cmp_sms_abc",
    )

    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["playbook_id"] == "cmp_sms_abc"


# ---------------------------------------------------------------------------
# SpixEmailTool
# ---------------------------------------------------------------------------


@patch("crewai_tools.tools.spix_tool.spix_tool.httpx.post")
def test_spix_email_tool_success(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "data": {"message_id": "em_abc456", "credits_used": 2},
    }
    mock_post.return_value = mock_response

    tool = SpixEmailTool()
    result = tool._run(
        sender="support@spix.sh",
        to="john@example.com",
        subject="Your order is confirmed",
        body="Hi John, your order #4421 is confirmed.",
    )

    assert "em_abc456" in result
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["to"] == "john@example.com"
    assert "/email/send" in call_kwargs[0][0]


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


def test_tool_names_and_descriptions():
    assert SpixCallTool().name == "Spix Call"
    assert SpixSMSTool().name == "Spix SMS"
    assert SpixEmailTool().name == "Spix Email"
    assert "phone" in SpixCallTool().description.lower()
    assert "sms" in SpixSMSTool().description.lower()
    assert "email" in SpixEmailTool().description.lower()


def test_args_schema_fields():
    from pydantic import BaseModel

    call_schema = SpixCallTool().args_schema
    assert issubclass(call_schema, BaseModel)
    assert "to" in call_schema.model_fields
    assert "playbook_id" in call_schema.model_fields
    assert "sender" in call_schema.model_fields
