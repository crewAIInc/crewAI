"""Tests for the CLIProvider and formatting helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from crewai.new_agent.cli_provider import (
    CLIProvider,
    format_elapsed,
    format_status_line,
    format_tokens,
)
from crewai.new_agent.models import AgentStatus, Message
from crewai.new_agent.provider import ConversationalProvider


# ── format_tokens ────────────────────────────────────────────


class TestFormatTokens:
    def test_zero(self):
        assert format_tokens(0) == "0"

    def test_small(self):
        assert format_tokens(999) == "999"

    def test_one_thousand(self):
        assert format_tokens(1000) == "1.0k"

    def test_thousands(self):
        assert format_tokens(1234) == "1.2k"

    def test_tens_of_thousands(self):
        assert format_tokens(12345) == "12.3k"

    def test_hundreds_of_thousands(self):
        assert format_tokens(123456) == "123.5k"

    def test_millions(self):
        assert format_tokens(1234567) == "1.2M"

    def test_large_millions(self):
        assert format_tokens(12345678) == "12.3M"

    def test_one(self):
        assert format_tokens(1) == "1"

    def test_boundary_999(self):
        assert format_tokens(999) == "999"

    def test_boundary_999999(self):
        assert format_tokens(999999) == "1000.0k"

    def test_boundary_1000000(self):
        assert format_tokens(1000000) == "1.0M"


# ── format_elapsed ───────────────────────────────────────────


class TestFormatElapsed:
    def test_seconds(self):
        assert format_elapsed(12000) == "12s"

    def test_zero(self):
        assert format_elapsed(0) == "0s"

    def test_one_minute(self):
        assert format_elapsed(60000) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert format_elapsed(72000) == "1m 12s"

    def test_one_hour(self):
        assert format_elapsed(3600000) == "1h 0m"

    def test_hours_and_minutes(self):
        assert format_elapsed(3723000) == "1h 2m"

    def test_under_one_second(self):
        assert format_elapsed(500) == "0s"

    def test_59_seconds(self):
        assert format_elapsed(59000) == "59s"


# ── format_status_line ───────────────────────────────────────


class TestFormatStatusLine:
    def test_basic_status(self):
        status = AgentStatus(state="thinking")
        line = format_status_line(status)
        assert line == "⠋ thinking…"

    def test_with_detail(self):
        status = AgentStatus(state="using_tool", detail="Searching the web")
        line = format_status_line(status)
        assert line == "⠋ Searching the web…"

    def test_with_elapsed(self):
        status = AgentStatus(state="thinking", detail="Analyzing", elapsed_ms=12000)
        line = format_status_line(status)
        assert line == "⠋ Analyzing… (12s)"

    def test_with_tokens(self):
        status = AgentStatus(
            state="using_tool",
            detail="Searching the web",
            elapsed_ms=12000,
            input_tokens=3400,
            output_tokens=1200,
        )
        line = format_status_line(status)
        assert line == "⠋ Searching the web… (12s · ↓ 3.4k tokens · ↑ 1.2k tokens)"

    def test_custom_spinner_frame(self):
        status = AgentStatus(state="thinking", detail="Working")
        line = format_status_line(status, spinner_frame="⠸")
        assert line.startswith("⠸ Working…")

    def test_only_input_tokens(self):
        status = AgentStatus(
            state="thinking",
            detail="Reading",
            elapsed_ms=5000,
            input_tokens=500,
            output_tokens=0,
        )
        line = format_status_line(status)
        assert line == "⠋ Reading… (5s · ↓ 500 tokens)"

    def test_only_output_tokens(self):
        status = AgentStatus(
            state="thinking",
            detail="Writing",
            elapsed_ms=0,
            input_tokens=0,
            output_tokens=2500,
        )
        line = format_status_line(status)
        assert line == "⠋ Writing… (↑ 2.5k tokens)"


# ── CLIProvider protocol conformance ─────────────────────────


class TestCLIProviderProtocol:
    def test_implements_protocol(self):
        provider = CLIProvider(agent_name="test-agent")
        assert isinstance(provider, ConversationalProvider)

    def test_has_required_methods(self):
        provider = CLIProvider()
        assert hasattr(provider, "send_message")
        assert hasattr(provider, "receive_message")
        assert hasattr(provider, "send_status")
        assert hasattr(provider, "get_history")
        assert hasattr(provider, "save_history")
        assert hasattr(provider, "reset_history")


# ── CLIProvider history persistence ──────────────────────────


class TestCLIProviderHistory:
    @pytest.fixture()
    def provider(self, tmp_path, monkeypatch):
        """Create a CLIProvider that stores history in a temp dir."""
        monkeypatch.chdir(tmp_path)
        return CLIProvider(agent_name="test-agent")

    def test_get_history_empty(self, provider):
        assert provider.get_history() == []

    def test_save_and_load(self, provider):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="agent", content="Hi there", sender="TestAgent"),
        ]
        provider.save_history(messages)
        loaded = provider.get_history()
        assert len(loaded) == 2
        assert loaded[0].role == "user"
        assert loaded[0].content == "Hello"
        assert loaded[1].role == "agent"
        assert loaded[1].content == "Hi there"
        assert loaded[1].sender == "TestAgent"

    def test_reset_history(self, provider, tmp_path):
        messages = [Message(role="user", content="Hello")]
        provider.save_history(messages)
        assert len(provider.get_history()) == 1

        provider.reset_history()
        assert provider.get_history() == []

    def test_reset_nonexistent_history(self, provider):
        # Should not raise
        provider.reset_history()

    def test_history_creates_directories(self, provider, tmp_path):
        messages = [Message(role="user", content="Hello")]
        provider.save_history(messages)
        db_path = tmp_path / ".crewai" / "conversations" / "test-agent.db"
        assert db_path.exists()

    def test_history_roundtrip_preserves_fields(self, provider):
        msg = Message(
            role="agent",
            content="Result",
            sender="Researcher",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            tools_used=["search"],
        )
        provider.save_history([msg])
        loaded = provider.get_history()
        assert loaded[0].sender == "Researcher"
        assert loaded[0].model == "gpt-4o"
        assert loaded[0].input_tokens == 100
        assert loaded[0].output_tokens == 50
        assert loaded[0].tools_used == ["search"]


# ── CLIProvider send_message ─────────────────────────────────


class TestCLIProviderSendMessage:
    def test_send_agent_message(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        provider = CLIProvider(agent_name="test")
        msg = Message(role="agent", content="Hello!", sender="Researcher")
        asyncio.run(provider.send_message(msg))
        captured = capsys.readouterr()
        assert "Researcher: Hello!" in captured.out

    def test_send_system_message(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        provider = CLIProvider(agent_name="test")
        msg = Message(role="system", content="Agent initialized")
        asyncio.run(provider.send_message(msg))
        captured = capsys.readouterr()
        assert "[system] Agent initialized" in captured.out

    def test_send_agent_message_no_sender(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        provider = CLIProvider(agent_name="test")
        msg = Message(role="agent", content="Hi")
        asyncio.run(provider.send_message(msg))
        captured = capsys.readouterr()
        assert "Agent: Hi" in captured.out
