"""Tests for stdio transport."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.mcp.transports.stdio import StdioTransport


@pytest.mark.asyncio
async def test_ambient_env_does_not_leak_to_server(monkeypatch):
    """Ambient env vars outside the MCP SDK's default allowlist must not reach the server.

    Regression guard: previously StdioTransport did os.environ.copy(), which leaked
    every ambient var (COMPANY_SECRET, AWS_*, etc.) into every spawned MCP server.
    """
    monkeypatch.setenv("COMPANY_SECRET", "leaked")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leaked")

    transport = StdioTransport(
        command="python",
        args=["server.py"],
        env={"OPENAI_API_KEY": "sk-test"},
    )

    captured: dict[str, dict[str, str] | None] = {}

    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    fake_ctx.__aexit__ = AsyncMock(return_value=None)

    def fake_stdio_client(server_params):
        captured["env"] = server_params.env
        return fake_ctx

    with (
        patch("mcp.client.stdio.stdio_client", side_effect=fake_stdio_client),
        patch(
            "mcp.client.stdio.get_default_environment",
            return_value={"PATH": "/usr/bin", "HOME": "/home/user"},
        ),
    ):
        await transport.connect()

    env = captured["env"]
    assert env is not None
    assert "COMPANY_SECRET" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env.get("OPENAI_API_KEY") == "sk-test"
    assert env.get("PATH") == "/usr/bin"


@pytest.mark.asyncio
async def test_user_env_overrides_default_environment():
    """User-supplied env values must override keys returned by get_default_environment()."""
    transport = StdioTransport(
        command="python",
        args=["server.py"],
        env={"PATH": "/custom/bin"},
    )

    captured: dict[str, dict[str, str] | None] = {}

    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    fake_ctx.__aexit__ = AsyncMock(return_value=None)

    def fake_stdio_client(server_params):
        captured["env"] = server_params.env
        return fake_ctx

    with (
        patch("mcp.client.stdio.stdio_client", side_effect=fake_stdio_client),
        patch(
            "mcp.client.stdio.get_default_environment",
            return_value={"PATH": "/usr/bin"},
        ),
    ):
        await transport.connect()

    assert captured["env"]["PATH"] == "/custom/bin"
