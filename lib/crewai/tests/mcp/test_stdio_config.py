"""Tests for MCPServerStdio allowed_commands config integration."""

import pytest

from crewai.mcp.config import MCPServerStdio
from crewai.mcp.transports.stdio import DEFAULT_ALLOWED_COMMANDS


class TestMCPServerStdioConfig:
    """Tests for the allowed_commands field on MCPServerStdio."""

    def test_default_allowed_commands(self):
        """MCPServerStdio should default to DEFAULT_ALLOWED_COMMANDS."""
        config = MCPServerStdio(command="python", args=["server.py"])
        assert config.allowed_commands == DEFAULT_ALLOWED_COMMANDS

    def test_custom_allowed_commands(self):
        """Users can override allowed_commands in config."""
        custom = frozenset({"my-runtime"})
        config = MCPServerStdio(
            command="my-runtime", args=[], allowed_commands=custom
        )
        assert config.allowed_commands == custom

    def test_none_allowed_commands(self):
        """Users can disable the allowlist via config."""
        config = MCPServerStdio(
            command="anything", args=[], allowed_commands=None
        )
        assert config.allowed_commands is None
