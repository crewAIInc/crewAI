"""Tests for StdioTransport command allowlist validation."""

import pytest

from crewai.mcp.transports.stdio import DEFAULT_ALLOWED_COMMANDS, StdioTransport


class TestStdioTransportAllowlist:
    """Tests for the command allowlist feature."""

    def test_default_allowed_commands_contains_common_runtimes(self):
        """DEFAULT_ALLOWED_COMMANDS should include all common MCP server runtimes."""
        expected = {"python", "python3", "node", "npx", "uvx", "uv", "deno", "docker"}
        assert expected == DEFAULT_ALLOWED_COMMANDS

    def test_allowed_command_passes_validation(self):
        """Commands in the default allowlist should be accepted."""
        for cmd in DEFAULT_ALLOWED_COMMANDS:
            transport = StdioTransport(command=cmd, args=["server.py"])
            assert transport.command == cmd

    def test_allowed_command_with_full_path(self):
        """Full paths to allowed commands should pass (basename is checked)."""
        transport = StdioTransport(command="/usr/bin/python3", args=["server.py"])
        assert transport.command == "/usr/bin/python3"

    def test_disallowed_command_raises_value_error(self):
        """Commands not in the allowlist should raise ValueError."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            StdioTransport(command="malicious-binary", args=["--evil"])

    def test_disallowed_command_with_full_path_raises(self):
        """Full paths to disallowed commands should also be rejected."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            StdioTransport(command="/tmp/evil/script", args=[])

    def test_allowed_commands_none_disables_validation(self):
        """Setting allowed_commands=None should disable the check entirely."""
        transport = StdioTransport(
            command="any-custom-binary",
            args=["--flag"],
            allowed_commands=None,
        )
        assert transport.command == "any-custom-binary"

    def test_custom_allowlist(self):
        """Users should be able to pass a custom allowlist."""
        custom = frozenset({"my-server", "python"})

        # Allowed
        transport = StdioTransport(
            command="my-server", args=[], allowed_commands=custom
        )
        assert transport.command == "my-server"

        # Not allowed
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            StdioTransport(command="node", args=[], allowed_commands=custom)

    def test_extended_allowlist(self):
        """Users should be able to extend the default allowlist."""
        extended = DEFAULT_ALLOWED_COMMANDS | frozenset({"my-custom-runtime"})

        transport = StdioTransport(
            command="my-custom-runtime", args=[], allowed_commands=extended
        )
        assert transport.command == "my-custom-runtime"

        # Original defaults still work
        transport2 = StdioTransport(
            command="python", args=["server.py"], allowed_commands=extended
        )
        assert transport2.command == "python"

    def test_error_message_includes_sorted_allowed_commands(self):
        """The error message should list the allowed commands for discoverability."""
        with pytest.raises(ValueError) as exc_info:
            StdioTransport(command="bad-cmd", args=[])

        error_msg = str(exc_info.value)
        assert "bad-cmd" in error_msg
        assert "allowed_commands=None" in error_msg

    def test_args_and_env_still_work(self):
        """Existing args and env functionality should be unaffected."""
        transport = StdioTransport(
            command="python",
            args=["server.py", "--port", "8080"],
            env={"API_KEY": "test123"},
        )
        assert transport.command == "python"
        assert transport.args == ["server.py", "--port", "8080"]
        assert transport.env == {"API_KEY": "test123"}
