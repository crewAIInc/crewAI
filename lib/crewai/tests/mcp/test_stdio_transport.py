"""Tests for StdioTransport command allowlist validation."""

import pytest

from crewai.mcp.config import MCPServerStdio
from crewai.mcp.transports.stdio import DEFAULT_ALLOWED_COMMANDS, StdioTransport


class TestDefaultAllowedCommands:
    """Verify the default allowlist contains expected runtimes."""

    def test_default_allowlist_contains_common_runtimes(self):
        expected = {"python", "python3", "node", "npx", "uvx", "deno"}
        assert expected == DEFAULT_ALLOWED_COMMANDS

    def test_default_allowlist_is_frozenset(self):
        assert isinstance(DEFAULT_ALLOWED_COMMANDS, frozenset)


class TestStdioTransportAllowlist:
    """StdioTransport should validate commands against an allowlist."""

    # -- Allowed commands (happy path) --

    @pytest.mark.parametrize("cmd", sorted(DEFAULT_ALLOWED_COMMANDS))
    def test_default_allowed_commands_accepted(self, cmd: str):
        transport = StdioTransport(command=cmd)
        assert transport.command == cmd

    def test_allowed_command_with_full_path(self):
        """Full paths should be validated by their basename."""
        transport = StdioTransport(command="/usr/bin/python3")
        assert transport.command == "/usr/bin/python3"

    def test_allowed_command_with_relative_path(self):
        transport = StdioTransport(command="./venv/bin/python")
        assert transport.command == "./venv/bin/python"

    def test_allowed_command_with_windows_style_path_requires_none(self):
        """Windows-style backslash paths aren't parsed by os.path.basename on
        POSIX, so they must opt out of the allowlist to work cross-platform."""
        transport = StdioTransport(
            command="C:\\Python311\\python", allowed_commands=None
        )
        assert transport.command == "C:\\Python311\\python"

    # -- Blocked commands --

    def test_blocked_command_raises_value_error(self):
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            StdioTransport(command="curl")

    def test_blocked_command_with_full_path(self):
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            StdioTransport(command="/usr/bin/curl")

    def test_error_message_includes_command_name(self):
        with pytest.raises(ValueError, match="curl"):
            StdioTransport(command="curl")

    def test_error_message_includes_sorted_allowlist(self):
        with pytest.raises(ValueError, match=r"\['deno', 'node', 'npx'"):
            StdioTransport(command="bash")

    def test_error_message_suggests_disabling_check(self):
        with pytest.raises(ValueError, match="allowed_commands=None"):
            StdioTransport(command="bash")

    # -- Opt-out: allowed_commands=None --

    def test_none_disables_allowlist_check(self):
        transport = StdioTransport(command="anything-goes", allowed_commands=None)
        assert transport.command == "anything-goes"

    def test_none_allows_arbitrary_path(self):
        transport = StdioTransport(
            command="/opt/custom/my-server", allowed_commands=None
        )
        assert transport.command == "/opt/custom/my-server"

    # -- Custom allowlist --

    def test_custom_allowlist_accepts_listed_command(self):
        custom = frozenset({"my-runtime"})
        transport = StdioTransport(command="my-runtime", allowed_commands=custom)
        assert transport.command == "my-runtime"

    def test_custom_allowlist_rejects_unlisted_command(self):
        custom = frozenset({"my-runtime"})
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            StdioTransport(command="python", allowed_commands=custom)

    def test_custom_allowlist_as_set(self):
        """Plain sets should also work (Set[str] type hint)."""
        custom = {"ruby", "perl"}
        transport = StdioTransport(command="ruby", allowed_commands=custom)
        assert transport.command == "ruby"

    # -- Other constructor args still work --

    def test_args_and_env_still_set(self):
        transport = StdioTransport(
            command="python",
            args=["server.py", "--port", "8080"],
            env={"KEY": "val"},
        )
        assert transport.args == ["server.py", "--port", "8080"]
        assert transport.env == {"KEY": "val"}

    def test_default_args_and_env(self):
        transport = StdioTransport(command="node")
        assert transport.args == []
        assert transport.env == {}


class TestMCPServerStdioAllowedCommands:
    """MCPServerStdio config should expose allowed_commands field."""

    def test_default_allowed_commands_on_config(self):
        config = MCPServerStdio(command="python", args=["server.py"])
        assert config.allowed_commands == DEFAULT_ALLOWED_COMMANDS

    def test_custom_allowed_commands_on_config(self):
        custom = frozenset({"ruby"})
        config = MCPServerStdio(
            command="ruby", args=["server.rb"], allowed_commands=custom
        )
        assert config.allowed_commands == custom

    def test_none_allowed_commands_on_config(self):
        config = MCPServerStdio(
            command="my-binary", args=[], allowed_commands=None
        )
        assert config.allowed_commands is None


class TestToolResolverPassesAllowedCommands:
    """_create_transport should forward allowed_commands to StdioTransport."""

    def test_create_transport_passes_allowed_commands(self):
        from crewai.mcp.tool_resolver import MCPToolResolver

        config = MCPServerStdio(
            command="python",
            args=["server.py"],
            allowed_commands=DEFAULT_ALLOWED_COMMANDS,
        )
        transport, server_name = MCPToolResolver._create_transport(config)
        assert isinstance(transport, StdioTransport)
        assert transport.command == "python"

    def test_create_transport_with_none_allowed_commands(self):
        from crewai.mcp.tool_resolver import MCPToolResolver

        config = MCPServerStdio(
            command="custom-binary",
            args=["--flag"],
            allowed_commands=None,
        )
        transport, server_name = MCPToolResolver._create_transport(config)
        assert isinstance(transport, StdioTransport)
        assert transport.command == "custom-binary"

    def test_create_transport_rejects_blocked_command(self):
        from crewai.mcp.tool_resolver import MCPToolResolver

        config = MCPServerStdio(
            command="curl",
            args=["http://evil.com"],
        )
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            MCPToolResolver._create_transport(config)
