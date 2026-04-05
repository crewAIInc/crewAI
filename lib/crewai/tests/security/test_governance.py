"""Tests for the security governance module.

Tests cover:
- SubprocessPolicy: command allowlist/blocklist, shell validation, custom validators
- HttpPolicy: domain allowlist/blocklist, URL pattern matching, custom validators
- ToolPolicy: tool allowlist/blocklist, custom validators
- GovernanceConfig: aggregated policy validation
- Integration with SecurityConfig
- Integration with agent subprocess calls
- Integration with tool execution governance
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crewai.security import (
    GovernanceConfig,
    GovernanceError,
    HttpPolicy,
    SecurityConfig,
    SubprocessPolicy,
    ToolPolicy,
)


# ---------------------------------------------------------------------------
# SubprocessPolicy tests
# ---------------------------------------------------------------------------


class TestSubprocessPolicy:
    """Tests for SubprocessPolicy."""

    def test_default_policy_allows_all(self) -> None:
        """Default policy with no restrictions should allow any command."""
        policy = SubprocessPolicy()
        # Should not raise
        policy.validate_command(["docker", "info"])
        policy.validate_command(["git", "status"])
        policy.validate_command(["python", "-c", "print('hello')"])

    def test_allowed_commands_allowlist(self) -> None:
        """Only commands in the allowlist should be permitted."""
        policy = SubprocessPolicy(allowed_commands=["docker", "git"])
        policy.validate_command(["docker", "info"])  # allowed
        policy.validate_command(["git", "status"])  # allowed

        with pytest.raises(GovernanceError, match="not in the allowed commands list"):
            policy.validate_command(["rm", "-rf", "/"])

    def test_blocked_commands_blocklist(self) -> None:
        """Blocked commands should always be denied."""
        policy = SubprocessPolicy(blocked_commands=["rm", "shutdown"])
        policy.validate_command(["docker", "info"])  # allowed

        with pytest.raises(GovernanceError, match="blocked by policy"):
            policy.validate_command(["rm", "-rf", "/"])

        with pytest.raises(GovernanceError, match="blocked by policy"):
            policy.validate_command(["shutdown", "-h", "now"])

    def test_blocked_takes_precedence_over_allowed(self) -> None:
        """A command in both allowed and blocked should be denied."""
        policy = SubprocessPolicy(
            allowed_commands=["docker", "rm"],
            blocked_commands=["rm"],
        )
        policy.validate_command(["docker", "info"])  # allowed

        with pytest.raises(GovernanceError, match="blocked by policy"):
            policy.validate_command(["rm", "-rf", "/"])

    def test_shell_not_allowed_by_default(self) -> None:
        """shell=True should be blocked by default."""
        policy = SubprocessPolicy()

        with pytest.raises(GovernanceError, match="shell=True is not permitted"):
            policy.validate_command(["echo", "hello"], shell=True)

    def test_shell_allowed_when_configured(self) -> None:
        """shell=True should be allowed when explicitly configured."""
        policy = SubprocessPolicy(allow_shell=True)
        # Should not raise
        policy.validate_command(["echo", "hello"], shell=True)

    def test_empty_command_rejected(self) -> None:
        """Empty command list should be rejected."""
        policy = SubprocessPolicy()

        with pytest.raises(GovernanceError, match="Empty command"):
            policy.validate_command([])

    def test_custom_validator_allows(self) -> None:
        """Custom validator returning True should allow execution."""
        validator = MagicMock(return_value=True)
        policy = SubprocessPolicy(custom_validator=validator)
        policy.validate_command(["docker", "info"])
        validator.assert_called_once()

    def test_custom_validator_blocks(self) -> None:
        """Custom validator returning False should block execution."""
        validator = MagicMock(return_value=False)
        policy = SubprocessPolicy(custom_validator=validator)

        with pytest.raises(GovernanceError, match="rejected by custom validator"):
            policy.validate_command(["docker", "info"])

    def test_command_basename_extraction(self) -> None:
        """Full paths should be reduced to basename for matching."""
        policy = SubprocessPolicy(allowed_commands=["docker"])
        # Should match "docker" even with full path
        policy.validate_command(["/usr/bin/docker", "info"])

    def test_governance_error_attributes(self) -> None:
        """GovernanceError should carry category and detail."""
        policy = SubprocessPolicy(blocked_commands=["rm"])

        with pytest.raises(GovernanceError) as exc_info:
            policy.validate_command(["rm", "-rf", "/"])

        error = exc_info.value
        assert error.category == "subprocess"
        assert "rm" in error.detail
        assert "blocked by policy" in error.detail


# ---------------------------------------------------------------------------
# HttpPolicy tests
# ---------------------------------------------------------------------------


class TestHttpPolicy:
    """Tests for HttpPolicy."""

    def test_default_policy_allows_all(self) -> None:
        """Default policy with no restrictions should allow any URL."""
        policy = HttpPolicy()
        policy.validate_request("https://api.openai.com/v1/chat")
        policy.validate_request("https://example.com/data", method="POST")

    def test_allowed_domains_allowlist(self) -> None:
        """Only requests to allowed domains should be permitted."""
        policy = HttpPolicy(allowed_domains=["api.openai.com", "api.anthropic.com"])
        policy.validate_request("https://api.openai.com/v1/chat")

        with pytest.raises(GovernanceError, match="not in the allowed domains list"):
            policy.validate_request("https://evil.example.com/steal")

    def test_blocked_domains_blocklist(self) -> None:
        """Blocked domains should always be denied."""
        policy = HttpPolicy(blocked_domains=["evil.example.com"])
        policy.validate_request("https://api.openai.com/v1/chat")

        with pytest.raises(GovernanceError, match="blocked by policy"):
            policy.validate_request("https://evil.example.com/steal")

    def test_blocked_takes_precedence_over_allowed(self) -> None:
        """A domain in both allowed and blocked should be denied."""
        policy = HttpPolicy(
            allowed_domains=["api.openai.com", "evil.example.com"],
            blocked_domains=["evil.example.com"],
        )

        with pytest.raises(GovernanceError, match="blocked by policy"):
            policy.validate_request("https://evil.example.com/data")

    def test_url_pattern_matching(self) -> None:
        """URLs should be validated against regex patterns."""
        policy = HttpPolicy(
            allowed_url_patterns=[r"https://api\.openai\.com/.*"]
        )
        policy.validate_request("https://api.openai.com/v1/chat")

        with pytest.raises(GovernanceError, match="does not match any allowed URL pattern"):
            policy.validate_request("https://evil.com/steal")

    def test_custom_validator_allows(self) -> None:
        """Custom validator returning True should allow the request."""
        validator = MagicMock(return_value=True)
        policy = HttpPolicy(custom_validator=validator)
        policy.validate_request("https://api.openai.com/v1/chat", method="POST")
        validator.assert_called_once()

    def test_custom_validator_blocks(self) -> None:
        """Custom validator returning False should block the request."""
        validator = MagicMock(return_value=False)
        policy = HttpPolicy(custom_validator=validator)

        with pytest.raises(GovernanceError, match="rejected by custom validator"):
            policy.validate_request("https://api.openai.com/v1/chat")

    def test_governance_error_attributes(self) -> None:
        """GovernanceError should carry category and detail for HTTP."""
        policy = HttpPolicy(blocked_domains=["evil.com"])

        with pytest.raises(GovernanceError) as exc_info:
            policy.validate_request("https://evil.com/data")

        error = exc_info.value
        assert error.category == "http"
        assert "evil.com" in error.detail


# ---------------------------------------------------------------------------
# ToolPolicy tests
# ---------------------------------------------------------------------------


class TestToolPolicy:
    """Tests for ToolPolicy."""

    def test_default_policy_allows_all(self) -> None:
        """Default policy with no restrictions should allow any tool."""
        policy = ToolPolicy()
        policy.validate_tool("search", {"query": "hello"})
        policy.validate_tool("read_file", {"path": "/tmp/test"})

    def test_allowed_tools_allowlist(self) -> None:
        """Only tools in the allowlist should be permitted."""
        policy = ToolPolicy(allowed_tools=["search", "read_file"])
        policy.validate_tool("search", {"query": "hello"})

        with pytest.raises(GovernanceError, match="not in the allowed tools list"):
            policy.validate_tool("delete_database", {})

    def test_blocked_tools_blocklist(self) -> None:
        """Blocked tools should always be denied."""
        policy = ToolPolicy(blocked_tools=["delete_database", "execute_code"])
        policy.validate_tool("search", {"query": "hello"})

        with pytest.raises(GovernanceError, match="blocked by policy"):
            policy.validate_tool("delete_database", {})

    def test_blocked_takes_precedence_over_allowed(self) -> None:
        """A tool in both allowed and blocked should be denied."""
        policy = ToolPolicy(
            allowed_tools=["search", "delete_database"],
            blocked_tools=["delete_database"],
        )

        with pytest.raises(GovernanceError, match="blocked by policy"):
            policy.validate_tool("delete_database", {})

    def test_custom_validator_allows(self) -> None:
        """Custom validator returning True should allow the tool."""
        validator = MagicMock(return_value=True)
        policy = ToolPolicy(custom_validator=validator)
        policy.validate_tool("search", {"query": "hello"})
        validator.assert_called_once_with("search", {"query": "hello"})

    def test_custom_validator_blocks(self) -> None:
        """Custom validator returning False should block the tool."""
        validator = MagicMock(return_value=False)
        policy = ToolPolicy(custom_validator=validator)

        with pytest.raises(GovernanceError, match="rejected by custom validator"):
            policy.validate_tool("search", {"query": "hello"})

    def test_none_input_handled(self) -> None:
        """None input should be handled gracefully."""
        policy = ToolPolicy()
        policy.validate_tool("search", None)

    def test_governance_error_attributes(self) -> None:
        """GovernanceError should carry category and detail for tools."""
        policy = ToolPolicy(blocked_tools=["danger"])

        with pytest.raises(GovernanceError) as exc_info:
            policy.validate_tool("danger", {})

        error = exc_info.value
        assert error.category == "tool"
        assert "danger" in error.detail


# ---------------------------------------------------------------------------
# GovernanceConfig tests
# ---------------------------------------------------------------------------


class TestGovernanceConfig:
    """Tests for the aggregated GovernanceConfig."""

    def test_default_governance_allows_all(self) -> None:
        """Default GovernanceConfig should allow all operations."""
        config = GovernanceConfig()
        config.validate_subprocess(["docker", "info"])
        config.validate_http("https://api.openai.com/v1/chat")
        config.validate_tool("search", {"query": "hello"})

    def test_governance_with_all_policies(self) -> None:
        """GovernanceConfig should enforce all configured policies."""
        config = GovernanceConfig(
            subprocess_policy=SubprocessPolicy(
                allowed_commands=["docker"],
                blocked_commands=["rm"],
            ),
            http_policy=HttpPolicy(
                allowed_domains=["api.openai.com"],
            ),
            tool_policy=ToolPolicy(
                blocked_tools=["delete_database"],
            ),
        )

        # Allowed operations
        config.validate_subprocess(["docker", "info"])
        config.validate_http("https://api.openai.com/v1/chat")
        config.validate_tool("search", {"query": "hello"})

        # Blocked operations
        with pytest.raises(GovernanceError, match="subprocess"):
            config.validate_subprocess(["rm", "-rf", "/"])

        with pytest.raises(GovernanceError, match="http"):
            config.validate_http("https://evil.com/steal")

        with pytest.raises(GovernanceError, match="tool"):
            config.validate_tool("delete_database", {})

    def test_governance_serialization(self) -> None:
        """GovernanceConfig should be serializable."""
        config = GovernanceConfig(
            subprocess_policy=SubprocessPolicy(
                allowed_commands=["docker"],
                blocked_commands=["rm"],
            ),
            http_policy=HttpPolicy(
                blocked_domains=["evil.com"],
            ),
            tool_policy=ToolPolicy(
                blocked_tools=["danger"],
            ),
        )

        data = config.model_dump()
        assert data["subprocess_policy"]["allowed_commands"] == ["docker"]
        assert data["subprocess_policy"]["blocked_commands"] == ["rm"]
        assert data["http_policy"]["blocked_domains"] == ["evil.com"]
        assert data["tool_policy"]["blocked_tools"] == ["danger"]


# ---------------------------------------------------------------------------
# SecurityConfig integration tests
# ---------------------------------------------------------------------------


class TestSecurityConfigGovernance:
    """Tests for governance integration in SecurityConfig."""

    def test_security_config_has_default_governance(self) -> None:
        """SecurityConfig should have default governance that allows all."""
        config = SecurityConfig()
        assert config.governance is not None
        assert isinstance(config.governance, GovernanceConfig)

        # Default governance allows everything
        config.governance.validate_subprocess(["docker", "info"])
        config.governance.validate_http("https://api.openai.com/v1/chat")
        config.governance.validate_tool("search", {})

    def test_security_config_with_custom_governance(self) -> None:
        """SecurityConfig should accept a custom GovernanceConfig."""
        governance = GovernanceConfig(
            subprocess_policy=SubprocessPolicy(blocked_commands=["rm"]),
            tool_policy=ToolPolicy(blocked_tools=["danger"]),
        )
        config = SecurityConfig(governance=governance)

        with pytest.raises(GovernanceError):
            config.governance.validate_subprocess(["rm", "-rf", "/"])

        with pytest.raises(GovernanceError):
            config.governance.validate_tool("danger", {})

    def test_security_config_to_dict_includes_governance(self) -> None:
        """to_dict() should include governance configuration."""
        governance = GovernanceConfig(
            subprocess_policy=SubprocessPolicy(blocked_commands=["rm"]),
        )
        config = SecurityConfig(governance=governance)
        config_dict = config.to_dict()

        assert "governance" in config_dict
        assert "subprocess_policy" in config_dict["governance"]
        assert config_dict["governance"]["subprocess_policy"]["blocked_commands"] == ["rm"]

    def test_security_config_from_dict_with_governance(self) -> None:
        """from_dict() should restore governance configuration."""
        original = SecurityConfig(
            governance=GovernanceConfig(
                subprocess_policy=SubprocessPolicy(blocked_commands=["rm"]),
                http_policy=HttpPolicy(blocked_domains=["evil.com"]),
                tool_policy=ToolPolicy(blocked_tools=["danger"]),
            )
        )

        config_dict = original.to_dict()
        restored = SecurityConfig.from_dict(config_dict)

        assert restored.governance is not None
        assert "rm" in restored.governance.subprocess_policy.blocked_commands
        assert "evil.com" in restored.governance.http_policy.blocked_domains
        assert "danger" in restored.governance.tool_policy.blocked_tools


# ---------------------------------------------------------------------------
# Agent subprocess governance integration tests
# ---------------------------------------------------------------------------


class TestAgentSubprocessGovernance:
    """Tests for governance enforcement in agent subprocess calls."""

    def test_validate_docker_blocked_by_governance(self) -> None:
        """Agent._validate_docker_installation should respect subprocess governance."""
        from crewai.agent.core import Agent

        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            security_config=SecurityConfig(
                governance=GovernanceConfig(
                    subprocess_policy=SubprocessPolicy(
                        blocked_commands=["docker"],
                    ),
                ),
            ),
        )

        with patch("shutil.which", return_value="/usr/bin/docker"):
            with pytest.raises(RuntimeError, match="Governance policy blocked"):
                agent._validate_docker_installation()

    def test_validate_docker_allowed_by_governance(self) -> None:
        """Agent._validate_docker_installation should pass with permissive governance."""
        from crewai.agent.core import Agent

        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            security_config=SecurityConfig(
                governance=GovernanceConfig(
                    subprocess_policy=SubprocessPolicy(
                        allowed_commands=["docker"],
                    ),
                ),
            ),
        )

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            # Should not raise
            agent._validate_docker_installation()

    def test_validate_docker_not_in_allowlist(self) -> None:
        """Subprocess governance should block docker when not in allowlist."""
        from crewai.agent.core import Agent

        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            security_config=SecurityConfig(
                governance=GovernanceConfig(
                    subprocess_policy=SubprocessPolicy(
                        allowed_commands=["git"],  # docker not allowed
                    ),
                ),
            ),
        )

        with patch("shutil.which", return_value="/usr/bin/docker"):
            with pytest.raises(RuntimeError, match="Governance policy blocked"):
                agent._validate_docker_installation()


# ---------------------------------------------------------------------------
# Tool governance integration tests
# ---------------------------------------------------------------------------


class TestToolGovernanceIntegration:
    """Tests for tool governance enforcement in the tool execution path."""

    def test_tool_blocked_by_governance_sync(self) -> None:
        """execute_tool_and_check_finality should block tools per governance."""
        from crewai.agents.parser import AgentAction
        from crewai.security.security_config import SecurityConfig
        from crewai.utilities.tool_utils import execute_tool_and_check_finality

        # Create a mock crew with governance that blocks 'danger_tool'
        mock_crew = MagicMock()
        mock_crew.verbose = False
        mock_crew.security_config = SecurityConfig(
            governance=GovernanceConfig(
                tool_policy=ToolPolicy(blocked_tools=["danger_tool"]),
            )
        )

        # Create a mock tool with attributes needed by ToolUsage internals
        mock_tool = MagicMock()
        mock_tool.name = "Danger Tool"
        mock_tool.result_as_answer = False
        mock_tool.description = "Danger Tool: A dangerous tool, args: {}"

        agent_action = AgentAction(
            tool="danger_tool",
            tool_input="{}",
            text='Action: Danger Tool\nAction Input: {}',
            thought="",
        )

        from crewai.utilities.i18n import get_i18n

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[mock_tool],
            i18n=get_i18n(),
            crew=mock_crew,
        )

        assert "blocked by governance policy" in result.result

    def test_tool_allowed_by_governance(self) -> None:
        """Tools not blocked by governance should proceed normally."""
        governance = GovernanceConfig(
            tool_policy=ToolPolicy(
                allowed_tools=["search_tool"],
            ),
        )

        # Allowed tool should not raise
        governance.validate_tool("search_tool", {"query": "test"})

        # Disallowed tool should raise
        with pytest.raises(GovernanceError):
            governance.validate_tool("other_tool", {})


# ---------------------------------------------------------------------------
# GovernanceError tests
# ---------------------------------------------------------------------------


class TestGovernanceError:
    """Tests for the GovernanceError exception."""

    def test_error_message_format(self) -> None:
        """GovernanceError message should include category and detail."""
        error = GovernanceError("subprocess", "Command 'rm' is blocked")
        assert str(error) == "[subprocess] Command 'rm' is blocked"
        assert error.category == "subprocess"
        assert error.detail == "Command 'rm' is blocked"

    def test_error_inherits_from_exception(self) -> None:
        """GovernanceError should be catchable as Exception."""
        with pytest.raises(Exception):
            raise GovernanceError("test", "test detail")

    def test_error_categories(self) -> None:
        """Different governance categories should be properly tracked."""
        subprocess_err = GovernanceError("subprocess", "blocked")
        http_err = GovernanceError("http", "blocked")
        tool_err = GovernanceError("tool", "blocked")

        assert subprocess_err.category == "subprocess"
        assert http_err.category == "http"
        assert tool_err.category == "tool"
