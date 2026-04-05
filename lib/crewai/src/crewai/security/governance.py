"""Security Governance Module

This module provides configurable governance policies for CrewAI to address
ungoverned call sites identified by security audits (OWASP Agentic Top 10).

Governance policies allow users to define allowlists and blocklists for:
- Subprocess execution (ASI08: Uncontrolled Code Execution)
- HTTP requests (ASI07: Data Leakage & Exfiltration)
- Tool invocations (ASI02: Tool Misuse & Exploitation)

Each policy validates operations before they execute and raises
GovernanceError when a policy violation is detected.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class GovernanceError(Exception):
    """Raised when a governance policy blocks an operation.

    Attributes:
        category: The governance category that was violated
            (e.g., 'subprocess', 'http', 'tool').
        detail: A human-readable description of the violation.
    """

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"[{category}] {detail}")


class SubprocessPolicy(BaseModel):
    """Policy governing subprocess execution.

    Controls which subprocess commands are allowed or blocked during
    agent execution. By default, all commands are allowed unless
    explicitly configured.

    Attributes:
        allowed_commands: If set, only these command basenames are permitted.
            Example: ["docker", "git", "uv"]
        blocked_commands: Commands that are always blocked, even if they
            appear in allowed_commands.
            Example: ["rm", "shutdown"]
        allow_shell: Whether shell=True is permitted in subprocess calls.
            Defaults to False for security.
        custom_validator: Optional callable that receives (command, kwargs)
            and returns True to allow or False to block.
    """

    allowed_commands: list[str] | None = Field(
        default=None,
        description=(
            "Allowlist of command basenames. "
            "If None, all commands are allowed (unless blocked)."
        ),
    )
    blocked_commands: list[str] = Field(
        default_factory=list,
        description="Blocklist of command basenames that are always denied.",
    )
    allow_shell: bool = Field(
        default=False,
        description="Whether shell=True is permitted in subprocess calls.",
    )
    custom_validator: Callable[[list[str], dict[str, Any]], bool] | None = Field(
        default=None,
        exclude=True,
        description=(
            "Optional callable(command_list, kwargs) -> bool. "
            "Return True to allow, False to block."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    def validate_command(
        self, command: list[str], *, shell: bool = False, **kwargs: Any
    ) -> None:
        """Validate a subprocess command against this policy.

        Args:
            command: The command as a list of strings (e.g., ["docker", "info"]).
            shell: Whether shell mode is requested.
            **kwargs: Additional subprocess keyword arguments.

        Raises:
            GovernanceError: If the command violates this policy.
        """
        if not command:
            raise GovernanceError("subprocess", "Empty command is not allowed.")

        if shell and not self.allow_shell:
            raise GovernanceError(
                "subprocess",
                "shell=True is not permitted by the subprocess policy.",
            )

        cmd_basename = command[0].rsplit("/", 1)[-1]

        if cmd_basename in self.blocked_commands:
            raise GovernanceError(
                "subprocess",
                f"Command '{cmd_basename}' is blocked by policy.",
            )

        if self.allowed_commands is not None and cmd_basename not in self.allowed_commands:
            raise GovernanceError(
                "subprocess",
                f"Command '{cmd_basename}' is not in the allowed commands list: "
                f"{self.allowed_commands}.",
            )

        if self.custom_validator is not None:
            if not self.custom_validator(command, kwargs):
                raise GovernanceError(
                    "subprocess",
                    f"Command '{' '.join(command)}' was rejected by custom validator.",
                )

        logger.debug("Subprocess governance: allowed command '%s'", " ".join(command))


class HttpPolicy(BaseModel):
    """Policy governing HTTP requests.

    Controls which HTTP endpoints agents are allowed to call. By default,
    all requests are allowed unless explicitly configured.

    Attributes:
        allowed_domains: If set, only requests to these domains are allowed.
            Example: ["api.openai.com", "api.anthropic.com"]
        blocked_domains: Domains that are always blocked.
            Example: ["evil.example.com"]
        allowed_url_patterns: Regex patterns that URLs must match.
            Example: [r"https://api\\.openai\\.com/.*"]
        custom_validator: Optional callable that receives (url, method, kwargs)
            and returns True to allow or False to block.
    """

    allowed_domains: list[str] | None = Field(
        default=None,
        description=(
            "Allowlist of domains. "
            "If None, all domains are allowed (unless blocked)."
        ),
    )
    blocked_domains: list[str] = Field(
        default_factory=list,
        description="Blocklist of domains that are always denied.",
    )
    allowed_url_patterns: list[str] | None = Field(
        default=None,
        description="Regex patterns that requested URLs must match.",
    )
    custom_validator: Callable[[str, str, dict[str, Any]], bool] | None = Field(
        default=None,
        exclude=True,
        description=(
            "Optional callable(url, method, kwargs) -> bool. "
            "Return True to allow, False to block."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    def validate_request(
        self, url: str, method: str = "GET", **kwargs: Any
    ) -> None:
        """Validate an HTTP request against this policy.

        Args:
            url: The target URL.
            method: HTTP method (GET, POST, etc.).
            **kwargs: Additional request keyword arguments.

        Raises:
            GovernanceError: If the request violates this policy.
        """
        parsed = urlparse(url)
        domain = parsed.hostname or ""

        if domain in self.blocked_domains:
            raise GovernanceError(
                "http",
                f"Domain '{domain}' is blocked by policy.",
            )

        if self.allowed_domains is not None and domain not in self.allowed_domains:
            raise GovernanceError(
                "http",
                f"Domain '{domain}' is not in the allowed domains list: "
                f"{self.allowed_domains}.",
            )

        if self.allowed_url_patterns is not None:
            matched = any(
                re.match(pattern, url) for pattern in self.allowed_url_patterns
            )
            if not matched:
                raise GovernanceError(
                    "http",
                    f"URL '{url}' does not match any allowed URL pattern.",
                )

        if self.custom_validator is not None:
            if not self.custom_validator(url, method, kwargs):
                raise GovernanceError(
                    "http",
                    f"Request to '{url}' ({method}) was rejected by custom validator.",
                )

        logger.debug("HTTP governance: allowed %s %s", method, url)


class ToolPolicy(BaseModel):
    """Policy governing tool invocations.

    Controls which tools agents are allowed to use. By default, all tools
    are allowed unless explicitly configured.

    Attributes:
        allowed_tools: If set, only tools with these names can be invoked.
            Example: ["search", "read_file"]
        blocked_tools: Tools that are always blocked.
            Example: ["delete_database", "execute_code"]
        custom_validator: Optional callable that receives
            (tool_name, tool_input) and returns True to allow or False to block.
    """

    allowed_tools: list[str] | None = Field(
        default=None,
        description=(
            "Allowlist of tool names. "
            "If None, all tools are allowed (unless blocked)."
        ),
    )
    blocked_tools: list[str] = Field(
        default_factory=list,
        description="Blocklist of tool names that are always denied.",
    )
    custom_validator: Callable[[str, dict[str, Any]], bool] | None = Field(
        default=None,
        exclude=True,
        description=(
            "Optional callable(tool_name, tool_input) -> bool. "
            "Return True to allow, False to block."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    def validate_tool(
        self, tool_name: str, tool_input: dict[str, Any] | None = None
    ) -> None:
        """Validate a tool invocation against this policy.

        Args:
            tool_name: Name of the tool being invoked.
            tool_input: Input arguments for the tool.

        Raises:
            GovernanceError: If the tool invocation violates this policy.
        """
        if tool_name in self.blocked_tools:
            raise GovernanceError(
                "tool",
                f"Tool '{tool_name}' is blocked by policy.",
            )

        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            raise GovernanceError(
                "tool",
                f"Tool '{tool_name}' is not in the allowed tools list: "
                f"{self.allowed_tools}.",
            )

        if self.custom_validator is not None:
            if not self.custom_validator(tool_name, tool_input or {}):
                raise GovernanceError(
                    "tool",
                    f"Tool '{tool_name}' was rejected by custom validator.",
                )

        logger.debug("Tool governance: allowed tool '%s'", tool_name)


class GovernanceConfig(BaseModel):
    """Aggregated governance configuration for a CrewAI crew.

    Combines subprocess, HTTP, and tool policies into a single
    configuration object that can be attached to a Crew's SecurityConfig.

    Example:
        >>> governance = GovernanceConfig(
        ...     subprocess_policy=SubprocessPolicy(
        ...         allowed_commands=["docker", "git"],
        ...         blocked_commands=["rm"],
        ...     ),
        ...     http_policy=HttpPolicy(
        ...         allowed_domains=["api.openai.com"],
        ...     ),
        ...     tool_policy=ToolPolicy(
        ...         blocked_tools=["delete_database"],
        ...     ),
        ... )
        >>> crew = Crew(
        ...     security_config=SecurityConfig(governance=governance),
        ...     ...
        ... )
    """

    subprocess_policy: SubprocessPolicy = Field(
        default_factory=SubprocessPolicy,
        description="Policy for subprocess execution governance.",
    )
    http_policy: HttpPolicy = Field(
        default_factory=HttpPolicy,
        description="Policy for HTTP request governance.",
    )
    tool_policy: ToolPolicy = Field(
        default_factory=ToolPolicy,
        description="Policy for tool invocation governance.",
    )

    def validate_subprocess(
        self, command: list[str], *, shell: bool = False, **kwargs: Any
    ) -> None:
        """Validate a subprocess command.

        Args:
            command: The command as a list of strings.
            shell: Whether shell mode is requested.
            **kwargs: Additional subprocess keyword arguments.

        Raises:
            GovernanceError: If the command violates the subprocess policy.
        """
        self.subprocess_policy.validate_command(command, shell=shell, **kwargs)

    def validate_http(
        self, url: str, method: str = "GET", **kwargs: Any
    ) -> None:
        """Validate an HTTP request.

        Args:
            url: The target URL.
            method: HTTP method.
            **kwargs: Additional request keyword arguments.

        Raises:
            GovernanceError: If the request violates the HTTP policy.
        """
        self.http_policy.validate_request(url, method, **kwargs)

    def validate_tool(
        self, tool_name: str, tool_input: dict[str, Any] | None = None
    ) -> None:
        """Validate a tool invocation.

        Args:
            tool_name: Name of the tool.
            tool_input: Input arguments for the tool.

        Raises:
            GovernanceError: If the tool invocation violates the tool policy.
        """
        self.tool_policy.validate_tool(tool_name, tool_input)
