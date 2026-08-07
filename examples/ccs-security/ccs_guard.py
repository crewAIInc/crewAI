"""CCS Security Guard for CrewAI agent tool execution.

Integrates CCS runtime verification into CrewAI's tool calling pipeline,
providing security verification before any tool executes.

Usage:
    from examples.ccs_security.ccs_guard import CCSSecurityMiddleware

    # Wrap any CrewAI tool with CCS verification
    secured_tool = CCSSecurityMiddleware.wrap_tool(my_tool)
"""
import logging
from typing import Any, Callable, Optional

try:
    from ccs_verifier import Verifier, Command
    from ccs_verifier.builtin_rules import RCERule, SSRFRule, CredentialLeakRule
    CCS_AVAILABLE = True
except ImportError:
    CCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CCSSecurityMiddleware:
    """CCS security layer for CrewAI tool execution.

    Provides sub-millisecond runtime verification (~7.5μs P50) for all
    tool invocations in a CrewAI agent workflow.

    Detects and blocks:
    - Remote Code Execution (RCE) attempts
    - Server-Side Request Forgery (SSRF)
    - Credential/secret leakage in tool arguments
    """

    _instance = None
    _verifier = None

    @classmethod
    def get_verifier(cls) -> "Optional[Verifier]":
        """Lazy-initialize CCS verifier."""
        if not CCS_AVAILABLE:
            logger.warning("ccs-verifier not installed. Install: pip install ccs-verifier")
            return None
        if cls._verifier is None:
            cls._verifier = Verifier(rules=[RCERule(), SSRFRule(), CredentialLeakRule()])
        return cls._verifier

    @classmethod
    def verify_tool_call(cls, tool_name: str, arguments: dict, agent_id: str = "crewai-agent") -> tuple[bool, str]:
        """Verify a tool call through CCS before execution.

        Args:
            tool_name: Name of the tool being called.
            arguments: Tool call arguments.
            agent_id: Agent identifier for audit trail.

        Returns:
            (allowed, reason) tuple.
        """
        verifier = cls.get_verifier()
        if verifier is None:
            return True, "CCS not available"

        cmd = Command(
            agent_id=agent_id,
            tool=tool_name,
            params=arguments or {},
        )
        result = verifier.verify(cmd)
        if result.verdict.value == "deny":
            reason = getattr(result, "reason", "policy violation") or "policy violation"
            logger.warning(f"[CCS] Tool call denied: {tool_name}({arguments}) | reason={reason}")
            return False, reason
        return True, "allowed"

    @classmethod
    def wrap_tool(cls, tool, agent_id: str = "crewai-agent"):
        """Wrap a CrewAI tool with CCS security verification.

        Args:
            tool: CrewAI tool instance to wrap.
            agent_id: Agent identifier for audit.

        Returns:
            Wrapped tool with CCS verification.
        """
        original_run = tool.run if hasattr(tool, "run") else None
        if original_run is None:
            return tool

        def secured_run(*args, **kwargs):
            tool_name = getattr(tool, "name", type(tool).__name__)
            arguments = kwargs.get("arguments", kwargs)
            allowed, reason = cls.verify_tool_call(tool_name, arguments, agent_id)
            if not allowed:
                return f"[CCS Security] Tool call blocked: {reason}"
            return original_run(*args, **kwargs)

        tool.run = secured_run
        return tool
