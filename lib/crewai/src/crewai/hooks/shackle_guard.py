"""
SHACKLE Guard Integration for crewAI
====================================
Pre-execution circuit breaker that plugs into crewAI's tool hook system.

Integration: One line to activate.
  from crewai.hooks.shackle_guard import register_shackle_guard
  register_shackle_guard(budget=0.25, max_repeat_calls=3)

How it works:
  - Registers as a before_tool_call hook
  - Tracks budget consumption across all tool calls
  - Detects repeat calls (loop of death)
  - Error amplification: tightens limits on 401/403/500 signals
  - HITL: uses crewAI's built-in human_input for approval gates

Does NOT require the standalone SHACKLE package.
This is a lightweight, self-contained integration for crewAI users.
"""

from __future__ import annotations

import time
from typing import Any

from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    register_before_tool_call_hook,
)


class ShackleGuard:
    """Pre-execution circuit breaker for crewAI tool calls.

    Tracks budget consumption, detects runaway loops, and provides
    human-in-the-loop approval for high-risk tool executions.

    Attributes:
        budget: Maximum cumulative cost in USD
        max_repeat_calls: Max identical tool+params calls before blocking
        error_amplification: Tighten limits when error signals detected
        timeout_seconds: Wall-clock timeout for the entire session
    """

    def __init__(
        self,
        budget: float = 0.25,
        max_repeat_calls: int = 3,
        error_amplification: bool = True,
        timeout_seconds: int = 300,
        hitl_tools: list[str] | None = None,
    ) -> None:
        self.budget = budget
        self.max_repeat_calls = max_repeat_calls
        self.error_amplification = error_amplification
        self.timeout_seconds = timeout_seconds
        self.hitl_tools = hitl_tools or [
            "execute_code", "write_file", "delete_file",
            "run_shell", "deploy", "terraform",
        ]

        # Runtime state
        self._budget_spent: float = 0.0
        self._total_calls: int = 0
        self._repeat_counts: dict[str, int] = {}
        self._last_tool_name: str = ""
        self._last_tool_input_hash: int = 0
        self._circuit_tripped: bool = False
        self._circuit_reason: str = ""
        self._start_time: float = time.time()

        # Error signals that trigger amplification
        self._error_signals = (
            "401", "unauthorized", "403", "forbidden", "500",
            "internal server error", "502", "bad gateway", "503",
            "service unavailable", "504", "gateway timeout", "timeout",
            "connection refused", "permission denied", "rate limit",
            "quota exceeded", "invalid api key", "token expired",
        )

    def _hash_input(self, tool_input: dict[str, Any]) -> int:
        """Simple hash of tool input for repeat detection."""
        return hash(str(sorted(tool_input.items())))

    def _detect_error(self, tool_input: dict[str, Any]) -> bool:
        """Check if tool input contains error signals."""
        input_str = str(tool_input).lower()
        return any(signal in input_str for signal in self._error_signals)

    def _cost_estimate(self, context: ToolCallHookContext) -> float:
        """Estimate cost of a tool call based on tool type."""
        cost_map = {
            "web_search": 0.001,
            "read_file": 0.0001,
            "write_file": 0.0005,
            "execute_code": 0.005,
            "query_db": 0.002,
            "call_api": 0.003,
            "send_email": 0.001,
            "create_agent": 0.01,
        }
        return cost_map.get(context.tool_name, 0.001)

    def __call__(self, context: ToolCallHookContext) -> bool | None:
        """Hook function called before every tool execution.

        Returns:
            False to block execution (SHACKLE DENY)
            None to trigger HITL approval (SHACKLE HITL)
            True or None to allow (SHACKLE ALLOW — default pass-through)
        """
        # Layer 1: Circuit breaker
        if self._circuit_tripped:
            print(
                f"\n⛓️ SHACKLE CIRCUIT OPEN: {self._circuit_reason}\n"
                f"   All tool calls blocked for this session."
            )
            return False

        # Layer 2: Timeout
        elapsed = time.time() - self._start_time
        if elapsed > self.timeout_seconds:
            self._circuit_tripped = True
            self._circuit_reason = f"Session timeout ({self.timeout_seconds}s)"
            print(
                f"\n⛓️ SHACKLE TIMEOUT: Session exceeded {self.timeout_seconds}s\n"
                f"   Circuit opened. All further calls blocked."
            )
            return False

        # Layer 3: Budget
        cost = self._cost_estimate(context)
        remaining = self.budget - self._budget_spent
        if remaining <= 0:
            self._circuit_tripped = True
            self._circuit_reason = (
                f"Budget exhausted: ${self._budget_spent:.4f} / ${self.budget:.2f}"
            )
            print(
                f"\n💰 SHACKLE BUDGET EXHAUSTED: "
                f"${self._budget_spent:.4f} / ${self.budget:.2f}\n"
                f"   Circuit opened. All further calls blocked."
            )
            return False

        # Layer 4: Repeat call detection
        call_hash = self._hash_input(context.tool_input)
        is_repeat = (
            context.tool_name == self._last_tool_name
            and call_hash == self._last_tool_input_hash
        )
        if is_repeat:
            key = context.tool_name
            self._repeat_counts[key] = self._repeat_counts.get(key, 0) + 1
            limit = self.max_repeat_calls

            # Error amplification: tighten limit when error signals detected
            if self.error_amplification and self._detect_error(context.tool_input):
                limit = max(1, self.max_repeat_calls - 1)

            if self._repeat_counts[key] >= limit:
                print(
                    f"\n🔁 SHACKLE LOOP DETECTED: '{context.tool_name}' "
                    f"called {self._repeat_counts[key]}x with same input\n"
                    f"   Limit: {self.max_repeat_calls}. Call blocked."
                )
                return False
        else:
            self._repeat_counts[context.tool_name] = 1

        # Layer 5: HITL for high-risk tools
        if context.tool_name in self.hitl_tools:
            response = context.request_human_input(
                prompt=(
                    f"\n🛑 SHACKLE HITL: High-risk tool '{context.tool_name}'\n"
                    f"   Budget remaining: ${remaining:.4f} / ${self.budget:.2f}\n"
                    f"   Input: {str(context.tool_input)[:100]}\n"
                    f"   Allow this execution?"
                ),
                default_message="Type 'approve' to allow, or press Enter to block:",
            )
            if response.lower() != "approve":
                print(f"   Blocked by human operator.")
                return False

        # Update state for next call
        self._budget_spent += cost
        self._total_calls += 1
        self._last_tool_name = context.tool_name
        self._last_tool_input_hash = call_hash

        return None  # Allow execution


# ── Public API ──

def register_shackle_guard(
    budget: float = 0.25,
    max_repeat_calls: int = 3,
    error_amplification: bool = True,
    timeout_seconds: int = 300,
    hitl_tools: list[str] | None = None,
) -> ShackleGuard:
    """Register SHACKLE as a global before_tool_call hook in crewAI.

    One-line activation. Blocks runaway tool calls BEFORE execution.

    Args:
        budget: Maximum cumulative cost in USD before circuit opens
        max_repeat_calls: Max identical tool+params calls before blocking
        error_amplification: Tighten limits on 401/403/500 error signals
        timeout_seconds: Wall-clock timeout for entire session
        hitl_tools: Tool names that require human approval

    Returns:
        ShackleGuard instance (can be used to query state)

    Example:
        >>> from crewai.hooks.shackle_guard import register_shackle_guard
        >>> shackle = register_shackle_guard(
        ...     budget=0.25,
        ...     max_repeat_calls=3,
        ...     hitl_tools=["execute_code", "deploy"],
        ... )
        >>> # Your crew runs with SHACKLE protection now
        >>> print(f"Budget spent: ${shackle._budget_spent:.4f}")
    """
    guard = ShackleGuard(
        budget=budget,
        max_repeat_calls=max_repeat_calls,
        error_amplification=error_amplification,
        timeout_seconds=timeout_seconds,
        hitl_tools=hitl_tools,
    )
    register_before_tool_call_hook(guard)
    return guard

# ── FAIL-CLOSED WRAPPER ──
# Override __call__ with fail-closed version
_original_call = ShackleGuard.__call__

def _fail_closed_call(self, context):
    """Wrapped call that fails closed — any guard error DENIES execution."""
    try:
        return _original_call(self, context)
    except Exception as e:
        self._circuit_tripped = True
        self._circuit_reason = f"Guard error (fail-closed): {e}"
        print(f"\n⛓️ SHACKLE FAIL-CLOSED: {e}\n   Circuit opened for safety.")
        return False

ShackleGuard.__call__ = _fail_closed_call
