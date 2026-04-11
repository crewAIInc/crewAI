"""A2A conversation-level exceptions.

Note: For A2A protocol errors (JSON-RPC), see crewai.a2a.errors.A2AError.
"""

from __future__ import annotations


class A2AConversationMaxTurnsExceeded(Exception):
    """Raised when A2A conversation exceeds maximum turns."""

    def __init__(self, max_turns: int, message: str | None = None):
        self.max_turns = max_turns
        super().__init__(
            message or f"A2A conversation exceeded maximum turns ({max_turns})"
        )
