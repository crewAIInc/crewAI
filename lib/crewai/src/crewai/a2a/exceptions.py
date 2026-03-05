"""A2A-specific exceptions."""

from __future__ import annotations


class A2AError(Exception):
    """Base exception for A2A-related errors."""

    pass


class A2AConversationMaxTurnsExceeded(A2AError):
    """Raised when A2A conversation exceeds maximum turns."""

    def __init__(self, max_turns: int, message: str | None = None):
        self.max_turns = max_turns
        super().__init__(
            message or f"A2A conversation exceeded maximum turns ({max_turns})"
        )


class A2AAgentCardFetchError(A2AError):
    """Raised when fetching an agent card fails."""

    def __init__(self, agent_name: str, reason: str):
        self.agent_name = agent_name
        self.reason = reason
        super().__init__(f"Failed to fetch agent card for '{agent_name}': {reason}")


class A2ADelegationError(A2AError):
    """Raised when A2A delegation fails."""

    def __init__(self, message: str, endpoint: str | None = None):
        self.endpoint = endpoint
        super().__init__(message)


class A2AResponseValidationError(A2AError):
    """Raised when A2A response validation fails."""

    def __init__(self, message: str, response_data: dict | None = None):
        self.response_data = response_data
        super().__init__(message)
