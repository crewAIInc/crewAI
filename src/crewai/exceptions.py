"""Exceptions for CrewAI."""

class AgentLookupError(Exception):
    """Exception raised when an agent cannot be found."""
    pass


class UnauthorizedDelegationError(Exception):
    """Exception raised when an agent attempts unauthorized delegation."""
    pass
