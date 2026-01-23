"""A2A protocol error types."""

from a2a.client.errors import A2AClientTimeoutError


class A2APollingTimeoutError(A2AClientTimeoutError):
    """Raised when polling exceeds the configured timeout."""
