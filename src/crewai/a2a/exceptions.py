"""Custom exceptions for A2A integration."""


class A2AServerError(Exception):
    """Base exception for A2A server errors."""
    pass


class TransportError(A2AServerError):
    """Error related to transport configuration."""
    pass


class ExecutionError(A2AServerError):
    """Error during crew execution."""
    pass
