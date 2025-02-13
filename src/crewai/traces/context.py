from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator


class TraceContext:
    """Maintains the current trace context throughout the execution stack.

    This class provides a context manager for tracking trace execution across
    async and sync code paths using ContextVars.
    """

    _context: ContextVar = ContextVar("trace_context", default=None)

    @classmethod
    def get_current(cls):
        """Get the current trace context.

        Returns:
            Optional[UnifiedTraceController]: The current trace controller or None if not set.
        """
        return cls._context.get()

    @classmethod
    @contextmanager
    def set_current(cls, trace):
        """Set the current trace context within a context manager.

        Args:
            trace: The trace controller to set as current.

        Yields:
            UnifiedTraceController: The current trace controller.
        """
        token = cls._context.set(trace)
        try:
            yield trace
        finally:
            cls._context.reset(token)
