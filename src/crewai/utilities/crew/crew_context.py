"""Context management utilities for tracking crew and task execution context."""

from contextvars import ContextVar
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Optional, Callable, TypeVar, ParamSpec, Any, Generator, Union

from pydantic import BaseModel, UUID4, Field


class CrewContext(BaseModel):
    """Model representing crew context information."""

    id: Optional[UUID4] = Field(
        default=None, description="Unique identifier for the crew"
    )
    key: Optional[str] = Field(
        default=None, description="Optional crew key/name for identification"
    )


_crew_context: ContextVar[Optional[CrewContext]] = ContextVar(
    "crew_context", default=None
)


@contextmanager
def crew_context(
    crew_id: Optional[UUID4] = None, crew_key: Optional[str] = None
) -> Generator[Union[CrewContext, nullcontext], None, None]:
    """Context manager to track crew execution context.

    This context manager sets crew information that can be accessed by
    any component running within the crew's execution, including guardrails,
    tasks, and agents.

    Args:
        crew_id: The unique identifier for the crew
        crew_key: Optional crew key/name

    Example:
        with crew_context(crew_id="crew-123", crew_key="research-crew"):
            # Any code executed here can access crew context
            task.execute()
    """
    if crew_id is None and crew_key is None:
        yield nullcontext()
        return

    crew = CrewContext(id=crew_id, key=crew_key)
    token = _crew_context.set(crew)
    try:
        yield crew
    finally:
        _crew_context.reset(token)


R = TypeVar("R")
P = ParamSpec("P")


def with_crew_context(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that sets crew context for the duration of the method execution.

    This decorator automatically extracts crew_id and crew_key from the instance
    and sets them in the context for the entire method execution.

    Args:
        func: The method to decorate (should be a method of a class with 'id' and 'key' attributes)

    Returns:
        The decorated function with crew context set

    Example:
        class Crew:
            @with_crew_context
            def kickoff(self, inputs=None):
                # crew context is automatically set here
                ...
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Wrapper function that sets crew context before calling the decorated function.

        Args:
            *args: Positional arguments passed to the decorated function.
            **kwargs: Keyword arguments passed to the decorated function.

        Returns:
            The return value of the decorated function.
        """
        self: Any = args[0]
        if not hasattr(self, "id") and not hasattr(self, "key"):
            return func(*args, **kwargs)
        crew_id = getattr(self, "id", None)
        crew_key = getattr(self, "key", None)

        with crew_context(crew_id=crew_id, crew_key=crew_key):
            return func(*args, **kwargs)

    return wrapper


def get_crew_context() -> Optional[CrewContext]:
    """Get the current crew context.

    Returns:
        CrewContext instance containing crew context information, or None if no context is set
    """
    return _crew_context.get()
