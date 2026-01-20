"""Event context management for parent-child relationship tracking."""

from collections.abc import Generator
from contextlib import contextmanager
import contextvars


_event_id_stack: contextvars.ContextVar[tuple[tuple[str, str], ...]] = (
    contextvars.ContextVar("_event_id_stack", default=())
)


def get_current_parent_id() -> str | None:
    """Get the current parent event ID from the stack.

    Returns:
        The top event ID if stack is non-empty, otherwise None.
    """
    stack = _event_id_stack.get()
    return stack[-1][0] if stack else None


def get_enclosing_parent_id() -> str | None:
    """Get the parent of the current scope (stack[-2]).

    Used by ending events to become siblings of their matching started events.

    Returns:
        The second-to-top event ID, or None if stack has fewer than 2 items.
    """
    stack = _event_id_stack.get()
    return stack[-2][0] if len(stack) >= 2 else None


def push_event_scope(event_id: str, event_type: str = "") -> None:
    """Push an event ID and type onto the scope stack.

    Args:
        event_id: The event ID to push.
        event_type: The event type name (for pairing validation).
    """
    stack = _event_id_stack.get()
    _event_id_stack.set((*stack, (event_id, event_type)))


def pop_event_scope() -> tuple[str, str] | None:
    """Pop an event entry from the scope stack.

    Returns:
        Tuple of (event_id, event_type), or None if stack was empty.
    """
    stack = _event_id_stack.get()
    if not stack:
        return None
    _event_id_stack.set(stack[:-1])
    return stack[-1]


@contextmanager
def event_scope(event_id: str, event_type: str = "") -> Generator[None, None, None]:
    """Context manager to establish a parent event scope.

    Safe to use alongside emit() auto-management. If the event_id is already
    on the stack (e.g., from a starting event's auto-push), this will not
    double-push or double-pop.

    Args:
        event_id: The event ID to set as the current parent.
        event_type: The event type name (for pairing validation).
    """
    stack = _event_id_stack.get()
    already_on_stack = any(entry[0] == event_id for entry in stack)
    if not already_on_stack:
        push_event_scope(event_id, event_type)
    try:
        yield
    finally:
        if not already_on_stack:
            pop_event_scope()


SCOPE_STARTING_EVENTS: frozenset[str] = frozenset(
    {
        "flow_started",
        "crew_kickoff_started",
        "agent_execution_started",
        "task_started",
        "llm_call_started",
        "tool_usage_started",
        "memory_retrieval_started",
        "memory_save_started",
        "memory_query_started",
        "a2a_delegation_started",
        "a2a_conversation_started",
        "agent_reasoning_started",
    }
)

SCOPE_ENDING_EVENTS: frozenset[str] = frozenset(
    {
        "flow_finished",
        "flow_paused",
        "crew_kickoff_completed",
        "crew_kickoff_failed",
        "agent_execution_completed",
        "agent_execution_error",
        "task_completed",
        "task_failed",
        "llm_call_completed",
        "llm_call_failed",
        "tool_usage_finished",
        "tool_usage_error",
        "memory_retrieval_completed",
        "memory_save_completed",
        "memory_save_failed",
        "memory_query_completed",
        "memory_query_failed",
        "a2a_delegation_completed",
        "a2a_conversation_completed",
        "agent_reasoning_completed",
        "agent_reasoning_failed",
    }
)

VALID_EVENT_PAIRS: dict[str, str] = {
    "flow_finished": "flow_started",
    "flow_paused": "flow_started",
    "crew_kickoff_completed": "crew_kickoff_started",
    "crew_kickoff_failed": "crew_kickoff_started",
    "agent_execution_completed": "agent_execution_started",
    "agent_execution_error": "agent_execution_started",
    "task_completed": "task_started",
    "task_failed": "task_started",
    "llm_call_completed": "llm_call_started",
    "llm_call_failed": "llm_call_started",
    "tool_usage_finished": "tool_usage_started",
    "tool_usage_error": "tool_usage_started",
    "memory_retrieval_completed": "memory_retrieval_started",
    "memory_save_completed": "memory_save_started",
    "memory_save_failed": "memory_save_started",
    "memory_query_completed": "memory_query_started",
    "memory_query_failed": "memory_query_started",
    "a2a_delegation_completed": "a2a_delegation_started",
    "a2a_conversation_completed": "a2a_conversation_started",
    "agent_reasoning_completed": "agent_reasoning_started",
    "agent_reasoning_failed": "agent_reasoning_started",
}
