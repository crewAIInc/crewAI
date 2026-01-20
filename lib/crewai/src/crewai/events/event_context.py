"""Event context management for parent-child relationship tracking."""

from collections.abc import Generator
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
from enum import Enum

from crewai.events.utils.console_formatter import ConsoleFormatter


class MismatchBehavior(Enum):
    """Behavior when event pairs don't match."""

    WARN = "warn"
    RAISE = "raise"
    SILENT = "silent"


@dataclass
class EventContextConfig:
    """Configuration for event context behavior."""

    max_stack_depth: int = 100
    mismatch_behavior: MismatchBehavior = MismatchBehavior.WARN
    empty_pop_behavior: MismatchBehavior = MismatchBehavior.WARN


class StackDepthExceededError(Exception):
    """Raised when stack depth limit is exceeded."""


class EventPairingError(Exception):
    """Raised when event pairs don't match."""


class EmptyStackError(Exception):
    """Raised when popping from empty stack."""


_event_id_stack: contextvars.ContextVar[tuple[tuple[str, str], ...]] = (
    contextvars.ContextVar("_event_id_stack", default=())
)

_event_context_config: contextvars.ContextVar[EventContextConfig | None] = (
    contextvars.ContextVar("_event_context_config", default=None)
)

_default_config = EventContextConfig()

_console = ConsoleFormatter()


def get_current_parent_id() -> str | None:
    """Get the current parent event ID from the stack."""
    stack = _event_id_stack.get()
    return stack[-1][0] if stack else None


def get_enclosing_parent_id() -> str | None:
    """Get the parent of the current scope (stack[-2])."""
    stack = _event_id_stack.get()
    return stack[-2][0] if len(stack) >= 2 else None


def push_event_scope(event_id: str, event_type: str = "") -> None:
    """Push an event ID and type onto the scope stack."""
    config = _event_context_config.get() or _default_config
    stack = _event_id_stack.get()

    if config.max_stack_depth > 0 and len(stack) >= config.max_stack_depth:
        raise StackDepthExceededError(
            f"Event stack depth limit ({config.max_stack_depth}) exceeded. "
            f"This usually indicates missing ending events."
        )

    _event_id_stack.set((*stack, (event_id, event_type)))


def pop_event_scope() -> tuple[str, str] | None:
    """Pop an event entry from the scope stack."""
    stack = _event_id_stack.get()
    if not stack:
        return None
    _event_id_stack.set(stack[:-1])
    return stack[-1]


def handle_empty_pop(event_type_name: str) -> None:
    """Handle a pop attempt on an empty stack."""
    config = _event_context_config.get() or _default_config
    msg = (
        f"Ending event '{event_type_name}' emitted with empty scope stack. "
        "Missing starting event?"
    )

    if config.empty_pop_behavior == MismatchBehavior.RAISE:
        raise EmptyStackError(msg)
    if config.empty_pop_behavior == MismatchBehavior.WARN:
        _console.print(f"[CrewAIEventsBus] Warning: {msg}")


def handle_mismatch(
    event_type_name: str,
    popped_type: str,
    expected_start: str,
) -> None:
    """Handle a mismatched event pair."""
    config = _event_context_config.get() or _default_config
    msg = (
        f"Event pairing mismatch. '{event_type_name}' closed '{popped_type}' "
        f"(expected '{expected_start}')"
    )

    if config.mismatch_behavior == MismatchBehavior.RAISE:
        raise EventPairingError(msg)
    if config.mismatch_behavior == MismatchBehavior.WARN:
        _console.print(f"[CrewAIEventsBus] Warning: {msg}")


@contextmanager
def event_scope(event_id: str, event_type: str = "") -> Generator[None, None, None]:
    """Context manager to establish a parent event scope."""
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
        "method_execution_started",
        "crew_kickoff_started",
        "agent_execution_started",
        "lite_agent_execution_started",
        "task_started",
        "llm_call_started",
        "llm_guardrail_started",
        "tool_usage_started",
        "memory_retrieval_started",
        "memory_save_started",
        "memory_query_started",
        "knowledge_retrieval_started",
        "knowledge_query_started",
        "a2a_delegation_started",
        "a2a_conversation_started",
        "a2a_polling_started",
        "a2a_streaming_started",
        "a2a_server_task_started",
        "a2a_parallel_delegation_started",
        "agent_reasoning_started",
    }
)

SCOPE_ENDING_EVENTS: frozenset[str] = frozenset(
    {
        "flow_finished",
        "flow_paused",
        "method_execution_finished",
        "method_execution_failed",
        "crew_kickoff_completed",
        "crew_kickoff_failed",
        "agent_execution_completed",
        "agent_execution_error",
        "lite_agent_execution_completed",
        "lite_agent_execution_error",
        "task_completed",
        "task_failed",
        "llm_call_completed",
        "llm_call_failed",
        "llm_guardrail_completed",
        "tool_usage_finished",
        "tool_usage_error",
        "memory_retrieval_completed",
        "memory_save_completed",
        "memory_save_failed",
        "memory_query_completed",
        "memory_query_failed",
        "knowledge_retrieval_completed",
        "knowledge_query_completed",
        "knowledge_query_failed",
        "a2a_delegation_completed",
        "a2a_conversation_completed",
        "a2a_polling_completed",
        "a2a_streaming_completed",
        "a2a_server_task_completed",
        "a2a_server_task_canceled",
        "a2a_server_task_failed",
        "a2a_parallel_delegation_completed",
        "agent_reasoning_completed",
        "agent_reasoning_failed",
    }
)

VALID_EVENT_PAIRS: dict[str, str] = {
    "flow_finished": "flow_started",
    "flow_paused": "flow_started",
    "method_execution_finished": "method_execution_started",
    "method_execution_failed": "method_execution_started",
    "crew_kickoff_completed": "crew_kickoff_started",
    "crew_kickoff_failed": "crew_kickoff_started",
    "agent_execution_completed": "agent_execution_started",
    "agent_execution_error": "agent_execution_started",
    "lite_agent_execution_completed": "lite_agent_execution_started",
    "lite_agent_execution_error": "lite_agent_execution_started",
    "task_completed": "task_started",
    "task_failed": "task_started",
    "llm_call_completed": "llm_call_started",
    "llm_call_failed": "llm_call_started",
    "llm_guardrail_completed": "llm_guardrail_started",
    "tool_usage_finished": "tool_usage_started",
    "tool_usage_error": "tool_usage_started",
    "memory_retrieval_completed": "memory_retrieval_started",
    "memory_save_completed": "memory_save_started",
    "memory_save_failed": "memory_save_started",
    "memory_query_completed": "memory_query_started",
    "memory_query_failed": "memory_query_started",
    "knowledge_retrieval_completed": "knowledge_retrieval_started",
    "knowledge_query_completed": "knowledge_query_started",
    "knowledge_query_failed": "knowledge_query_started",
    "a2a_delegation_completed": "a2a_delegation_started",
    "a2a_conversation_completed": "a2a_conversation_started",
    "a2a_polling_completed": "a2a_polling_started",
    "a2a_streaming_completed": "a2a_streaming_started",
    "a2a_server_task_completed": "a2a_server_task_started",
    "a2a_server_task_canceled": "a2a_server_task_started",
    "a2a_server_task_failed": "a2a_server_task_started",
    "a2a_parallel_delegation_completed": "a2a_parallel_delegation_started",
    "agent_reasoning_completed": "agent_reasoning_started",
    "agent_reasoning_failed": "agent_reasoning_started",
}
