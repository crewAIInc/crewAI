"""Regression tests for EvaluationTraceCallback listener registration."""

from __future__ import annotations

from datetime import datetime

from crewai.agent import Agent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.events.types.tool_usage_events import (
    ToolExecutionErrorEvent,
    ToolUsageFinishedEvent,
)
from crewai.experimental.evaluation.evaluation_listener import (
    EvaluationTraceCallback,
    create_evaluation_callbacks,
)
from crewai.task import Task


def _make_agent_and_task() -> tuple[Agent, Task]:
    agent = Agent(
        role="Researcher",
        goal="Find facts",
        backstory="Experienced researcher",
    )
    task = Task(
        description="Verify deduplication",
        expected_output="Verified summary",
        agent=agent,
    )
    return agent, task


def test_create_evaluation_callbacks_does_not_duplicate_trace_entries() -> None:
    """Single and repeated create_evaluation_callbacks() calls attach each listener once."""
    with crewai_event_bus.scoped_handlers():
        EvaluationTraceCallback._instance = None
        EvaluationTraceCallback._initialized = False

        callback = create_evaluation_callbacks()
        second = create_evaluation_callbacks()
        assert second is callback

        agent, task = _make_agent_and_task()
        crewai_event_bus.emit(
            agent,
            AgentExecutionStartedEvent(
                agent=agent,
                task=task,
                tools=[],
                task_prompt="Verify deduplication",
            ),
        )

        now = datetime.now()
        crewai_event_bus.emit(
            agent,
            ToolUsageFinishedEvent(
                tool_name="search",
                tool_args={"query": "crewai"},
                output="ok",
                started_at=now,
                finished_at=now,
            ),
        )
        crewai_event_bus.emit(
            agent,
            ToolExecutionErrorEvent(
                tool_name="search",
                tool_args={"query": "bad"},
                tool_class=object,
                error="boom",
            ),
        )
        crewai_event_bus.emit(
            agent,
            LLMCallStartedEvent(
                call_id="call-1",
                messages=[{"role": "user", "content": "hi"}],
            ),
        )
        crewai_event_bus.emit(
            agent,
            LLMCallCompletedEvent(
                call_id="call-1",
                call_type=LLMCallType.LLM_CALL,
                messages=[{"role": "user", "content": "hi"}],
                response="hello",
            ),
        )
        crewai_event_bus.emit(
            agent,
            AgentExecutionCompletedEvent(agent=agent, task=task, output="done"),
        )
        crewai_event_bus.flush()

        trace = callback.get_trace(str(agent.id), str(task.id))
        assert trace is not None
        assert len(trace["tool_uses"]) == 2
        assert len(trace["llm_calls"]) == 1
        assert trace["final_output"] == "done"


def test_scoped_handlers_can_reregister_singleton_callback() -> None:
    """Exiting scoped_handlers clears handlers, and a later scope still attaches once."""
    with crewai_event_bus.scoped_handlers():
        EvaluationTraceCallback._instance = None
        EvaluationTraceCallback._initialized = False
        callback = create_evaluation_callbacks()
        assert isinstance(callback, EvaluationTraceCallback)

    with crewai_event_bus.scoped_handlers():
        callback = create_evaluation_callbacks()
        agent, task = _make_agent_and_task()
        callback.on_agent_start(agent, task)
        now = datetime.now()
        crewai_event_bus.emit(
            agent,
            ToolUsageFinishedEvent(
                tool_name="write",
                tool_args={},
                output="ok",
                started_at=now,
                finished_at=now,
            ),
        )
        crewai_event_bus.flush()

        trace = callback.get_trace(str(agent.id), str(task.id))
        assert trace is not None
        assert len(trace["tool_uses"]) == 1
