"""The execute task, execute agent and call tool spans carry what a reader needs.

A consumer of a run's spans could see a task's raw text but not the format it
declared, nor whether a Pydantic object or a JSON dict actually came out of it;
it could see an agent's goal, backstory and model but not the prompt the agent
was handed or the answer it gave; and it could see a tool's result but not
whether the tool ran or the cache answered. Each had to be reconstructed from
other spans, or could not be known at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

from crewai import Agent, Crew, Task
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry.tracing import handlers
from crewai.telemetry.tracing.context import TelemetryExecutionContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
import pytest


# Long and non-repeating, so a truncated or elided copy cannot compare equal.
# Under the default 32 KiB attribute cap, so it must arrive whole.
LONG_TEXT = "".join(
    f"paragraph {i}: the quick brown fox jumps over the lazy dog\n" for i in range(400)
)
assert 20_000 < len(LONG_TEXT.encode("utf-8")) < 32 * 1024


@pytest.fixture(autouse=True)
def enable_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite otherwise runs with OTEL_SDK_DISABLED, which makes every
    assertion here pass vacuously against non-recording spans."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TRACKING", raising=False)


class _Providers:
    """All a handler asks of its providers: a tracer and somewhere to log."""

    def __init__(self, tracer) -> None:
        self._tracer = tracer

    def get_tracer(self, name: str | None = None):
        return self._tracer

    def emit_log(self, *args, **kwargs) -> None:
        pass


@pytest.fixture
def pipeline():
    """Handler inputs whose spans land in memory, exported as each one ends."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    ctx = TelemetryExecutionContext(
        kickoff_id="kickoff", automation_name="test", tracer=tracer
    )
    # yield, not return: the frame keeps `provider` alive, and a collected
    # provider shuts down its processor and silently loses spans.
    yield _Providers(tracer), ctx, exporter


def _only_span(exporter: InMemorySpanExporter, name: str):
    """The single exported span with ``name``, failing loudly if not unique."""
    matches = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(matches) == 1, [s.name for s in exporter.get_finished_spans()]
    return matches[0]


class Finding(BaseModel):
    title: str


def _crewed_task(**declaration) -> Task:
    """A task whose agent belongs to a crew, as the task handlers require."""
    agent = Agent(role="Researcher", goal="g", backstory="b")
    task = Task(description="d", expected_output="e", agent=agent, **declaration)
    crew = Crew(agents=[agent], tasks=[task])
    # A kickoff attaches the crew to its agents; these events skip the kickoff.
    agent.crew = crew
    return task


@pytest.mark.parametrize(
    ("declaration", "produced", "expected"),
    [
        (
            {"output_pydantic": Finding},
            {"pydantic": Finding(title="t"), "output_format": OutputFormat.PYDANTIC},
            ("pydantic", True, False),
        ),
        (
            {"output_json": Finding},
            {"json_dict": {"title": "t"}, "output_format": OutputFormat.JSON},
            ("json", False, True),
        ),
        (
            {},
            {"output_format": OutputFormat.RAW},
            ("raw", False, False),
        ),
    ],
    ids=["pydantic", "json", "raw"],
)
def test_execute_task_span_says_what_was_declared_and_what_came_out(
    pipeline, declaration, produced, expected
) -> None:
    providers, ctx, exporter = pipeline
    task = _crewed_task(**declaration)
    output = TaskOutput(
        description="d", raw='{"title": "t"}', agent="Researcher", **produced
    )

    started = TaskStartedEvent(context=None, task=task)
    handlers.handle_task_started(providers, ctx, task, started)
    handlers.handle_task_completed(
        providers,
        ctx,
        task,
        TaskCompletedEvent(output=output, task=task, started_event_id=started.event_id),
    )

    span = _only_span(exporter, "execute task")
    output_format, pydantic_produced, json_produced = expected
    assert span.attributes["crewai.task.output_format"] == output_format
    assert span.attributes["crewai.task.output_pydantic_produced"] is pydantic_produced
    assert span.attributes["crewai.task.output_json_produced"] is json_produced
    # The raw text is still there, as before.
    assert span.attributes["crewai.task.output"] == '{"title": "t"}'


def test_a_declared_pydantic_task_that_produced_none_is_told_apart(pipeline) -> None:
    """The gap the two flags close: declared is not the same as produced."""
    providers, ctx, exporter = pipeline
    task = _crewed_task(output_pydantic=Finding)
    # The producer could not convert, so the output carries only raw text.
    output = TaskOutput(
        description="d",
        raw="not json at all",
        agent="Researcher",
        output_format=OutputFormat.PYDANTIC,
    )

    started = TaskStartedEvent(context=None, task=task)
    handlers.handle_task_started(providers, ctx, task, started)
    handlers.handle_task_completed(
        providers,
        ctx,
        task,
        TaskCompletedEvent(output=output, task=task, started_event_id=started.event_id),
    )

    span = _only_span(exporter, "execute task")
    assert span.attributes["crewai.task.output_format"] == "pydantic"
    assert span.attributes["crewai.task.output_pydantic_produced"] is False
    assert span.attributes["crewai.task.output_json_produced"] is False


def test_a_failed_task_still_says_what_it_declared(pipeline) -> None:
    """There is no output to inspect, so the format comes from the declaration."""
    providers, ctx, exporter = pipeline
    task = _crewed_task(output_pydantic=Finding)

    started = TaskStartedEvent(context=None, task=task)
    handlers.handle_task_started(providers, ctx, task, started)
    handlers.handle_task_failed(
        providers,
        ctx,
        task,
        TaskFailedEvent(error="boom", task=task, started_event_id=started.event_id),
    )

    span = _only_span(exporter, "execute task")
    assert span.attributes["crewai.task.output_format"] == "pydantic"
    assert "crewai.task.output_pydantic_produced" not in span.attributes
    assert "crewai.task.output_json_produced" not in span.attributes


def _agent_span(pipeline, *, prompt: str, output: str):
    providers, ctx, exporter = pipeline
    agent = Agent(role="Researcher", goal="g", backstory="b")

    started = AgentExecutionStartedEvent(
        agent=agent, task=None, tools=None, task_prompt=prompt
    )
    handlers.handle_agent_execution_started(providers, ctx, agent, started)
    # The completed handler waits (up to 5 s) for the agent's LLM-call count to
    # settle; one recorded call is what a real run would have signalled.
    handlers._record_agent_llm_call(ctx, str(agent.id))
    handlers.handle_agent_execution_completed(
        providers,
        ctx,
        agent,
        AgentExecutionCompletedEvent(
            agent=agent, task=None, output=output, started_event_id=started.event_id
        ),
    )
    return _only_span(exporter, "execute agent")


def test_execute_agent_span_carries_the_exact_prompt_and_answer(pipeline) -> None:
    span = _agent_span(pipeline, prompt=LONG_TEXT, output="The fox is quick.")

    # The same spec shape the task span already uses for its own text.
    prompt = json.loads(span.attributes["gen_ai.input.messages"])
    assert prompt[0]["parts"][0]["content"] == LONG_TEXT
    assert "gen_ai.input.messages.truncated" not in span.attributes
    answer = json.loads(span.attributes["gen_ai.output.messages"])
    assert answer[0]["parts"][0]["content"] == "The fox is quick."
    assert "gen_ai.output.messages.truncated" not in span.attributes


def test_an_over_cap_prompt_is_declared_truncated_never_silently_cut(
    pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = 2048
    monkeypatch.setenv("CREWAI_OTEL_MAX_ATTR_BYTES", str(cap))

    span = _agent_span(pipeline, prompt=LONG_TEXT, output="short")

    payload = span.attributes["gen_ai.input.messages"]
    assert span.attributes["gen_ai.input.messages.truncated"] is True
    assert span.attributes["gen_ai.input.messages.original_size_bytes"] > cap
    assert len(payload.encode("utf-8")) <= cap
    # The answer fit, so it arrives whole and unmarked.
    answer = json.loads(span.attributes["gen_ai.output.messages"])
    assert answer[0]["parts"][0]["content"] == "short"
    assert "gen_ai.output.messages.truncated" not in span.attributes


@pytest.mark.parametrize("from_cache", [True, False], ids=["cached", "ran"])
def test_call_tool_span_says_whether_the_cache_answered(pipeline, from_cache) -> None:
    providers, ctx, exporter = pipeline
    now = datetime.now(timezone.utc)

    started = ToolUsageStartedEvent(
        tool_name="search", tool_args={"q": "fox"}, agent_key="k", agent_role="r"
    )
    handlers.handle_tool_usage_started(providers, ctx, None, started)
    handlers.handle_tool_usage_finished(
        providers,
        ctx,
        None,
        ToolUsageFinishedEvent(
            tool_name="search",
            tool_args={"q": "fox"},
            agent_key="k",
            agent_role="r",
            started_at=now,
            finished_at=now,
            output="found",
            from_cache=from_cache,
            started_event_id=started.event_id,
        ),
    )

    span = _only_span(exporter, "call tool")
    assert span.attributes["crewai.tool.from_cache"] is from_cache
