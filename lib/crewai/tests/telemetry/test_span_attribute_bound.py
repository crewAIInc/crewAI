"""What a run produced reaches the exported span whole, up to one stated bound.

A graded run of a flow could not be judged because the span carried a cut copy
of a tool's result: a summary task had to report "the issue count and
source_issue_ids exactly match the Linear results", and the grader saw 4 KB of
a 33 KB result. Tool results, task outputs and LLM messages are what a reader
of the trace checks the run against, so they now arrive whole up to
``DEFAULT_MAX_ATTR_BYTES`` (sized to Wharf's request limit), and a value over
it is cut as little as possible and says so — never silently.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

from crewai import Agent, Crew, Task
from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry.tracing import gen_ai_shapes, handlers, semantic_conventions
from crewai.telemetry.tracing.context import TelemetryExecutionContext
from crewai.telemetry.tracing.grants import MAX_EXPORT_BODY_BYTES
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


BOUND = gen_ai_shapes.DEFAULT_MAX_ATTR_BYTES


def _text(size: int, label: str = "issue") -> str:
    """Non-repeating text of about ``size`` bytes: a cut copy cannot compare equal."""
    lines: list[str] = []
    total = 0
    i = 0
    while total < size:
        line = f'{{"id": "{label}-{i}", "title": "Linear issue number {i}"}}\n'
        lines.append(line)
        total += len(line)
        i += 1
    return "".join(lines)


@pytest.fixture(autouse=True)
def enable_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite otherwise runs with OTEL_SDK_DISABLED, which makes every
    assertion here pass vacuously against non-recording spans."""
    for name in (
        "OTEL_SDK_DISABLED",
        "CREWAI_DISABLE_TELEMETRY",
        "CREWAI_DISABLE_TRACKING",
        "CREWAI_OTEL_MAX_ATTR_BYTES",
        "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT",
        "OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)


class _Providers:
    def __init__(self, tracer) -> None:
        self._tracer = tracer

    def get_tracer(self, name: str | None = None):
        return self._tracer

    def emit_log(self, *args, **kwargs) -> None:
        pass


def _pipeline():
    """Built inside each test, after any env change: the SDK reads its span
    limits when the provider is constructed."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    ctx = TelemetryExecutionContext(
        kickoff_id="kickoff", automation_name="test", tracer=tracer
    )
    return provider, _Providers(tracer), ctx, exporter


def _only_span(exporter: InMemorySpanExporter, name: str):
    matches = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(matches) == 1, [s.name for s in exporter.get_finished_spans()]
    return matches[0]


def _tool_span(result: str):
    provider, providers, ctx, exporter = _pipeline()
    now = datetime.now(timezone.utc)
    args = {"query": "issues in cycle 42"}
    started = ToolUsageStartedEvent(
        tool_name="linear_run_query", tool_args=args, agent_key="k", agent_role="r"
    )
    handlers.handle_tool_usage_started(providers, ctx, None, started)
    handlers.handle_tool_usage_finished(
        providers,
        ctx,
        None,
        ToolUsageFinishedEvent(
            tool_name="linear_run_query",
            tool_args=args,
            agent_key="k",
            agent_role="r",
            started_at=now,
            finished_at=now,
            output=result,
            started_event_id=started.event_id,
        ),
    )
    span = _only_span(exporter, "call tool")
    provider.shutdown()
    return span


@pytest.mark.parametrize("size", [50_000, 300_000], ids=["50KB", "300KB"])
def test_a_tool_result_arrives_whole(size: int) -> None:
    result = _text(size)

    span = _tool_span(result)

    assert json.loads(span.attributes["gen_ai.tool.call.result"]) == result
    assert "gen_ai.tool.call.result.truncated" not in span.attributes


def test_a_tool_result_over_the_bound_is_cut_with_the_marker_and_its_size() -> None:
    result = _text(BOUND + 200_000)
    original = len(json.dumps(result).encode("utf-8"))

    span = _tool_span(result)

    payload = span.attributes["gen_ai.tool.call.result"]
    assert span.attributes["gen_ai.tool.call.result.truncated"] is True
    assert span.attributes["gen_ai.tool.call.result.original_size_bytes"] == original
    assert len(payload.encode("utf-8")) <= BOUND
    envelope = json.loads(payload)
    assert envelope["_truncated"] is True
    assert envelope["_original_size_bytes"] == original
    # Cut as little as possible: the preview fills the bound, it is not a
    # 4 KB sample of a value that was barely over it.
    assert len(payload.encode("utf-8")) > BOUND * 0.99
    assert json.dumps(result).startswith(envelope["_preview"])


def _task_span(raw: str):
    provider, providers, ctx, exporter = _pipeline()
    agent = Agent(role="Analyst", goal="g", backstory="b")
    task = Task(description="d", expected_output="e", agent=agent)
    agent.crew = Crew(agents=[agent], tasks=[task])
    started = TaskStartedEvent(context=None, task=task)
    handlers.handle_task_started(providers, ctx, task, started)
    handlers.handle_task_completed(
        providers,
        ctx,
        task,
        TaskCompletedEvent(
            output=TaskOutput(description="d", raw=raw, agent="Analyst"),
            task=task,
            started_event_id=started.event_id,
        ),
    )
    span = _only_span(exporter, "execute task")
    provider.shutdown()
    return span


def test_a_task_output_arrives_whole_under_both_keys() -> None:
    raw = _text(300_000, label="summary")

    span = _task_span(raw)

    assert span.attributes["crewai.task.output"] == raw
    assert "crewai.task.output.truncated" not in span.attributes
    messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert messages[0]["parts"][0]["content"] == raw
    assert "gen_ai.output.messages.truncated" not in span.attributes


def test_a_plain_attribute_over_the_bound_marks_itself_too() -> None:
    """``crewai.task.output`` used to have no bound at all: one huge output made
    the whole span too large for Wharf, which then never stored it."""
    raw = _text(BOUND + 50_000, label="summary")

    span = _task_span(raw)

    assert span.attributes["crewai.task.output.truncated"] is True
    assert span.attributes["crewai.task.output.original_size_bytes"] == len(
        raw.encode("utf-8")
    )
    assert len(span.attributes["crewai.task.output"].encode("utf-8")) <= BOUND
    assert span.attributes["gen_ai.output.messages.truncated"] is True


def test_llm_messages_and_response_arrive_whole() -> None:
    tool_result = _text(300_000)
    messages = [
        {"role": "system", "content": "You are an analyst."},
        {"role": "user", "content": "Summarise the cycle."},
        {"role": "tool", "content": tool_result},
    ]
    answer = _text(100_000, label="answer")

    attrs = semantic_conventions.gen_ai(
        input_messages=messages, output_messages=answer
    )

    shaped = json.loads(attrs["gen_ai.input.messages"])
    assert shaped[-1]["parts"][0]["content"] == tool_result
    assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"][0][
        "content"
    ] == answer
    assert not any(key.endswith(".truncated") for key in attrs)


def test_a_conversation_over_the_bound_keeps_its_ends_and_names_the_cut() -> None:
    """The repeated conversation on an LLM call is where a large tool result
    would otherwise be copied into every later call: past the bound its middle
    is replaced by a placeholder; the tool span keeps the result whole."""
    messages = [{"role": "system", "content": "You are an analyst."}]
    messages += [{"role": "tool", "content": _text(200_000, f"r{i}")} for i in range(3)]
    messages.append({"role": "user", "content": "Now write the summary."})

    attrs = semantic_conventions.gen_ai(input_messages=messages)

    payload = attrs["gen_ai.input.messages"]
    assert attrs["gen_ai.input.messages.truncated"] is True
    assert len(payload.encode("utf-8")) <= BOUND
    shaped = json.loads(payload)
    assert shaped[0]["parts"][0]["content"] == "You are an analyst."
    assert shaped[-1]["parts"][0]["content"] == "Now write the summary."
    assert "[truncated 3 messages" in shaped[1]["parts"][0]["content"]


def test_one_cut_message_keeps_all_but_the_overshoot() -> None:
    """A single message over the bound loses what is over, not half of itself."""
    content = _text(BOUND + 20_000)

    attrs = semantic_conventions.gen_ai(output_messages=content)

    payload = attrs["gen_ai.output.messages"]
    assert attrs["gen_ai.output.messages.truncated"] is True
    assert len(payload.encode("utf-8")) <= BOUND
    kept = json.loads(payload)[0]["parts"][0]["content"]
    assert "...[truncated " in kept
    assert len(payload.encode("utf-8")) > BOUND * 0.99


def test_an_sdk_length_limit_lowers_the_bound_so_the_sdk_never_cuts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenTelemetry SDK cuts a value over its limit with no marker. A user
    who set that limit gets it, but cut by crewAI, which says so."""
    limit = 20_000
    monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", str(limit))
    assert gen_ai_shapes.max_attr_bytes() == limit

    span = _tool_span(_text(50_000))

    payload = span.attributes["gen_ai.tool.call.result"]
    assert span.attributes["gen_ai.tool.call.result.truncated"] is True
    assert len(payload) <= limit
    # Still valid JSON: crewAI's envelope, not a string the SDK chopped.
    assert json.loads(payload)["_truncated"] is True


def test_the_span_limit_wins_over_the_general_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "10000")
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "30000")
    assert gen_ai_shapes.max_attr_bytes() == 30_000
    monkeypatch.setenv("CREWAI_OTEL_MAX_ATTR_BYTES", "5000")
    assert gen_ai_shapes.max_attr_bytes() == 5_000


def test_crewai_own_setting_still_replaces_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREWAI_OTEL_MAX_ATTR_BYTES", str(1024 * 1024))
    assert gen_ai_shapes.max_attr_bytes() == 1024 * 1024
    monkeypatch.setenv("CREWAI_OTEL_MAX_ATTR_BYTES", "not a number")
    assert gen_ai_shapes.max_attr_bytes() == BOUND


def test_a_span_with_every_content_attribute_at_the_bound_fits_one_wharf_request() -> (
    None
):
    """Wharf refuses a request over 3,072,000 encoded bytes and the exporter
    drops a span that alone is over it. An LLM call carries the most content
    attributes, so seven of them at the bound must still fit."""
    provider, providers, ctx, exporter = _pipeline()
    span = providers.get_tracer().start_span("call llm")
    big = _text(BOUND * 2)
    handlers._set_span_attributes(
        span,
        {
            **semantic_conventions.gen_ai(
                input_messages=[{"role": "user", "content": big}],
                output_messages=big,
                system_instructions=big,
                tool_definitions=[{"name": "t", "description": big}],
            ),
            "crewai.a": big,
            "crewai.b": big,
            "crewai.c": big,
        },
    )
    span.end()
    finished = exporter.get_finished_spans()
    provider.shutdown()

    assert sum(1 for key in finished[0].attributes if key.endswith(".truncated")) == 7
    assert encode_spans(finished).ByteSize() < MAX_EXPORT_BODY_BYTES


def test_a_role_bearing_plain_attribute_is_cut_plainly_never_reshaped() -> None:
    """A task may produce a JSON list whose items carry ``role`` (a roster, a
    chat log). On ``crewai.task.output`` that is data, not a GenAI
    conversation: over the bound it keeps its head, nothing is replaced."""
    roster = json.dumps(
        [
            {"role": "engineer", "name": f"person-{i}", "team": f"team-{i % 7}"}
            for i in range(12_000)
        ]
    )
    assert len(roster.encode("utf-8")) > BOUND

    span = _task_span(roster)

    kept = span.attributes["crewai.task.output"]
    assert span.attributes["crewai.task.output.truncated"] is True
    assert span.attributes["crewai.task.output.original_size_bytes"] == len(
        roster.encode("utf-8")
    )
    assert len(kept.encode("utf-8")) == BOUND
    assert roster.startswith(kept)
    assert "truncated" not in kept


def test_an_sdk_span_limit_of_zero_is_kept_and_cuts_everything_with_the_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK accepts 0 and cuts every string to nothing; crewAI does the
    same cut first, so it carries the marker."""
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "0")
    monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "20000")
    assert gen_ai_shapes.max_attr_bytes() == 0

    span = _task_span("a summary")

    assert span.attributes["crewai.task.output"] == ""
    assert span.attributes["crewai.task.output.truncated"] is True
    assert span.attributes["crewai.task.output.original_size_bytes"] == len(
        "a summary"
    )


def test_an_explicitly_empty_span_limit_is_unlimited_not_the_general_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In the SDK an empty span setting means unlimited and wins over the
    general one; the bound is then crewAI's own."""
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "")
    monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "20000")
    assert gen_ai_shapes.max_attr_bytes() == BOUND

    result = _text(50_000)
    span = _tool_span(result)

    assert json.loads(span.attributes["gen_ai.tool.call.result"]) == result
    assert "gen_ai.tool.call.result.truncated" not in span.attributes
