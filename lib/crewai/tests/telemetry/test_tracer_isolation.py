"""Telemetry must export our spans and only our spans.

Regression cover for the collector receiving third-party application traces:
``set_tracer()`` used to install CrewAI's ``TracerProvider`` as the global one,
so every OTel-instrumented library in the host process - HTTP servers, Redis
clients, ORMs - resolved ``trace.get_tracer()`` to our provider and shipped its
spans to CrewAI's endpoint.
"""

from typing import Any
from unittest.mock import Mock, patch

import opentelemetry.trace as ot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import pytest

from crewai.telemetry.constants import TRACER_NAME
from crewai.telemetry.telemetry import Telemetry


class _NullExporter:
    """Stands in for the OTLP exporter so no test attempts a real export."""

    def export(self, spans: Any) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@pytest.fixture
def telemetry_with_exporter(monkeypatch):
    """A fresh Telemetry whose provider exports into memory.

    Telemetry is a process-wide singleton that registers atexit and signal
    handlers on init, so the instance is replaced for the duration of the test
    and lifecycle registration is suppressed.
    """
    monkeypatch.setattr(Telemetry, "_instance", None)
    monkeypatch.setattr(Telemetry, "_register_shutdown_handlers", lambda self: None)

    # Set for the whole test: _is_telemetry_disabled() is re-read on every span
    # call, and the suite runs with OTEL_SDK_DISABLED set.
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "false")
    monkeypatch.setenv("CREWAI_DISABLE_TRACKING", "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")

    # Patched before construction: __init__ wires the real OTLP exporter, which
    # would make every test here attempt a live export.
    monkeypatch.setattr(
        "crewai.telemetry.telemetry.SafeOTLPSpanExporter",
        lambda **_kwargs: _NullExporter(),
    )

    telemetry = Telemetry()

    exporter = InMemorySpanExporter()
    telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        yield telemetry, exporter
    finally:
        telemetry.provider.shutdown()
        Telemetry._instance = None


def test_third_party_spans_never_reach_our_exporter(telemetry_with_exporter):
    """A dependency instrumenting itself must not export to CrewAI."""
    telemetry, exporter = telemetry_with_exporter
    telemetry.set_tracer()

    ot.get_tracer("redis.client").start_span("XLEN").end()
    ot.get_tracer("opentelemetry.instrumentation.asgi").start_span(
        "GET /status http send"
    ).end()

    assert exporter.get_finished_spans() == ()


def test_our_own_spans_still_reach_our_exporter(telemetry_with_exporter):
    """The isolation must not cost us the telemetry we do want."""
    telemetry, exporter = telemetry_with_exporter
    telemetry.set_tracer()

    telemetry.feature_usage_span("cli_usage:view_traces")

    assert [span.name for span in exporter.get_finished_spans()] == ["Feature Usage"]


def test_a_feature_span_keeps_only_what_the_feature_may_send(telemetry_with_exporter):
    """The same policy as crewai-core's: each feature names what it may add, and
    the dimensions every span already carries cannot be overwritten."""
    telemetry, exporter = telemetry_with_exporter
    telemetry.set_tracer()
    offered = {
        "authenticated": "false",
        "execution_id": "run-1",
        "organization_id": "org-1",
        "feature": "someone_else",
    }

    telemetry.feature_usage_span("cli_usage:eval", offered)
    telemetry.feature_usage_span("cli_usage:deploy", offered)

    eval_span, deploy_span = exporter.get_finished_spans()
    assert eval_span.attributes["authenticated"] == "false"
    assert eval_span.attributes["feature"] == "cli_usage:eval"
    assert "execution_id" not in eval_span.attributes
    assert "organization_id" not in eval_span.attributes
    assert "authenticated" not in deploy_span.attributes
    assert deploy_span.attributes["feature"] == "cli_usage:deploy"


def test_the_models_span_carries_the_models_compared_and_nothing_else(
    telemetry_with_exporter,
):
    telemetry, exporter = telemetry_with_exporter
    telemetry.set_tracer()

    telemetry.feature_usage_span(
        "cli_usage:eval_models",
        {
            "authenticated": "true",
            "models": "openai/gpt-4o-mini,anthropic/claude-haiku-4-5",
            "models_count": "2",
            "evaluation_id": "ev-1",
            "organization_id": "org-1",
        },
    )

    (span,) = exporter.get_finished_spans()
    assert span.attributes["feature"] == "cli_usage:eval_models"
    assert span.attributes["authenticated"] == "true"
    assert span.attributes["models"] == "openai/gpt-4o-mini,anthropic/claude-haiku-4-5"
    assert span.attributes["models_count"] == "2"
    assert "evaluation_id" not in span.attributes
    assert "organization_id" not in span.attributes


def test_our_spans_are_unaffected_by_an_application_provider(telemetry_with_exporter):
    """An app that installs its own provider must not divert our telemetry.

    Resolving our tracer globally meant that in an already-instrumented
    application our spans were created by the application's provider and went
    to its collector, so CrewAI received nothing at all from those processes.
    """
    telemetry, exporter = telemetry_with_exporter

    app_exporter = InMemorySpanExporter()
    app_provider = TracerProvider()
    app_provider.add_span_processor(SimpleSpanProcessor(app_exporter))

    with patch.object(ot, "get_tracer_provider", return_value=app_provider):
        telemetry.set_tracer()
        telemetry.feature_usage_span("cli_usage:deploy")

    assert [span.name for span in exporter.get_finished_spans()] == ["Feature Usage"]
    assert app_exporter.get_finished_spans() == ()


def test_set_tracer_is_idempotent(telemetry_with_exporter):
    """Repeated calls must not stack processors or duplicate exports."""
    telemetry, exporter = telemetry_with_exporter

    telemetry.set_tracer()
    telemetry.set_tracer()
    telemetry.set_tracer()

    telemetry.provider.get_tracer(TRACER_NAME).start_span("Crew Created").end()

    assert len(exporter.get_finished_spans()) == 1


@pytest.fixture
def standalone_exporter(telemetry_with_exporter, monkeypatch):
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.event_listener import event_listener

    telemetry, exporter = telemetry_with_exporter
    telemetry.set_tracer()
    with crewai_event_bus.scoped_handlers():
        monkeypatch.setattr(event_listener, "_telemetry", telemetry)
        event_listener.setup_listeners(crewai_event_bus)
        try:
            yield exporter
        finally:
            crewai_event_bus.flush()


def _conversation_agent():
    from crewai import Agent, BaseLLM

    class ReplyLLM(BaseLLM):
        def supports_function_calling(self):
            return False

        def call(self, messages, **kwargs):
            return "Final Answer: Done"

        async def acall(self, messages, **kwargs):
            return self.call(messages, **kwargs)

    return Agent(
        role="private-agent-role",
        goal="Help",
        backstory="Helpful",
        llm=ReplyLLM(model="recording"),
        conversational=True,
        max_retry_limit=0,
    )


def _standalone_counts(exporter):
    from collections import Counter
    from crewai.events.event_bus import crewai_event_bus

    crewai_event_bus.flush()
    return Counter(
        span.attributes["feature"]
        for span in exporter.get_finished_spans()
        if span.name == "Feature Usage"
        and str(span.attributes.get("feature", "")).startswith("standalone_agent:")
    )


@pytest.mark.parametrize("mode", ["handle_turn", "repl", "tui", "managed_bridge"])
def test_standalone_usage_counts_entrypoints_and_each_completed_turn(
    standalone_exporter, monkeypatch, mode
):
    from crewai.agent.conversation import agent_flow

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = _conversation_agent()
    messages = iter(["private-message-908", "Second turn", "exit"])
    if mode == "handle_turn":
        agent.handle_turn(next(messages))
        agent.handle_turn(next(messages))
    elif mode == "managed_bridge":
        flow = agent_flow(lambda: agent)()
        flow.handle_turn(next(messages))
        flow.handle_turn(next(messages))
    elif mode == "repl":
        agent.chat(input_fn=lambda _: next(messages), output_fn=lambda _: None)
    else:

        def run_tui(flow, **kwargs):
            for message in ("private-message-908", "Second turn"):
                with flow.stream_turn(message, stream=None) as turn:
                    list(turn)
                    assert turn.result.raw == "Done"

        monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
        monkeypatch.setattr(
            "crewai_cli.kickoff_flow._run_conversational_flow_tui", run_tui
        )
        agent.chat(tui=True)

    expected = {"standalone_agent:conversation_turn": 2}
    if mode == "handle_turn":
        expected["standalone_agent:handle_turn"] = 2
    elif mode in {"repl", "tui"}:
        expected[f"standalone_agent:chat:{mode}"] = 1
    assert _standalone_counts(standalone_exporter) == expected
    for span in standalone_exporter.get_finished_spans():
        if span.name == "Feature Usage":
            assert "private-message-908" not in str(span.attributes)
            assert "private-agent-role" not in str(span.attributes)
            assert (
                not {"session_id", "messages", "agent_role", "error"}
                & span.attributes.keys()
            )


def test_standalone_failed_turn_counts_no_success_or_error_contents(
    standalone_exporter, monkeypatch
):
    agent = _conversation_agent()
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.setattr(
        agent.llm, "call", Mock(side_effect=RuntimeError("private-model-error-309"))
    )
    with pytest.raises(RuntimeError):
        agent.handle_turn("private-message-908")
    assert _standalone_counts(standalone_exporter) == {
        "standalone_agent:handle_turn": 1,
        "standalone_agent:conversation_turn_failed": 1,
    }
    assert "private-model-error-309" not in str(
        [
            span.attributes
            for span in standalone_exporter.get_finished_spans()
            if span.name == "Feature Usage"
        ]
    )


@pytest.mark.parametrize(
    "flag", ["CREWAI_DISABLE_TELEMETRY", "CREWAI_DISABLE_TRACKING", "OTEL_SDK_DISABLED"]
)
def test_standalone_usage_respects_telemetry_opt_out(
    standalone_exporter, monkeypatch, flag
):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.setenv(flag, "true")
    assert _conversation_agent().handle_turn("Private request").raw == "Done"
    assert _standalone_counts(standalone_exporter) == {}


def test_ordinary_agents_and_flows_do_not_emit_standalone_usage(
    standalone_exporter, monkeypatch
):
    from crewai.flow import ConversationState, Flow, listen

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = _conversation_agent()
    agent.conversational = False
    with pytest.raises(ValueError, match="conversational=True"):
        agent.handle_turn("Private request")
    agent.kickoff("Independent request")

    class ChatFlow(Flow[ConversationState]):
        conversational = True

        @listen("converse")
        def converse_turn(self):
            return "Done"

    assert ChatFlow().handle_turn("Private request") == "Done"
    assert _standalone_counts(standalone_exporter) == {}
