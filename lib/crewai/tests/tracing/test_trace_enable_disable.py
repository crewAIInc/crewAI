"""Explicit tracing controls gate the execution session and its transport."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from crewai import Agent, Crew, Task
from crewai.events.listeners.tracing.utils import is_tracing_enabled_in_context
from crewai.llms.base_llm import BaseLLM
from crewai.telemetry.tracing.grants import (
    GrantSpanExporter,
    TraceGrant,
    TraceGrantClient,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


class LocalLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="local-test")

    def call(self, messages, **kwargs):
        return "Final Answer: hello"

    def supports_function_calling(self):
        return False

    def supports_stop_words(self):
        return False


@pytest.mark.parametrize(
    "environment,override,enabled",
    [
        ("false", None, False),
        ("true", False, False),
        ("true", None, True),
        ("false", True, True),
    ],
)
def test_tracing_controls_gate_execution_export(
    monkeypatch, environment, override, enabled
):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", environment)
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    exporter = InMemorySpanExporter()
    grant = Mock(
        side_effect=lambda execution_uuid: TraceGrant(
            token="synthetic-grant",
            collector_url="https://collector.invalid/v1/traces",
            execution_uuid=execution_uuid,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
    )
    monkeypatch.setattr(TraceGrantClient, "create", grant)
    monkeypatch.setattr(
        GrantSpanExporter, "_exporter", staticmethod(lambda _: exporter)
    )
    legacy = Mock(side_effect=AssertionError("legacy trace transport used"))
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.trace_batch_manager.TraceBatchManager.initialize_batch",
        legacy,
    )
    agent = Agent(role="tester", goal="greet", backstory="tester", llm=LocalLLM())
    task = Task(description="say hello", expected_output="hello", agent=agent)
    result = Crew(agents=[agent], tasks=[task], tracing=override).kickoff()
    assert result.raw == "hello"
    assert grant.call_count == int(enabled)
    assert bool(exporter.get_finished_spans()) is enabled
    legacy.assert_not_called()


@pytest.mark.parametrize(
    "environment,override,enabled",
    [
        ("false", None, False),
        ("true", None, True),
        ("true", False, False),
        ("false", True, True),
    ],
)
def test_explicit_crew_tracing_survives_agent_execution(
    monkeypatch, environment, override, enabled
):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", environment)
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    seen = []

    class ObservingLLM(LocalLLM):
        def call(self, messages, **kwargs):
            seen.append(is_tracing_enabled_in_context())
            return super().call(messages, **kwargs)

    agent = Agent(role="tester", goal="greet", backstory="tester", llm=ObservingLLM())
    task = Task(description="say hello", expected_output="hello", agent=agent)
    Crew(agents=[agent], tasks=[task], tracing=override).kickoff()
    assert seen == [enabled]
