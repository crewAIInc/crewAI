"""Default-behaviour tests for OpenTelemetry instrumentation.

These tests assert that, when no SDK ``TracerProvider`` is installed,
``operation()`` and every hot-path wrapper degrade to NoOp spans and
``Crew.kickoff`` runs without exception. They MUST live in their own file
because ``ProxyTracer`` instances cache the first resolved real tracer
process-wide; once another test (in any other file under the same xdist
worker) installs an SDK provider, the proxy is no longer observable.

``pytest --dist=loadfile`` (configured in ``pyproject.toml``) is what
guarantees this file gets its own worker.
"""

from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.telemetry.otel import operation
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, ProxyTracerProvider


class _FakeLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(model="test-model")

    def call(  # type: ignore[override]
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        return "ok"

    def supports_function_calling(self) -> bool:
        return False


def test_default_provider_is_proxy() -> None:
    assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)


def test_operation_yields_non_recording_span_when_no_provider() -> None:
    with operation("standalone") as span:
        assert isinstance(span, NonRecordingSpan)


def test_constructing_crew_does_not_globalize_anonymous_telemetry_provider() -> None:
    agent = Agent(
        role="tester",
        goal="goal",
        backstory="backstory",
        llm=_FakeLLM(),
        allow_delegation=False,
    )
    Crew(
        agents=[agent],
        tasks=[Task(description="d", expected_output="o", agent=agent)],
    )
    assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)


def test_kickoff_runs_cleanly_without_provider() -> None:
    agent = Agent(
        role="tester",
        goal="goal",
        backstory="backstory",
        llm=_FakeLLM(),
        allow_delegation=False,
    )
    task = Task(description="do a thing", expected_output="anything", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    result = crew.kickoff()

    assert result is not None
    assert str(result)
    # Provider must still be the proxy; operation() should not have flipped a
    # real SDK provider into place.
    assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)
