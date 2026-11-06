"""The ungated end-of-run crew span.

Crew was the one level with no terminal record for most users: `Crew Execution`
and `end_crew` are both behind `share_crew`, which defaults False, so a normal
run produced no end-of-crew span at all -- not one with fields missing. Task and
flow outcomes already ship ungated.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from crewai import Agent, Crew, Task
from crewai.events import crewai_event_bus
from crewai.events.event_listener import EventListener
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.telemetry.telemetry import Telemetry
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


@pytest.fixture(autouse=True)
def enable_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite otherwise runs with OTEL_SDK_DISABLED, which makes every
    assertion here pass vacuously against non-recording spans."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TRACKING", raising=False)


@pytest.fixture
def telemetry():
    """Telemetry whose spans land in memory rather than being exported."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instance = Telemetry()
    with (
        patch.object(instance, "provider", provider, create=True),
        patch.object(instance, "_should_execute_telemetry", return_value=True),
    ):
        # yield, not return: the frame keeps `provider` alive, and a collected
        # provider shuts down its processor and silently loses spans.
        yield instance, exporter


@pytest.fixture
def crew() -> Crew:
    """A minimal one-agent, one-task crew for the span assertions below."""
    agent = Agent(role="r", goal="g", backstory="b")
    return Crew(
        agents=[agent],
        tasks=[Task(description="d", expected_output="e", agent=agent)],
    )


def _span(exporter: InMemorySpanExporter, name: str):
    """The single exported span with ``name``, failing loudly if it is not unique."""
    matches = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(matches) == 1, [s.name for s in exporter.get_finished_spans()]
    return matches[0]


class TestTheSpanItself:
    def test_it_records_the_outcome_and_an_explicit_duration(
        self, telemetry, crew: Crew
    ) -> None:
        """A completed run reports outcome and an explicit duration_ms."""
        instance, exporter = telemetry

        instance.crew_completed_span(crew, 1234.5, "completed")

        span = _span(exporter, "Crew Completed")
        assert span.attributes["outcome"] == "completed"
        assert span.attributes["duration_ms"] == 1234.5

    def test_a_failed_run_is_distinguishable_from_a_successful_one(
        self, telemetry, crew: Crew
    ) -> None:
        """The gap this closes: crew failure had no ungated record at all."""
        instance, exporter = telemetry

        instance.crew_completed_span(crew, 1.0, "completed")
        instance.crew_completed_span(crew, 2.0, "failed")

        outcomes = [
            s.attributes["outcome"]
            for s in exporter.get_finished_spans()
            if s.name == "Crew Completed"
        ]
        assert outcomes == ["completed", "failed"]

    def test_it_joins_to_crew_created_by_key_and_id(
        self, telemetry, crew: Crew
    ) -> None:
        """Models and shape live on Crew Created; this must be joinable to them."""
        instance, exporter = telemetry

        instance.crew_completed_span(crew, 1.0, "completed")

        span = _span(exporter, "Crew Completed")
        assert span.attributes["crew_key"] == crew.key
        assert span.attributes["crew_id"] == str(crew.id)

    def test_it_is_not_gated_by_share_crew(self, telemetry, crew: Crew) -> None:
        """The whole point. `Crew Execution` and `end_crew` are both gated."""
        instance, exporter = telemetry
        assert crew.share_crew is False

        instance.crew_completed_span(crew, 1.0, "completed")

        assert _span(exporter, "Crew Completed") is not None

    def test_it_carries_no_token_count(self, telemetry, crew: Crew) -> None:
        """`crew.token_usage` double-counts agents that share one LLM object, so
        putting it here would propagate a known-wrong number into a metric."""
        instance, exporter = telemetry

        instance.crew_completed_span(crew, 1.0, "completed")

        attributes = _span(exporter, "Crew Completed").attributes
        assert not [k for k in attributes if "token" in k.lower()]


class TestTheListenerWiring:
    """Asserts on the stamp the listener manages, not on a mock of the shared
    singleton `EventListener._telemetry` -- swapping that leaks across tests."""

    def _run(self, crew: Crew, *events) -> None:
        """Emit ``events`` through a scoped bus with a freshly wired listener."""
        with crewai_event_bus.scoped_handlers():
            EventListener().setup_listeners(crewai_event_bus)
            for event in events:
                crewai_event_bus.emit(crew, event)
                assert crewai_event_bus.flush(), "event bus did not drain"

    def test_kickoff_stamps_a_start_time(self, crew: Crew) -> None:
        """Kickoff records a start time; a terminal event consumes and clears it."""
        assert crew._telemetry_started_at is None

        self._run(crew, CrewKickoffStartedEvent(crew_name="c", inputs=None))

        assert crew._telemetry_started_at is not None

    @pytest.mark.parametrize("terminal", ["completed", "failed"])
    def test_either_terminal_event_consumes_the_stamp(
        self, crew: Crew, terminal: str
    ) -> None:
        """Both paths must report, and both must clear -- a stamp left behind
        would let a later event report a duration measured from the wrong run."""
        output = MagicMock()
        output.raw = "done"
        end = (
            CrewKickoffCompletedEvent(crew_name="c", output=output, total_tokens=0)
            if terminal == "completed"
            else CrewKickoffFailedEvent(crew_name="c", error="boom")
        )

        self._run(crew, CrewKickoffStartedEvent(crew_name="c", inputs=None), end)

        assert crew._telemetry_started_at is None

    def test_a_terminal_event_without_a_start_is_inert(self, crew: Crew) -> None:
        """A listener attached mid-run has no stamp; that is not an error."""
        self._run(crew, CrewKickoffFailedEvent(crew_name="c", error="boom"))

        assert crew._telemetry_started_at is None
