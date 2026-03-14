"""Tests for Verifiable Audit Log (VAL) integration."""

from dataclasses import replace
import uuid

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
)
from crewai.val import InMemoryVALAttestor, VALAuditListener, VALCrew, verify_val_chain


class _CrewStub:
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.tasks: list[object] = []
        self.agents: list[object] = []


class _KickoffCrewStub(_CrewStub):
    def kickoff(self, *args, **kwargs):
        crewai_event_bus.emit(
            self,
            CrewKickoffStartedEvent(crew_name="wrapped", inputs={"topic": "audit"}),
        )
        crewai_event_bus.emit(
            self,
            CrewKickoffCompletedEvent(crew_name="wrapped", output="ok", total_tokens=2),
        )
        return "wrapped-result"


def test_val_listener_filters_other_crews_and_builds_hash_chain() -> None:
    primary_crew = _CrewStub()
    other_crew = _CrewStub()
    attestor = InMemoryVALAttestor()

    with crewai_event_bus.scoped_handlers():
        listener = VALAuditListener(
            crew=primary_crew,
            topic_id="0.0.12345",
            attestor=attestor,
            event_types=(CrewKickoffStartedEvent, CrewKickoffCompletedEvent),
        )

        crewai_event_bus.emit(
            primary_crew,
            CrewKickoffStartedEvent(crew_name="primary", inputs={"topic": "audit"}),
        )
        crewai_event_bus.emit(
            other_crew,
            CrewKickoffStartedEvent(crew_name="other", inputs={"topic": "ignore"}),
        )
        crewai_event_bus.emit(
            primary_crew,
            CrewKickoffCompletedEvent(crew_name="primary", output="done", total_tokens=3),
        )
        crewai_event_bus.flush()

        assert len(listener.records) == 2
        assert len(attestor.records) == 2
        assert listener.verify()


def test_verify_val_chain_detects_tampering() -> None:
    primary_crew = _CrewStub()
    attestor = InMemoryVALAttestor()

    with crewai_event_bus.scoped_handlers():
        listener = VALAuditListener(
            crew=primary_crew,
            topic_id="0.0.12345",
            attestor=attestor,
            event_types=(CrewKickoffStartedEvent, CrewKickoffCompletedEvent),
        )
        crewai_event_bus.emit(
            primary_crew,
            CrewKickoffStartedEvent(crew_name="primary", inputs=None),
        )
        crewai_event_bus.emit(
            primary_crew,
            CrewKickoffCompletedEvent(crew_name="primary", output="done", total_tokens=1),
        )
        crewai_event_bus.flush()

        records = listener.records
        assert verify_val_chain(records)
        tampered = replace(records[1], payload={"tampered": True})
        assert not verify_val_chain([records[0], tampered])


def test_valcrew_kickoff_attests_events() -> None:
    attestor = InMemoryVALAttestor()
    val_crew = VALCrew.from_crew(
        _KickoffCrewStub(),
        topic_id="0.0.12345",
        attestor=attestor,
    )
    result = val_crew.kickoff(inputs={"topic": "audit"})

    assert result == "wrapped-result"
    assert len(val_crew.audit_records) == 2
    assert val_crew.verify_audit_trail()


def test_valcrew_import_compatibility() -> None:
    from crewai import VALCrew as PublicVALCrew
    from crewai.val import VALCrew as CoreVALCrew
    from crewai_val import VALCrew as CompatVALCrew

    assert PublicVALCrew is CoreVALCrew
    assert CompatVALCrew is CoreVALCrew
