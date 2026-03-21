"""VALCrew wrapper for verifiable audit trails around Crew execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crewai.crew import Crew
from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus
from crewai.val.listener import DEFAULT_VAL_EVENT_TYPES, VALAuditListener
from crewai.val.types import VALAttestor, VALRecord, verify_val_chain


class VALCrew:
    """Wrap a ``Crew`` and automatically attest its runtime events."""

    def __init__(
        self,
        *,
        topic_id: str | None = None,
        attestor: VALAttestor | None = None,
        event_types: Sequence[type[BaseEvent]] | None = None,
        wait_for_attestations: bool = True,
        **crew_kwargs: Any,
    ) -> None:
        self.topic_id = topic_id
        self.wait_for_attestations = wait_for_attestations
        self.crew = Crew(**crew_kwargs)
        self._listener = VALAuditListener(
            crew=self.crew,
            topic_id=topic_id,
            attestor=attestor,
            event_types=event_types or DEFAULT_VAL_EVENT_TYPES,
        )

    @classmethod
    def from_crew(
        cls,
        crew: Crew,
        *,
        topic_id: str | None = None,
        attestor: VALAttestor | None = None,
        event_types: Sequence[type[BaseEvent]] | None = None,
        wait_for_attestations: bool = True,
    ) -> VALCrew:
        """Wrap an already-constructed crew."""
        wrapper = cls.__new__(cls)
        wrapper.topic_id = topic_id
        wrapper.wait_for_attestations = wait_for_attestations
        wrapper.crew = crew
        wrapper._listener = VALAuditListener(
            crew=crew,
            topic_id=topic_id,
            attestor=attestor,
            event_types=event_types or DEFAULT_VAL_EVENT_TYPES,
        )
        return wrapper

    @property
    def audit_records(self) -> list[VALRecord]:
        """Get immutable snapshot of current attested records."""
        return self._listener.records

    @property
    def attestor(self) -> VALAttestor:
        """Get active attestor backend."""
        return self._listener.attestor

    def verify_audit_trail(self) -> bool:
        """Verify local hash chain integrity for collected records."""
        return verify_val_chain(self.audit_records)

    def kickoff(self, *args: Any, **kwargs: Any) -> Any:
        """Run kickoff while guaranteeing attestation handlers are drained."""
        result = self.crew.kickoff(*args, **kwargs)
        self._flush()
        return result

    async def kickoff_async(self, *args: Any, **kwargs: Any) -> Any:
        """Run async kickoff while guaranteeing attestation handlers are drained."""
        result = await self.crew.kickoff_async(*args, **kwargs)
        self._flush()
        return result

    async def akickoff(self, *args: Any, **kwargs: Any) -> Any:
        """Run native async kickoff while guaranteeing handlers are drained."""
        result = await self.crew.akickoff(*args, **kwargs)
        self._flush()
        return result

    def _flush(self) -> None:
        if self.wait_for_attestations:
            crewai_event_bus.flush()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes and methods to the wrapped Crew."""
        return getattr(self.crew, name)
