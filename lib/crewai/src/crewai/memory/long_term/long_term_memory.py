import time
from typing import Any

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


class LongTermMemory(Memory):
    """
    LongTermMemory klasse voor het beheren van cross-run data gerelateerd aan de algehele
    crew's uitvoering en prestaties.
    Erft van de Memory klasse en gebruikt een instantie van een klasse die
    voldoet aan de Storage voor data opslag, specifiek werkend met
    LongTermMemoryItem instanties.
    """

    def __init__(
        self,
        storage: LTMSQLiteStorage | None = None,
        path: str | None = None,
    ) -> None:
        if not storage:
            storage = LTMSQLiteStorage(db_path=path) if path else LTMSQLiteStorage()
        super().__init__(storage=storage)

    def save(self, item: LongTermMemoryItem) -> None:  # type: ignore # BUG?: Signature of "save" incompatible with supertype "Memory"
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=item.task,
                metadata=item.metadata,
                agent_role=item.agent,
                source_type="long_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            metadata = item.metadata
            metadata.update(
                {"agent": item.agent, "expected_output": item.expected_output}
            )
            self.storage.save(
                task_description=item.task,
                score=metadata["quality"],
                metadata=metadata,
                datetime=item.datetime,
            )

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=item.task,
                    metadata=item.metadata,
                    agent_role=item.agent,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="long_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    value=item.task,
                    metadata=item.metadata,
                    agent_role=item.agent,
                    error=str(e),
                    source_type="long_term_memory",
                ),
            )
            raise

    def search(  # type: ignore[override]
        self,
        task: str,
        latest_n: int = 3,
    ) -> list[dict[str, Any]]:
        """Zoek in lange-termijn geheugen naar relevante entries.

        Args:
            task: De taakbeschrijving om naar te zoeken.
            latest_n: Maximaal aantal resultaten om te retourneren.

        Retourneert:
            Lijst van overeenkomende geheugen entries.
        """
        crewai_event_bus.emit(
            self,
            event=MemoryQueryStartedEvent(
                query=task,
                limit=latest_n,
                source_type="long_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            results = self.storage.load(task, latest_n)

            crewai_event_bus.emit(
                self,
                event=MemoryQueryCompletedEvent(
                    query=task,
                    results=results,
                    limit=latest_n,
                    query_time_ms=(time.time() - start_time) * 1000,
                    source_type="long_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )

            return results or []
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemoryQueryFailedEvent(
                    query=task,
                    limit=latest_n,
                    error=str(e),
                    source_type="long_term_memory",
                ),
            )
            raise

    async def asave(self, item: LongTermMemoryItem) -> None:  # type: ignore[override]
        """Sla een item asynchroon op in lange-termijn geheugen.

        Args:
            item: Het LongTermMemoryItem om op te slaan.
        """
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=item.task,
                metadata=item.metadata,
                agent_role=item.agent,
                source_type="long_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            metadata = item.metadata
            metadata.update(
                {"agent": item.agent, "expected_output": item.expected_output}
            )
            await self.storage.asave(
                task_description=item.task,
                score=metadata["quality"],
                metadata=metadata,
                datetime=item.datetime,
            )

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=item.task,
                    metadata=item.metadata,
                    agent_role=item.agent,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="long_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    value=item.task,
                    metadata=item.metadata,
                    agent_role=item.agent,
                    error=str(e),
                    source_type="long_term_memory",
                ),
            )
            raise

    async def asearch(  # type: ignore[override]
        self,
        task: str,
        latest_n: int = 3,
    ) -> list[dict[str, Any]]:
        """Zoek asynchroon in lange-termijn geheugen.

        Args:
            task: De taakbeschrijving om naar te zoeken.
            latest_n: Maximaal aantal resultaten om te retourneren.

        Retourneert:
            Lijst van overeenkomende geheugen entries.
        """
        crewai_event_bus.emit(
            self,
            event=MemoryQueryStartedEvent(
                query=task,
                limit=latest_n,
                source_type="long_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            results = await self.storage.aload(task, latest_n)

            crewai_event_bus.emit(
                self,
                event=MemoryQueryCompletedEvent(
                    query=task,
                    results=results,
                    limit=latest_n,
                    query_time_ms=(time.time() - start_time) * 1000,
                    source_type="long_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )

            return results or []
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemoryQueryFailedEvent(
                    query=task,
                    limit=latest_n,
                    error=str(e),
                    source_type="long_term_memory",
                ),
            )
            raise

    def reset(self) -> None:
        """Reset lange-termijn geheugen."""
        self.storage.reset()
