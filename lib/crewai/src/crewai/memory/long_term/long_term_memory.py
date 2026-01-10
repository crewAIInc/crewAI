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
    LongTermMemory class for managing cross runs data related to overall crew's
    execution and performance.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    LongTermMemoryItem instances.
    """

    def __init__(
        self,
        storage: LTMSQLiteStorage | None = None,
        path: str | None = None,
    ) -> None:
        if not storage:
            storage = LTMSQLiteStorage(db_path=path) if path else LTMSQLiteStorage()
        super().__init__(storage=storage)

    def save(
        self,
        value: LongTermMemoryItem,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save an item to long-term memory.

        Args:
            value: The LongTermMemoryItem to save.
            metadata: Optional metadata dict (not used, metadata is extracted from the
                LongTermMemoryItem). Included for supertype compatibility.
        """
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=value.task,
                metadata=value.metadata,
                agent_role=value.agent,
                source_type="long_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            item_metadata = value.metadata
            item_metadata.update(
                {"agent": value.agent, "expected_output": value.expected_output}
            )
            self.storage.save(
                task_description=value.task,
                score=item_metadata["quality"],
                metadata=item_metadata,
                datetime=value.datetime,
            )

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=value.task,
                    metadata=value.metadata,
                    agent_role=value.agent,
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
                    value=value.task,
                    metadata=value.metadata,
                    agent_role=value.agent,
                    error=str(e),
                    source_type="long_term_memory",
                ),
            )
            raise

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Search long-term memory for relevant entries.

        Args:
            query: The task description to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results (not used for
                long-term memory, included for supertype compatibility).

        Returns:
            List of matching memory entries.
        """
        crewai_event_bus.emit(
            self,
            event=MemoryQueryStartedEvent(
                query=query,
                limit=limit,
                source_type="long_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            results = self.storage.load(query, limit)

            crewai_event_bus.emit(
                self,
                event=MemoryQueryCompletedEvent(
                    query=query,
                    results=results,
                    limit=limit,
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
                    query=query,
                    limit=limit,
                    error=str(e),
                    source_type="long_term_memory",
                ),
            )
            raise

    async def asave(
        self,
        value: LongTermMemoryItem,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save an item to long-term memory asynchronously.

        Args:
            value: The LongTermMemoryItem to save.
            metadata: Optional metadata dict (not used, metadata is extracted from the
                LongTermMemoryItem). Included for supertype compatibility.
        """
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=value.task,
                metadata=value.metadata,
                agent_role=value.agent,
                source_type="long_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            item_metadata = value.metadata
            item_metadata.update(
                {"agent": value.agent, "expected_output": value.expected_output}
            )
            await self.storage.asave(
                task_description=value.task,
                score=item_metadata["quality"],
                metadata=item_metadata,
                datetime=value.datetime,
            )

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=value.task,
                    metadata=value.metadata,
                    agent_role=value.agent,
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
                    value=value.task,
                    metadata=value.metadata,
                    agent_role=value.agent,
                    error=str(e),
                    source_type="long_term_memory",
                ),
            )
            raise

    async def asearch(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Search long-term memory asynchronously.

        Args:
            query: The task description to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results (not used for
                long-term memory, included for supertype compatibility).

        Returns:
            List of matching memory entries.
        """
        crewai_event_bus.emit(
            self,
            event=MemoryQueryStartedEvent(
                query=query,
                limit=limit,
                source_type="long_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            results = await self.storage.aload(query, limit)

            crewai_event_bus.emit(
                self,
                event=MemoryQueryCompletedEvent(
                    query=query,
                    results=results,
                    limit=limit,
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
                    query=query,
                    limit=limit,
                    error=str(e),
                    source_type="long_term_memory",
                ),
            )
            raise

    def reset(self) -> None:
        """Reset long-term memory."""
        self.storage.reset()
