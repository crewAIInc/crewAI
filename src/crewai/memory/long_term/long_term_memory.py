import time
from typing import Any, Optional

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

    def __init__(self, storage: Any = None, path: Any = None) -> None:
        if not storage:
            storage = LTMSQLiteStorage(db_path=path) if path else LTMSQLiteStorage()
        super().__init__(storage=storage)

    def save(
        self,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        # Handle both LongTermMemoryItem and regular save calls
        if isinstance(value, LongTermMemoryItem):
            item = value
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
                metadata = item.metadata.copy()
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
        else:
            # Regular save for compatibility with parent class
            metadata = metadata or {}
            self.storage.save(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> list[Any]:
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
            # The storage.load method uses different parameter names
            # but we'll call it with the aligned names
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

            return results if results is not None else []
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemoryQueryFailedEvent(
                    query=query,
                    limit=limit,
                    error=str(e),
                    source_type="long_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
            raise

    def reset(self) -> None:
        self.storage.reset()
