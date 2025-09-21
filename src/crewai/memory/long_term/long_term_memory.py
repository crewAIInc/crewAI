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

    def __init__(self, storage=None, path=None):
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
            self.storage.save(  # type: ignore # BUG?: Unexpected keyword argument "task_description","score","datetime" for "save" of "Storage"
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

    def search(  # type: ignore # signature of "search" incompatible with supertype "Memory"
        self,
        task: str,
        latest_n: int = 3,
    ) -> list[dict[str, Any]]:  # type: ignore # signature of "search" incompatible with supertype "Memory"
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
            results = self.storage.load(task, latest_n)  # type: ignore # BUG?: "Storage" has no attribute "load"

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

            return results
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
        self.storage.reset()
