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
        value: Any | LongTermMemoryItem,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a value to long-term memory.
        
        Args:
            value: Either a LongTermMemoryItem instance (for backward compatibility)
                   or any value to be saved with metadata.
            metadata: Optional metadata (ignored if value is LongTermMemoryItem).
        """
        # Handle backward compatibility: if value is LongTermMemoryItem, use it directly
        if isinstance(value, LongTermMemoryItem):
            item = value
        else:
            # Convert value and metadata to LongTermMemoryItem
            # Note: This path may need adjustment based on actual usage patterns
            item = LongTermMemoryItem(
                task=str(value),
                metadata=metadata or {},
                agent=self.agent.role if self.agent else None,
                expected_output="",
                datetime=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        
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
            metadata_dict = item.metadata.copy() if item.metadata else {}
            metadata_dict.update(
                {"agent": item.agent, "expected_output": item.expected_output}
            )
            self.storage.save(
                task_description=item.task,
                score=metadata_dict.get("quality", 0.0),
                metadata=metadata_dict,
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

    def search(
        self,
        query: str | None = None,
        limit: int | None = None,
        score_threshold: float = 0.6,
        *,
        task: str | None = None,
        latest_n: int | None = None,
    ) -> list[Any]:
        """Search long-term memory for relevant entries.

        Args:
            query: The task description to search for (or use 'task' for backward compatibility).
            limit: Maximum number of results to return (or use 'latest_n' for backward compatibility).
            score_threshold: Minimum similarity score (not used in long-term memory).
            task: (Deprecated) Old parameter name for query.
            latest_n: (Deprecated) Old parameter name for limit.

        Returns:
            List of matching memory entries.
        """
        # Handle backward compatibility for parameter names
        if task is not None:
            query = task
        if latest_n is not None:
            limit = latest_n
        
        if query is None:
            raise ValueError("Either 'query' or 'task' parameter is required")
        if limit is None:
            limit = 3
        
        task = query
        latest_n = limit
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

    async def asave(
        self,
        value: Any | LongTermMemoryItem,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a value to long-term memory asynchronously.

        Args:
            value: Either a LongTermMemoryItem instance (for backward compatibility)
                   or any value to be saved with metadata.
            metadata: Optional metadata (ignored if value is LongTermMemoryItem).
        """
        # Handle backward compatibility: if value is LongTermMemoryItem, use it directly
        if isinstance(value, LongTermMemoryItem):
            item = value
        else:
            # Convert value and metadata to LongTermMemoryItem
            item = LongTermMemoryItem(
                task=str(value),
                metadata=metadata or {},
                agent=self.agent.role if self.agent else None,
                expected_output="",
                datetime=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        
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
            metadata_dict = item.metadata.copy() if item.metadata else {}
            metadata_dict.update(
                {"agent": item.agent, "expected_output": item.expected_output}
            )
            await self.storage.asave(
                task_description=item.task,
                score=metadata_dict.get("quality", 0.0),
                metadata=metadata_dict,
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

    async def asearch(
        self,
        query: str | None = None,
        limit: int | None = None,
        score_threshold: float = 0.6,
        *,
        task: str | None = None,
        latest_n: int | None = None,
    ) -> list[Any]:
        """Search long-term memory asynchronously.

        Args:
            query: The task description to search for (or use 'task' for backward compatibility).
            limit: Maximum number of results to return (or use 'latest_n' for backward compatibility).
            score_threshold: Minimum similarity score (not used in long-term memory).
            task: (Deprecated) Old parameter name for query.
            latest_n: (Deprecated) Old parameter name for limit.

        Returns:
            List of matching memory entries.
        """
        # Handle backward compatibility for parameter names
        if task is not None:
            query = task
        if latest_n is not None:
            limit = latest_n
        
        if query is None:
            raise ValueError("Either 'query' or 'task' parameter is required")
        if limit is None:
            limit = 3
        
        task = query
        latest_n = limit
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
        """Reset long-term memory."""
        self.storage.reset()
