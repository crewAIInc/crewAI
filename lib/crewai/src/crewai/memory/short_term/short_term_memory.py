from __future__ import annotations

import time
from typing import Any

from pydantic import PrivateAttr

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.memory import Memory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.memory.storage.rag_storage import RAGStorage


class ShortTermMemory(Memory):
    """
    ShortTermMemory class for managing transient data related to immediate tasks
    and interactions.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    _memory_provider: str | None = PrivateAttr()

    def __init__(
        self,
        crew: Any = None,
        embedder_config: Any = None,
        storage: Any = None,
        path: str | None = None,
    ) -> None:
        memory_provider = None
        if embedder_config and isinstance(embedder_config, dict):
            memory_provider = embedder_config.get("provider")

        if memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError as e:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                ) from e
            config = (
                embedder_config.get("config")
                if embedder_config and isinstance(embedder_config, dict)
                else None
            )
            storage = Mem0Storage(type="short_term", crew=crew, config=config)  # type: ignore[no-untyped-call]
        else:
            storage = (
                storage
                if storage
                else RAGStorage(
                    type="short_term",
                    embedder_config=embedder_config,
                    crew=crew,
                    path=path,
                )
            )
        super().__init__(storage=storage)
        self._memory_provider = memory_provider

    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=value,
                metadata=metadata,
                source_type="short_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            item = ShortTermMemoryItem(
                data=value,
                metadata=metadata,
                agent=self.agent.role if self.agent else None,
            )
            if self._memory_provider == "mem0":
                item.data = (
                    f"Remember the following insights from Agent run: {item.data}"
                )

            super().save(value=item.data, metadata=item.metadata)

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=value,
                    metadata=metadata,
                    # agent_role=agent,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="short_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    value=value,
                    metadata=metadata,
                    error=str(e),
                    source_type="short_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
            raise

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """Search short-term memory for relevant entries.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.

        Returns:
            List of matching memory entries.
        """
        crewai_event_bus.emit(
            self,
            event=MemoryQueryStartedEvent(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                source_type="short_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            results = self.storage.search(
                query=query, limit=limit, score_threshold=score_threshold
            )

            crewai_event_bus.emit(
                self,
                event=MemoryQueryCompletedEvent(
                    query=query,
                    results=results,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_time_ms=(time.time() - start_time) * 1000,
                    source_type="short_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )

            return list(results)
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemoryQueryFailedEvent(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                    error=str(e),
                    source_type="short_term_memory",
                ),
            )
            raise

    async def asave(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a value to short-term memory asynchronously.

        Args:
            value: The value to save.
            metadata: Optional metadata to associate with the value.
        """
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=value,
                metadata=metadata,
                source_type="short_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            item = ShortTermMemoryItem(
                data=value,
                metadata=metadata,
                agent=self.agent.role if self.agent else None,
            )
            if self._memory_provider == "mem0":
                item.data = (
                    f"Remember the following insights from Agent run: {item.data}"
                )

            await super().asave(value=item.data, metadata=item.metadata)

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=value,
                    metadata=metadata,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="short_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    value=value,
                    metadata=metadata,
                    error=str(e),
                    source_type="short_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
            raise

    async def asearch(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """Search short-term memory asynchronously.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.

        Returns:
            List of matching memory entries.
        """
        crewai_event_bus.emit(
            self,
            event=MemoryQueryStartedEvent(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                source_type="short_term_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            results = await self.storage.asearch(
                query=query, limit=limit, score_threshold=score_threshold
            )

            crewai_event_bus.emit(
                self,
                event=MemoryQueryCompletedEvent(
                    query=query,
                    results=results,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_time_ms=(time.time() - start_time) * 1000,
                    source_type="short_term_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )

            return list(results)
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemoryQueryFailedEvent(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                    error=str(e),
                    source_type="short_term_memory",
                ),
            )
            raise

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the short-term memory: {e}"
            ) from e
