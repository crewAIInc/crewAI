from crewai.events.base_event_listener import BaseEventListener
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)


class MemoryListener(BaseEventListener):
    def __init__(self, formatter):
        super().__init__()
        self.formatter = formatter
        self.memory_retrieval_in_progress = False
        self.memory_save_in_progress = False

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MemoryRetrievalStartedEvent)
        def on_memory_retrieval_started(source, event: MemoryRetrievalStartedEvent):
            if self.memory_retrieval_in_progress:
                return

            self.memory_retrieval_in_progress = True

            self.formatter.handle_memory_retrieval_started(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(MemoryRetrievalCompletedEvent)
        def on_memory_retrieval_completed(source, event: MemoryRetrievalCompletedEvent):
            if not self.memory_retrieval_in_progress:
                return

            self.memory_retrieval_in_progress = False
            self.formatter.handle_memory_retrieval_completed(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
                event.memory_content,
                event.retrieval_time_ms,
            )

        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_memory_query_completed(source, event: MemoryQueryCompletedEvent):
            if not self.memory_retrieval_in_progress:
                return

            self.formatter.handle_memory_query_completed(
                self.formatter.current_agent_branch,
                event.source_type,
                event.query_time_ms,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(MemoryQueryFailedEvent)
        def on_memory_query_failed(source, event: MemoryQueryFailedEvent):
            if not self.memory_retrieval_in_progress:
                return

            self.formatter.handle_memory_query_failed(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
                event.error,
                event.source_type,
            )

        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_memory_save_started(source, event: MemorySaveStartedEvent):
            if self.memory_save_in_progress:
                return

            self.memory_save_in_progress = True

            self.formatter.handle_memory_save_started(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
            )

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_memory_save_completed(source, event: MemorySaveCompletedEvent):
            if not self.memory_save_in_progress:
                return

            self.memory_save_in_progress = False

            self.formatter.handle_memory_save_completed(
                self.formatter.current_agent_branch,
                self.formatter.current_crew_tree,
                event.save_time_ms,
                event.source_type,
            )

        @crewai_event_bus.on(MemorySaveFailedEvent)
        def on_memory_save_failed(source, event: MemorySaveFailedEvent):
            if not self.memory_save_in_progress:
                return

            self.formatter.handle_memory_save_failed(
                self.formatter.current_agent_branch,
                event.error,
                event.source_type,
                self.formatter.current_crew_tree,
            )
