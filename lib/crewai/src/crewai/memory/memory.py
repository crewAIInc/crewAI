from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from crewai.rag.embeddings.types import EmbedderConfig


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.task import Task


class Memory(BaseModel):
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    embedder_config: EmbedderConfig | dict[str, Any] | None = None
    crew: Any | None = None

    storage: Any
    _agent: Agent | None = None
    _task: Task | None = None

    def __init__(self, storage: Any, **data: Any):
        super().__init__(storage=storage, **data)

    @property
    def task(self) -> Task | None:
        """Get the current task associated with this memory."""
        return self._task

    @task.setter
    def task(self, task: Task | None) -> None:
        """Set the current task associated with this memory."""
        self._task = task

    @property
    def agent(self) -> Agent | None:
        """Get the current agent associated with this memory."""
        return self._agent

    @agent.setter
    def agent(self, agent: Agent | None) -> None:
        """Set the current agent associated with this memory."""
        self._agent = agent

    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}

        self.storage.save(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )

    def set_crew(self, crew: Any) -> Memory:
        self.crew = crew
        return self
