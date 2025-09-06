from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.task import Task


class Memory(BaseModel):
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    embedder_config: Optional[Dict[str, Any]] = None
    crew: Optional[Any] = None

    storage: Any
    _agent: Optional["Agent"] = None
    _task: Optional["Task"] = None

    def __init__(self, storage: Any, **data: Any):
        super().__init__(storage=storage, **data)

    @property
    def task(self) -> Optional["Task"]:
        """Get the current task associated with this memory."""
        return self._task

    @task.setter
    def task(self, task: Optional["Task"]) -> None:
        """Set the current task associated with this memory."""
        self._task = task

    @property
    def agent(self) -> Optional["Agent"]:
        """Get the current agent associated with this memory."""
        return self._agent

    @agent.setter
    def agent(self, agent: Optional["Agent"]) -> None:
        """Set the current agent associated with this memory."""
        self._agent = agent

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}

        self.storage.save(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )

    def set_crew(self, crew: Any) -> "Memory":
        self.crew = crew
        return self
