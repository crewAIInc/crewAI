from typing import Any

from pydantic import BaseModel


class Memory(BaseModel):
    """Base class for memory, now supporting agent tags and generic metadata."""

    embedder_config: dict[str, Any] | None = None
    crew: Any | None = None

    storage: Any

    def __init__(self, storage: Any, **data: Any) -> None:
        super().__init__(storage=storage, **data)

    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent

        self.storage.save(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> list[Any]:
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold,
        )

    def set_crew(self, crew: Any) -> "Memory":
        self.crew = crew
        return self
