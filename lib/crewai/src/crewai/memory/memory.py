from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from crewai.rag.embeddings.types import EmbedderConfig


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.task import Task


class Memory(BaseModel):
    """Basis klasse voor geheugen, ondersteunt agent tags en generieke metadata."""

    embedder_config: EmbedderConfig | dict[str, Any] | None = None
    crew: Any | None = None

    storage: Any
    _agent: Agent | None = None
    _task: Task | None = None

    def __init__(self, storage: Any, **data: Any):
        super().__init__(storage=storage, **data)

    @property
    def task(self) -> Task | None:
        """Haal de huidige taak op die aan dit geheugen is gekoppeld."""
        return self._task

    @task.setter
    def task(self, task: Task | None) -> None:
        """Stel de huidige taak in die aan dit geheugen is gekoppeld."""
        self._task = task

    @property
    def agent(self) -> Agent | None:
        """Haal de huidige agent op die aan dit geheugen is gekoppeld."""
        return self._agent

    @agent.setter
    def agent(self, agent: Agent | None) -> None:
        """Stel de huidige agent in die aan dit geheugen is gekoppeld."""
        self._agent = agent

    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Sla een waarde op in het geheugen.

        Args:
            value: De waarde om op te slaan.
            metadata: Optionele metadata om te koppelen aan de waarde.
        """
        metadata = metadata or {}
        self.storage.save(value, metadata)

    async def asave(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Sla een waarde asynchroon op in het geheugen.

        Args:
            value: De waarde om op te slaan.
            metadata: Optionele metadata om te koppelen aan de waarde.
        """
        metadata = metadata or {}
        await self.storage.asave(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """Zoek in het geheugen naar relevante entries.

        Args:
            query: De zoekquery.
            limit: Maximaal aantal resultaten om te retourneren.
            score_threshold: Minimale gelijkenisscore voor resultaten.

        Retourneert:
            Lijst van overeenkomende geheugen entries.
        """
        results: list[Any] = self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )
        return results

    async def asearch(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """Zoek asynchroon in het geheugen naar relevante entries.

        Args:
            query: De zoekquery.
            limit: Maximaal aantal resultaten om te retourneren.
            score_threshold: Minimale gelijkenisscore voor resultaten.

        Retourneert:
            Lijst van overeenkomende geheugen entries.
        """
        results: list[Any] = await self.storage.asearch(
            query=query, limit=limit, score_threshold=score_threshold
        )
        return results

    def set_crew(self, crew: Any) -> Memory:
        """Stel de crew in voor deze geheugen instantie."""
        self.crew = crew
        return self
