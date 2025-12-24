from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from crewai.memory import (
    EntityMemory,
    ExternalMemory,
    LongTermMemory,
    ShortTermMemory,
)


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.task import Task


class ContextualMemory:
    """Aggregeert en haalt context op uit meerdere geheugenbronnen."""

    def __init__(
        self,
        stm: ShortTermMemory,
        ltm: LongTermMemory,
        em: EntityMemory,
        exm: ExternalMemory,
        agent: Agent | None = None,
        task: Task | None = None,
    ) -> None:
        self.stm = stm
        self.ltm = ltm
        self.em = em
        self.exm = exm
        self.agent = agent
        self.task = task

        if self.stm is not None:
            self.stm.agent = self.agent
            self.stm.task = self.task
        if self.ltm is not None:
            self.ltm.agent = self.agent
            self.ltm.task = self.task
        if self.em is not None:
            self.em.agent = self.agent
            self.em.task = self.task
        if self.exm is not None:
            self.exm.agent = self.agent
            self.exm.task = self.task

    def build_context_for_task(self, task: Task, context: str) -> str:
        """Bouw contextuele informatie voor een taak synchroon.

        Args:
            task: De taak om context voor te bouwen.
            context: Aanvullende context string.

        Retourneert:
            Geformatteerde context string van alle geheugenbronnen.
        """
        query = f"{task.description} {context}".strip()

        if query == "":
            return ""

        context_parts = [
            self._fetch_ltm_context(task.description),
            self._fetch_stm_context(query),
            self._fetch_entity_context(query),
            self._fetch_external_context(query),
        ]
        return "\n".join(filter(None, context_parts))

    async def abuild_context_for_task(self, task: Task, context: str) -> str:
        """Bouw contextuele informatie voor een taak asynchroon.

        Args:
            task: De taak om context voor te bouwen.
            context: Aanvullende context string.

        Retourneert:
            Geformatteerde context string van alle geheugenbronnen.
        """
        query = f"{task.description} {context}".strip()

        if query == "":
            return ""

        # Fetch all contexts concurrently
        results = await asyncio.gather(
            self._afetch_ltm_context(task.description),
            self._afetch_stm_context(query),
            self._afetch_entity_context(query),
            self._afetch_external_context(query),
        )

        return "\n".join(filter(None, results))

    def _fetch_stm_context(self, query: str) -> str:
        """
        Haalt recente relevante inzichten op uit STM gerelateerd aan de taakbeschrijving en expected_output,
        geformatteerd als opsommingspunten.
        """

        if self.stm is None:
            return ""

        stm_results = self.stm.search(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in stm_results]
        )
        return f"Recente Inzichten:\n{formatted_results}" if stm_results else ""

    def _fetch_ltm_context(self, task: str) -> str | None:
        """
        Haalt historische data of inzichten op uit LTM die relevant zijn voor de taakbeschrijving en expected_output,
        geformatteerd als opsommingspunten.
        """

        if self.ltm is None:
            return ""

        ltm_results = self.ltm.search(task, latest_n=2)
        if not ltm_results:
            return None

        formatted_results = [
            suggestion
            for result in ltm_results
            for suggestion in result["metadata"]["suggestions"]
        ]
        formatted_results = list(dict.fromkeys(formatted_results))
        formatted_results = "\n".join([f"- {result}" for result in formatted_results])  # type: ignore # Incompatible types in assignment (expression has type "str", variable has type "list[str]")

        return f"Historische Data:\n{formatted_results}" if ltm_results else ""

    def _fetch_entity_context(self, query: str) -> str:
        """
        Haalt relevante entiteit informatie op uit Entity Memory gerelateerd aan de taakbeschrijving en expected_output,
        geformatteerd als opsommingspunten.
        """
        if self.em is None:
            return ""

        em_results = self.em.search(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in em_results]
        )
        return f"Entiteiten:\n{formatted_results}" if em_results else ""

    def _fetch_external_context(self, query: str) -> str:
        """
        Haalt relevante informatie op en formatteert deze uit Extern Geheugen.
        Args:
            query (str): De zoekquery om relevante informatie te vinden.
        Retourneert:
            str: Geformatteerde informatie als opsommingspunten, of een lege string als niets gevonden.
        """
        if self.exm is None:
            return ""

        external_memories = self.exm.search(query)

        if not external_memories:
            return ""

        formatted_memories = "\n".join(
            f"- {result['content']}" for result in external_memories
        )
        return f"Externe herinneringen:\n{formatted_memories}"

    async def _afetch_stm_context(self, query: str) -> str:
        """Haal recente relevante inzichten asynchroon op uit STM.

        Args:
            query: De zoekquery.

        Retourneert:
            Geformatteerde inzichten als opsommingspunten, of lege string als niets gevonden.
        """
        if self.stm is None:
            return ""

        stm_results = await self.stm.asearch(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in stm_results]
        )
        return f"Recente Inzichten:\n{formatted_results}" if stm_results else ""

    async def _afetch_ltm_context(self, task: str) -> str | None:
        """Haal historische data asynchroon op uit LTM.

        Args:
            task: De taakbeschrijving om naar te zoeken.

        Retourneert:
            Geformatteerde historische data als opsommingspunten, of None als niets gevonden.
        """
        if self.ltm is None:
            return ""

        ltm_results = await self.ltm.asearch(task, latest_n=2)
        if not ltm_results:
            return None

        formatted_results = [
            suggestion
            for result in ltm_results
            for suggestion in result["metadata"]["suggestions"]
        ]
        formatted_results = list(dict.fromkeys(formatted_results))
        formatted_results = "\n".join([f"- {result}" for result in formatted_results])  # type: ignore # Incompatible types in assignment (expression has type "str", variable has type "list[str]")

        return f"Historische Data:\n{formatted_results}" if ltm_results else ""

    async def _afetch_entity_context(self, query: str) -> str:
        """Haal relevante entiteit informatie asynchroon op.

        Args:
            query: De zoekquery.

        Retourneert:
            Geformatteerde entiteit informatie als opsommingspunten, of lege string als niets gevonden.
        """
        if self.em is None:
            return ""

        em_results = await self.em.asearch(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in em_results]
        )
        return f"Entiteiten:\n{formatted_results}" if em_results else ""

    async def _afetch_external_context(self, query: str) -> str:
        """Haal relevante informatie asynchroon op uit Extern Geheugen.

        Args:
            query: De zoekquery.

        Retourneert:
            Geformatteerde informatie als opsommingspunten, of lege string als niets gevonden.
        """
        if self.exm is None:
            return ""

        external_memories = await self.exm.asearch(query)

        if not external_memories:
            return ""

        formatted_memories = "\n".join(
            f"- {result['content']}" for result in external_memories
        )
        return f"Externe herinneringen:\n{formatted_memories}"
