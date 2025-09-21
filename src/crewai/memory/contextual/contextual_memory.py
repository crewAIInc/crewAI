from __future__ import annotations

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
        """
        Automatically builds a minimal, highly relevant set of contextual information
        for a given task.
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

    def _fetch_stm_context(self, query: str) -> str:
        """
        Fetches recent relevant insights from STM related to the task's description and expected_output,
        formatted as bullet points.
        """

        if self.stm is None:
            return ""

        stm_results = self.stm.search(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in stm_results]
        )
        return f"Recent Insights:\n{formatted_results}" if stm_results else ""

    def _fetch_ltm_context(self, task: str) -> str | None:
        """
        Fetches historical data or insights from LTM that are relevant to the task's description and expected_output,
        formatted as bullet points.
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

        return f"Historical Data:\n{formatted_results}" if ltm_results else ""

    def _fetch_entity_context(self, query: str) -> str:
        """
        Fetches relevant entity information from Entity Memory related to the task's description and expected_output,
        formatted as bullet points.
        """
        if self.em is None:
            return ""

        em_results = self.em.search(query)
        formatted_results = "\n".join(
            [f"- {result['content']}" for result in em_results]
        )
        return f"Entities:\n{formatted_results}" if em_results else ""

    def _fetch_external_context(self, query: str) -> str:
        """
        Fetches and formats relevant information from External Memory.
        Args:
            query (str): The search query to find relevant information.
        Returns:
            str: Formatted information as bullet points, or an empty string if none found.
        """
        if self.exm is None:
            return ""

        external_memories = self.exm.search(query)

        if not external_memories:
            return ""

        formatted_memories = "\n".join(
            f"- {result['content']}" for result in external_memories
        )
        return f"External memories:\n{formatted_memories}"
