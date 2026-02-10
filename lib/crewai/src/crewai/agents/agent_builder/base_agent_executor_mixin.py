from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.agents.parser import AgentFinish
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import sanitize_tool_name


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.crew import Crew
    from crewai.task import Task
    from crewai.utilities.i18n import I18N
    from crewai.utilities.types import LLMMessage


class CrewAgentExecutorMixin:
    crew: Crew | None
    agent: Agent
    task: Task | None
    iterations: int
    max_iter: int
    messages: list[LLMMessage]
    _i18n: I18N
    _printer: Printer = Printer()

    def _save_to_memory(self, output: AgentFinish) -> None:
        """Save task result to unified memory (memory or crew._memory)."""
        import logging as _logging

        _mem_logger = _logging.getLogger("crewai.memory.debug")

        memory = getattr(self.agent, "memory", None) or (
            getattr(self.crew, "_memory", None) if self.crew else None
        )
        _mem_logger.warning(
            "[_save_to_memory] agent.memory=%s, crew._memory=%s, resolved=%s",
            type(getattr(self.agent, "memory", None)).__name__,
            type(getattr(self.crew, "_memory", None) if self.crew else None).__name__,
            type(memory).__name__,
        )
        if memory is None or not self.task:
            _mem_logger.warning("[_save_to_memory] Skipped: memory=%s, task=%s", memory, self.task)
            return
        if (
            f"Action: {sanitize_tool_name('Delegate work to coworker')}"
            in output.text
        ):
            return
        try:
            raw = (
                f"Task: {self.task.description}\n"
                f"Agent: {self.agent.role}\n"
                f"Expected result: {self.task.expected_output}\n"
                f"Result: {output.text}"
            )
            _mem_logger.warning("[_save_to_memory] Calling extract_memories (%d chars)", len(raw))
            extracted = memory.extract_memories(raw)
            _mem_logger.warning("[_save_to_memory] extract_memories returned %d items", len(extracted))
            for i, mem in enumerate(extracted):
                memory.remember(mem)
                _mem_logger.warning("[_save_to_memory] remember(%d) succeeded", i)
        except Exception as e:
            _mem_logger.warning("[_save_to_memory] FAILED: %s: %s", type(e).__name__, e)
            self.agent._logger.log(
                "error", f"Failed to save to memory: {e}"
            )
