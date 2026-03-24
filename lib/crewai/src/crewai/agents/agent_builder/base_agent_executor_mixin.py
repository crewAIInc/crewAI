from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.agents.parser import AgentFinish
from crewai.memory.utils import sanitize_scope_name
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
        """Save task result to unified memory (memory or crew._memory).

        Extends the memory's root_scope with agent-specific path segment
        (e.g., '/crew/research-crew/agent/researcher') so that agent memories
        are scoped hierarchically under their crew.
        """
        memory = getattr(self.agent, "memory", None) or (
            getattr(self.crew, "_memory", None) if self.crew else None
        )
        if memory is None or not self.task or memory.read_only:
            return
        if f"Action: {sanitize_tool_name('Delegate work to coworker')}" in output.text:
            return
        try:
            raw = (
                f"Task: {self.task.description}\n"
                f"Agent: {self.agent.role}\n"
                f"Expected result: {self.task.expected_output}\n"
                f"Result: {output.text}"
            )
            extracted = memory.extract_memories(raw)
            if extracted:
                # Get the memory's existing root_scope
                base_root = getattr(memory, "root_scope", None)

                if isinstance(base_root, str) and base_root:
                    # Memory has a root_scope — extend it with agent info
                    agent_role = self.agent.role or "unknown"
                    sanitized_role = sanitize_scope_name(agent_role)
                    agent_root = f"{base_root.rstrip('/')}/agent/{sanitized_role}"
                    if not agent_root.startswith("/"):
                        agent_root = "/" + agent_root
                    memory.remember_many(
                        extracted, agent_role=self.agent.role, root_scope=agent_root
                    )
                else:
                    # No base root_scope — don't inject one, preserve backward compat
                    memory.remember_many(extracted, agent_role=self.agent.role)
        except Exception as e:
            self.agent._logger.log("error", f"Failed to save to memory: {e}")
