from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from crewai.agents.parser import AgentFinish
from crewai.memory.utils import sanitize_scope_name
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import sanitize_tool_name


if TYPE_CHECKING:
    pass


class CrewAgentExecutorMixin(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    _crew: Any = PrivateAttr(default=None)
    _agent: Any = PrivateAttr(default=None)
    _task: Any = PrivateAttr(default=None)
    iterations: int = Field(default=0)
    max_iter: int = Field(default=25)
    messages: list[Any] = Field(default_factory=list)
    _i18n: Any = PrivateAttr(default=None)
    _printer: Printer = PrivateAttr(default_factory=Printer)

    @property
    def crew(self) -> Any:
        return self._crew

    @crew.setter
    def crew(self, value: Any) -> None:
        self._crew = value

    @property
    def agent(self) -> Any:
        return self._agent

    @agent.setter
    def agent(self, value: Any) -> None:
        self._agent = value

    @property
    def task(self) -> Any:
        return self._task

    @task.setter
    def task(self, value: Any) -> None:
        self._task = value

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
                base_root = getattr(memory, "root_scope", None)

                if isinstance(base_root, str) and base_root:
                    agent_role = self.agent.role or "unknown"
                    sanitized_role = sanitize_scope_name(agent_role)
                    agent_root = f"{base_root.rstrip('/')}/agent/{sanitized_role}"
                    if not agent_root.startswith("/"):
                        agent_root = "/" + agent_root
                    memory.remember_many(
                        extracted, agent_role=self.agent.role, root_scope=agent_root
                    )
                else:
                    memory.remember_many(extracted, agent_role=self.agent.role)
        except Exception as e:
            self.agent._logger.log("error", f"Failed to save to memory: {e}")
