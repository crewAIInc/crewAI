from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from crewai.agents.parser import AgentFinish
from crewai.memory.utils import sanitize_scope_name
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.crew import Crew
    from crewai.task import Task


class BaseAgentExecutor(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    executor_type: str = "base"
    crew: Crew | None = Field(default=None, exclude=True)
    agent: BaseAgent | None = Field(default=None, exclude=True)
    task: Task | None = Field(default=None, exclude=True)
    iterations: int = Field(default=0)
    max_iter: int = Field(default=25)
    messages: list[LLMMessage] = Field(default_factory=list)
    _resuming: bool = PrivateAttr(default=False)

    def _save_to_memory(self, output: AgentFinish) -> None:
        """Save task result to unified memory (memory or crew._memory)."""
        if self.agent is None:
            return
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
