from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from crewai.agents.third_party_agents.base_agent import BaseAgent
from crewai.task import Task
from crewai.utilities import I18N


class BaseAgentTools(BaseModel, ABC):
    """Default tools around agent delegation"""

    agents: List[BaseAgent] = Field(description="List of agents in this crew.")
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")

    @abstractmethod
    def tools(self):
        pass

    def _get_coworker(self, coworker: Optional[str], **kwargs) -> Optional[str]:
        coworker = coworker or kwargs.get("co_worker") or kwargs.get("coworker")
        if coworker and coworker.startswith("[") and coworker.endswith("]"):
            coworker = coworker[1:-1].split(",")[0]
        return coworker

    def delegate_work(
        self, task: str, context: str, coworker: Optional[str] = None, **kwargs
    ):
        """Useful to delegate a specific task to a coworker passing all necessary context and names."""
        coworker = self._get_coworker(coworker, **kwargs)
        return self._execute(coworker, task, context)

    def ask_question(
        self, question: str, context: str, coworker: Optional[str] = None, **kwargs
    ):
        """Useful to ask a question, opinion or take from a coworker passing all necessary context and names."""
        coworker = self._get_coworker(coworker, **kwargs)
        return self._execute(coworker, question, context)

    def _execute(self, agent_role: Optional[str], task: str, context: str):
        """Execute the command."""
        if agent_role:
            agent = next(
                (
                    agent
                    for agent in self.agents
                    if agent.role.casefold().strip() == agent_role.casefold().strip()
                ),
                None,
            )
        else:
            agent = None

        if not agent:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers="\n".join(
                    [f"- {agent.role.casefold()}" for agent in self.agents]
                )
            )

        task = Task(
            description=task,
            agent=agent,
            expected_output="Your best answer to your coworker asking you this, accounting for the context shared.",
        )
        return agent.execute_task(task, context)
