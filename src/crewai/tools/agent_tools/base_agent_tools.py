from typing import Optional, Union

from pydantic import Field

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N


class BaseAgentTool(BaseTool):
    """Base class for agent-related tools"""

    agents: list[BaseAgent] = Field(description="List of available agents")
    i18n: I18N = Field(
        default_factory=I18N, description="Internationalization settings"
    )

    def _get_coworker(self, coworker: Optional[str], **kwargs) -> Optional[str]:
        coworker = coworker or kwargs.get("co_worker") or kwargs.get("coworker")
        if coworker:
            is_list = coworker.startswith("[") and coworker.endswith("]")
            if is_list:
                coworker = coworker[1:-1].split(",")[0]
        return coworker

    def _execute(
        self, agent_name: Union[str, None], task: str, context: Union[str, None]
    ) -> str:
        try:
            if agent_name is None:
                agent_name = ""

            agent_name = agent_name.casefold().replace('"', "").replace("\n", "")
            available_agents = [
                available_agent
                for available_agent in self.agents
                if available_agent.role.casefold().replace("\n", "") == agent_name
            ]

            if not available_agents:
                return self.i18n.errors("agent_tool_unexisting_coworker").format(
                    coworkers="\n".join(
                        [f"- {agent.role.casefold()}" for agent in self.agents]
                    )
                )

            target_agent = available_agents[0]
            delegating_agent = next(
                (agent for agent in self.agents if agent.allow_delegation), None
            )

            if delegating_agent and delegating_agent.allowed_agents:
                if target_agent.role not in delegating_agent.allowed_agents:
                    return self.i18n.errors("agent_tool_unauthorized_delegation").format(
                        agent=delegating_agent.role,
                        target=target_agent.role,
                        allowed="\n".join(f"- {role}" for role in delegating_agent.allowed_agents)
                    )

            task_with_assigned_agent = Task(
                description=task,
                agent=target_agent,
                expected_output=target_agent.i18n.slice("manager_request"),
                i18n=target_agent.i18n,
            )
            return target_agent.execute_task(task_with_assigned_agent, context)
        except Exception as _:
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join(
                    [f"- {agent.role.casefold()}" for agent in self.agents]
                )
            )
