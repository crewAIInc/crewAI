from typing import Optional, Union

from pydantic import Field

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.exceptions import AgentLookupError, UnauthorizedDelegationError
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N
from crewai.utilities.agent_lookup import AgentLookupMixin


class BaseAgentTool(BaseTool, AgentLookupMixin):
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

    def can_delegate_to(self, delegating_agent: BaseAgent, target_agent: BaseAgent) -> bool:
        """Check if an agent can delegate to another agent.
        
        Args:
            delegating_agent: The agent attempting to delegate
            target_agent: The agent being delegated to
            
        Returns:
            bool: True if delegation is allowed, False otherwise
        """
        return (delegating_agent.allow_delegation and 
                (not delegating_agent.allowed_agents or 
                 target_agent.role in delegating_agent.allowed_agents))

    def _execute(
        self, agent_name: Union[str, None], task: str, context: Union[str, None]
    ) -> str:
        try:
            if agent_name is None:
                agent_name = ""

            target_agent = self.get_agent_by_role(agent_name, self.agents)
            if not target_agent:
                raise AgentLookupError(
                    f"Agent with role '{agent_name}' not found. Available agents: "
                    f"{', '.join(agent.role for agent in self.agents)}"
                )
            delegating_agent = next(
                (agent for agent in self.agents if agent.allow_delegation), None
            )

            if delegating_agent and not self.can_delegate_to(delegating_agent, target_agent):
                raise UnauthorizedDelegationError(
                    f"Agent '{delegating_agent.role}' cannot delegate to '{target_agent.role}'. "
                    f"Allowed targets: {', '.join(delegating_agent.allowed_agents or [])}"
                )

            task_with_assigned_agent = Task(
                description=task,
                agent=target_agent,
                expected_output=target_agent.i18n.slice("manager_request"),
                i18n=target_agent.i18n,
            )
            return target_agent.execute_task(task_with_assigned_agent, context)
        except AgentLookupError as e:
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join(f"- {agent.role}" for agent in self.agents)
            )
        except UnauthorizedDelegationError as e:
            return self.i18n.errors("agent_tool_unauthorized_delegation").format(
                agent=delegating_agent.role,
                target=target_agent.role,
                allowed="\n".join(f"- {role}" for role in (delegating_agent.allowed_agents or []))
            )
        except Exception as e:
            return self.i18n.errors("tool_usage_error").format(error=str(e))
