from typing import Optional, Union
from pydantic import UUID4, Field

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N


class BaseAgentTool(BaseTool):
    """Base class for agent-related tools"""

    agents: list[BaseAgent] = Field(description="List of available agents")
    agent_id: UUID4 = Field(description="ID of the agent using this tool")
    i18n: I18N = Field(
        default_factory=I18N, description="Internationalization settings"
    )

    def _get_agent_by_id(self, agent_id: UUID4) -> Optional[BaseAgent]:
        """Helper method to find agent by ID."""
        return next((a for a in self.agents if a.id == agent_id), None)

    def _get_agent_by_role(self, role: str) -> Optional[BaseAgent]:
        """Helper method to find agent by role (case-insensitive)."""
        return next(
            (a for a in self.agents if a.role.casefold() == role.casefold()),
            None
        )

    def _check_delegation_authorization(
        self, delegating_agent: BaseAgent, target_role: str
    ) -> Optional[str]:
        """Verify if delegation is authorized.
        
        Args:
            delegating_agent: The agent attempting to delegate
            target_role: The role of the agent being delegated to

        Returns:
            Optional[str]: Error message if delegation is not authorized, None otherwise
        """
        if (delegating_agent.allowed_agents is not None and
            not any(allowed.casefold() == target_role.casefold() 
                   for allowed in delegating_agent.allowed_agents)):
            return self.i18n.errors("agent_tool_unauthorized_delegation").format(
                coworker=target_role,
                allowed_agents="\n".join([f"- {role}" for role in delegating_agent.allowed_agents])
            )
        return None


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

            # It is important to remove the quotes from the agent name.
            # The reason we have to do this is because less-powerful LLM's
            # have difficulty producing valid JSON.
            # As a result, we end up with invalid JSON that is truncated like this:
            # {"task": "....", "coworker": "....
            # when it should look like this:
            # {"task": "....", "coworker": "...."}
            agent_name = agent_name.casefold().replace('"', "").replace("\n", "")
            agent = [  # type: ignore # Incompatible types in assignment (expression has type "list[BaseAgent]", variable has type "str | None")
                available_agent
                for available_agent in self.agents
                if available_agent.role.casefold().replace("\n", "") == agent_name
            ]
        except Exception as _:
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join(
                    [f"- {agent.role.casefold()}" for agent in self.agents]
                )
            )

        if not agent:
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join(
                    [f"- {agent.role.casefold()}" for agent in self.agents]
                )
            )

        # Get delegating agent and check authorization
        delegating_agent = self._get_agent_by_id(self.agent_id)
        if not delegating_agent:
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join([f"- {agent.role}" for agent in self.agents])
            )

        auth_error = self._check_delegation_authorization(delegating_agent, agent[0].role)
        if auth_error:
            return auth_error

        agent = agent[0]
        task_with_assigned_agent = Task(  # type: ignore # Incompatible types in assignment (expression has type "Task", variable has type "str")
            description=task,
            agent=agent,
            expected_output=agent.i18n.slice("manager_request"),
            i18n=agent.i18n,
        )
        return agent.execute_task(task_with_assigned_agent, context)
