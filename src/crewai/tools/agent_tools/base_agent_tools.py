from typing import Optional, Union
from pydantic import Field

from crewai.tools.base_tool import BaseTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
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
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers="\n".join(
                    [f"- {agent.role.casefold()}" for agent in self.agents]
                )
            )

        if not agent:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers="\n".join(
                    [f"- {agent.role.casefold()}" for agent in self.agents]
                )
            )

        agent = agent[0]
        task_with_assigned_agent = Task(  # type: ignore # Incompatible types in assignment (expression has type "Task", variable has type "str")
            description=task,
            agent=agent,
            expected_output=agent.i18n.slice("manager_request"),
            i18n=agent.i18n,
        )
        return agent.execute_task(task_with_assigned_agent, context)
