import logging

from pydantic import Field

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N

logger = logging.getLogger(__name__)


class BaseAgentTool(BaseTool):
    """Base class for agent-related tools"""

    agents: list[BaseAgent] = Field(description="List of available agents")
    i18n: I18N = Field(
        default_factory=I18N, description="Internationalization settings"
    )

    def sanitize_agent_name(self, name: str) -> str:
        """
        Sanitize agent role name by normalizing whitespace and setting to lowercase.
        Converts all whitespace (including newlines) to single spaces and removes quotes.

        Args:
            name (str): The agent role name to sanitize

        Returns:
            str: The sanitized agent role name, with whitespace normalized,
                 converted to lowercase, and quotes removed
        """
        if not name:
            return ""
        # Normalize all whitespace (including newlines) to single spaces
        normalized = " ".join(name.split())
        # Remove quotes and convert to lowercase
        return normalized.replace('"', "").casefold()

    def _get_coworker(self, coworker: str | None, **kwargs) -> str | None:
        coworker = coworker or kwargs.get("co_worker") or kwargs.get("coworker")
        if coworker:
            is_list = coworker.startswith("[") and coworker.endswith("]")
            if is_list:
                coworker = coworker[1:-1].split(",")[0]
        return coworker

    def _execute(
        self, agent_name: str | None, task: str, context: str | None = None
    ) -> str:
        """
        Execute delegation to an agent with case-insensitive and whitespace-tolerant matching.

        Args:
            agent_name: Name/role of the agent to delegate to (case-insensitive)
            task: The specific question or task to delegate
            context: Optional additional context for the task execution

        Returns:
            str: The execution result from the delegated agent or an error message
                 if the agent cannot be found
        """
        try:
            if agent_name is None:
                agent_name = ""
                logger.debug("No agent name provided, using empty string")

            # It is important to remove the quotes from the agent name.
            # The reason we have to do this is because less-powerful LLM's
            # have difficulty producing valid JSON.
            # As a result, we end up with invalid JSON that is truncated like this:
            # {"task": "....", "coworker": "....
            # when it should look like this:
            # {"task": "....", "coworker": "...."}
            sanitized_name = self.sanitize_agent_name(agent_name)
            logger.debug(
                f"Sanitized agent name from '{agent_name}' to '{sanitized_name}'"
            )

            available_agents = [agent.role for agent in self.agents]
            logger.debug(f"Available agents: {available_agents}")

            agent = [  # type: ignore # Incompatible types in assignment (expression has type "list[BaseAgent]", variable has type "str | None")
                available_agent
                for available_agent in self.agents
                if self.sanitize_agent_name(available_agent.role) == sanitized_name
            ]
            logger.debug(
                f"Found {len(agent)} matching agents for role '{sanitized_name}'"
            )
        except (AttributeError, ValueError) as e:
            # Handle specific exceptions that might occur during role name processing
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join(
                    [
                        f"- {self.sanitize_agent_name(agent.role)}"
                        for agent in self.agents
                    ]
                ),
                error=str(e),
            )

        if not agent:
            # No matching agent found after sanitization
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join(
                    [
                        f"- {self.sanitize_agent_name(agent.role)}"
                        for agent in self.agents
                    ]
                ),
                error=f"No agent found with role '{sanitized_name}'",
            )

        selected_agent = agent[0]
        try:
            task_with_assigned_agent = Task(
                description=task,
                agent=selected_agent,
                expected_output=selected_agent.i18n.slice("manager_request"),
                i18n=selected_agent.i18n,
            )
            logger.debug(
                f"Created task for agent '{self.sanitize_agent_name(selected_agent.role)}': {task}"
            )
            return selected_agent.execute_task(task_with_assigned_agent, context)
        except Exception as e:
            # Handle task creation or execution errors
            return self.i18n.errors("agent_tool_execution_error").format(
                agent_role=self.sanitize_agent_name(selected_agent.role), error=str(e)
            )
