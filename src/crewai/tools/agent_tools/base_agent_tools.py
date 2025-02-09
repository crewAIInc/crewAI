import logging
from typing import Optional

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

    def _get_coworker(self, coworker: Optional[str], **kwargs) -> Optional[str]:
        coworker = coworker or kwargs.get("co_worker") or kwargs.get("coworker")
        if coworker:
            is_list = coworker.startswith("[") and coworker.endswith("]")
            if is_list:
                coworker = coworker[1:-1].split(",")[0]
        return coworker
    
    def _execute(
        self,
        agent_name: Optional[str],
        task: str,
        context: Optional[str] = None
    ) -> str:
        try:
            print("\n=== Delegating Work ===")
            
            if agent_name is None:
                agent_name = ""
                logger.debug("No agent name provided, using empty string")
            sanitized_name = self.sanitize_agent_name(agent_name)
            target_agent = next(
                (agent for agent in self.agents if self.sanitize_agent_name(agent.role) == sanitized_name),
                None,
            )
            
            if not target_agent:
                return self.i18n.errors("agent_tool_unexisting_coworker").format(
                    coworkers="\n".join(
                        [f"- {self.sanitize_agent_name(agent.role)}" for agent in self.agents]
                    ),
                    error=f"No agent found with role '{sanitized_name}'"
                )
                
            print(f"Delegating task to: {target_agent.role}")
            
            new_task = Task(
                description=task,
                agent=target_agent,
                expected_output=target_agent.i18n.slice("manager_request"),
                i18n=target_agent.i18n,
            )
            
            tools = target_agent.crew._prepare_tools(
                target_agent,
                new_task,
                target_agent.tools or [],
            )
            
            result = target_agent.execute_task(new_task, context, tools)
            print("\n=== Delegation Complete ===")
            
            return result
        except Exception as e:
            return self.i18n.errors("agent_tool_execution_error").format(
                agent_role=sanitized_name,
                error=str(e)
            )