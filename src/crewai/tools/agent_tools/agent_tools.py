from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N

from .ask_question_tool import AskQuestionTool
from .delegate_work_tool import DelegateWorkTool


class AgentTools:
    """Manager class for agent-related tools"""

    def __init__(self, agents: list[BaseAgent], i18n: I18N = I18N(), task=None):
        self.agents = agents
        self.i18n = i18n
        self.task = task

    def tools(self) -> list[BaseTool]:
        """Get all available agent tools"""
        # Format coworkers list based on agents and task context
        if len(self.agents) == 1:
            coworkers = self.agents[0].role
        elif self.task and hasattr(self.task, 'async_execution') and self.task.async_execution and hasattr(self.task, 'agent') and self.task.agent:
            # For async tasks with a specific agent, only show that agent
            coworkers = self.task.agent.role
        else:
            # Show all agents for non-async tasks or when no specific agent is assigned
            coworkers = ", ".join([agent.role for agent in self.agents])

        # Ensure coworkers list doesn't have extra spaces or newlines
        coworkers = coworkers.strip()

        delegate_tool = DelegateWorkTool(
            agents=self.agents,
            i18n=self.i18n,
            description=f"Delegate a specific task to one of the following coworkers: {coworkers}\n",
        )

        ask_tool = AskQuestionTool(
            agents=self.agents,
            i18n=self.i18n,
            description=f"Ask a specific question to one of the following coworkers: {coworkers}\n",
        )

        return [delegate_tool, ask_tool]
