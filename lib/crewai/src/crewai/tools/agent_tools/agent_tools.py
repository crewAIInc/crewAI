from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N

from .ask_question_tool import AskQuestionTool
from .delegate_work_tool import DelegateWorkTool


class AgentTools:
    """Manager class for agent-related tools"""

    def __init__(self, agents: list[BaseAgent], i18n: I18N | None = None):
        self.agents = agents
        self.i18n = i18n if i18n is not None else I18N()

    def tools(self) -> list[BaseTool]:
        """Get all available agent tools"""
        coworkers = ", ".join([f"{agent.role}" for agent in self.agents])

        delegate_tool = DelegateWorkTool(
            agents=self.agents,
            i18n=self.i18n,
            description=self.i18n.tools("delegate_work").format(coworkers=coworkers),  # type: ignore
        )

        ask_tool = AskQuestionTool(
            agents=self.agents,
            i18n=self.i18n,
            description=self.i18n.tools("ask_question").format(coworkers=coworkers),  # type: ignore
        )

        return [delegate_tool, ask_tool]
