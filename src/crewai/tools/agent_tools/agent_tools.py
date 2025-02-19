from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N, Logger

from .ask_question_tool import AskQuestionTool
from .delegate_work_tool import DelegateWorkTool

# Tool name constants
DELEGATE_WORK_TOOL = "Delegate Work"
ASK_QUESTION_TOOL = "Ask Question"


class AgentTools:
    """Manager class for agent-related tools"""

    def __init__(self, agents: list[BaseAgent], i18n: I18N = I18N()):
        self.agents = agents
        self.i18n = i18n
        self._logger = Logger()

    def tools(self) -> list[BaseTool]:
        """Get all available agent tools"""
        coworkers = ", ".join([f"{agent.role}" for agent in self.agents])

        self._logger.log(
            "debug", f"Creating delegation tools for agents: {coworkers}", color="blue"
        )

        delegate_tool = DelegateWorkTool(
            agents=self.agents,
            i18n=self.i18n,
            description=self.i18n.tools("delegate_work").format(coworkers=coworkers),  # type: ignore
            name=DELEGATE_WORK_TOOL,  # Using constant for consistency
        )

        ask_tool = AskQuestionTool(
            agents=self.agents,
            i18n=self.i18n,
            description=self.i18n.tools("ask_question").format(coworkers=coworkers),  # type: ignore
            name=ASK_QUESTION_TOOL,  # Using constant for consistency
        )

        return [delegate_tool, ask_tool]
