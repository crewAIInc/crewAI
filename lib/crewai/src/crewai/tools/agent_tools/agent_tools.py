from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool
from crewai.tools.agent_tools.delegate_work_tool import DelegateWorkTool
from crewai.utilities.i18n import get_i18n


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.i18n import I18N


class AgentTools:
    """Manager class for agent-related tools"""

    def __init__(self, agents: list[BaseAgent], i18n: I18N | None = None) -> None:
        self.agents = agents
        self.i18n = i18n if i18n is not None else get_i18n()

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
