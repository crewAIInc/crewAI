from typing import List, Set

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N

from .ask_question_tool import AskQuestionTool
from .delegate_work_tool import DelegateWorkTool


class AgentTools:
    """Manager class for agent-related tools"""

    def __init__(self, agents: list[BaseAgent], i18n: I18N = I18N()):
        self.agents = agents
        self.i18n = i18n

    def tools(self) -> list[BaseTool]:
        """Get all available agent tools"""
        coworkers = ", ".join([f"{agent.role}" for agent in self.agents])

        delegate_tool = DelegateWorkTool(
            agents=self.agents,
            i18n=self.i18n,
            description=self.i18n.tools("delegate_work").format(coworkers=coworkers),  # type: ignore
        )
        delegate_tool._agent_tools = self._get_all_agent_tools()

        ask_tool = AskQuestionTool(
            agents=self.agents,
            i18n=self.i18n,
            description=self.i18n.tools("ask_question").format(coworkers=coworkers),  # type: ignore
        )
        ask_tool._agent_tools = self._get_all_agent_tools()

        return [delegate_tool, ask_tool]

    def _get_all_agent_tools(self) -> list[BaseTool]:
        """
        Get all tools from all agents for recursive invocation.

        Returns:
            list[BaseTool]: A deduplicated list of all tools from all agents.
        """
        seen_tools: Set[int] = set()
        unique_tools: List[BaseTool] = []

        for agent in self.agents:
            if agent.tools:
                for tool in agent.tools:
                    tool_id = id(tool)
                    if tool_id not in seen_tools:
                        seen_tools.add(tool_id)
                        unique_tools.append(tool)

        return unique_tools
