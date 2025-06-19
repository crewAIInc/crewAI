from typing import Optional

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

    def tools(self, delegating_agent: Optional[BaseAgent] = None) -> list[BaseTool]:
        """Get all available agent tools, filtered by delegating agent's allowed_agents if specified"""
        available_agents = self._filter_allowed_agents(delegating_agent)
        
        if not available_agents:
            return []
        
        coworkers = ", ".join([f"{agent.role}" for agent in available_agents])

        delegate_tool = DelegateWorkTool(
            agents=available_agents,
            i18n=self.i18n,
            description=self.i18n.tools("delegate_work").format(coworkers=coworkers),  # type: ignore
        )

        ask_tool = AskQuestionTool(
            agents=available_agents,
            i18n=self.i18n,
            description=self.i18n.tools("ask_question").format(coworkers=coworkers),  # type: ignore
        )

        return [delegate_tool, ask_tool]

    def _filter_allowed_agents(self, delegating_agent: Optional[BaseAgent]) -> list[BaseAgent]:
        """Filter agents based on the delegating agent's allowed_agents list"""
        if delegating_agent is None:
            return self.agents
        
        if not hasattr(delegating_agent, 'allowed_agents') or delegating_agent.allowed_agents is None:
            return self.agents
        
        if not delegating_agent.allowed_agents:
            return []
        
        filtered_agents = []
        for agent in self.agents:
            for allowed in delegating_agent.allowed_agents:
                if isinstance(allowed, str):
                    if agent.role.strip().lower() == allowed.strip().lower():
                        filtered_agents.append(agent)
                        break
                elif isinstance(allowed, BaseAgent):
                    if agent is allowed:
                        filtered_agents.append(agent)
                        break
        
        return filtered_agents
