from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N
import logging

from .ask_question_tool import AskQuestionTool
from .delegate_work_tool import DelegateWorkTool

logger = logging.getLogger(__name__)


class AgentTools:
    """Manager class for agent-related tools"""

    def __init__(self, agents: list[BaseAgent], i18n: I18N = I18N()):
        self.agents = agents
        self.i18n = i18n

    def tools(self) -> list[BaseTool]:
        """Get all available agent tools"""
        coworkers = ", ".join([f"{agent.role}" for agent in self.agents])
        
        # Check encryption capabilities of agents
        encryption_enabled_agents = [
            agent for agent in self.agents 
            if hasattr(agent, 'security_config') 
            and agent.security_config 
            and getattr(agent.security_config, 'encrypted_communication', False)
        ]
        
        if encryption_enabled_agents:
            logger.info(f"Creating agent communication tools with encryption support for {len(encryption_enabled_agents)} agent(s): {[agent.role for agent in encryption_enabled_agents]}")
        else:
            logger.debug(f"Creating agent communication tools without encryption (no agents have encrypted_communication enabled)")

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
