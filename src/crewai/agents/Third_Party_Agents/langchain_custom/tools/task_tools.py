from typing import List
from pydantic import BaseModel, Field

from crewai.agents.third_party_agents.utilities.base_agent_tool import BaseAgentTools
from crewai.agents.third_party_agents.base_agent import BaseAgent

from langchain.tools import StructuredTool


class LangchainCustomTools(BaseAgentTools, BaseModel):
    """
    Custom tools for Langchain Integrations to handle delegation and questioning amongst the crew.
    """

    agents: List[BaseAgent] = Field(description="List of agents in this crew.")

    def __init__(self, agents: List[BaseAgent], *args, **kwargs):
        super().__init__(agents=agents, *args, **kwargs)

    def tools(self) -> List[StructuredTool]:
        """
        Generate a list of structured tools for delegation and questioning.

        Returns:
            List[StructuredTool]: A list of structured tools.
        """
        coworkers = f"[{', '.join([f'{agent.role}' for agent in self.agents])}]"
        tools = [
            StructuredTool.from_function(
                func=self.delegate_work,
                name="Delegate-work-to-coworker",
                description=self.i18n.tools("delegate_work").format(
                    coworkers=coworkers
                ),
            ),
            StructuredTool.from_function(
                func=self.ask_question,
                name="Ask-question-to-coworker",
                description=self.i18n.tools("ask_question").format(coworkers=coworkers),
            ),
        ]
        return tools
