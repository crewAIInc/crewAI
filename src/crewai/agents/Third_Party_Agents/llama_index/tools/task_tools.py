from typing import List
from pydantic import BaseModel, Field

from crewai.agents.third_party_agents.utilities.base_agent_tool import BaseAgentTools
from crewai.agents.third_party_agents.base_agent import BaseAgent

from llama_index.core.tools import FunctionTool


class LlamaAgentTools(BaseAgentTools, BaseModel):
    """
    A class to manage tools for Llama agents.
    """

    agents: List[BaseAgent] = Field(description="List of agents in this crew.")

    def __init__(self, agents: List[BaseAgent], *args, **kwargs):
        """
        Initialize the LlamaAgentTools with a list of agents.
        """
        super().__init__(agents=agents, *args, **kwargs)

    def tools(self) -> List[FunctionTool]:
        """
        Generate a list of FunctionTool instances for the agents.

        Returns:
            List[FunctionTool]: A list of tools for delegating work and asking questions.
        """
        coworkers = f"[{', '.join(agent.role for agent in self.agents)}]"
        tools = [
            FunctionTool.from_defaults(
                fn=self.delegate_work,
                name="Delegate-work-to-coworker",
                description=self.i18n.tools("delegate_work").format(
                    coworkers=coworkers
                ),
            ),
            FunctionTool.from_defaults(
                fn=self.ask_question,
                name="Ask-question-to-coworker",
                description=self.i18n.tools("ask_question").format(coworkers=coworkers),
            ),
        ]
        return tools
