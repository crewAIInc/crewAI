from typing import List

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities import I18N


class AgentTools(BaseModel):
    """Default tools around agent delegation"""

    agents: List[Agent] = Field(description="List of agents in this crew.")
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")

    def tools(self):
        tools = [
            StructuredTool.from_function(
                func=self.delegate_work,
                name="Delegate work to co-worker",
                description=self.i18n.tools("delegate_work").format(
                    coworkers=f"[{', '.join([f'{agent.role}' for agent in self.agents])}]"
                ),
            ),
            StructuredTool.from_function(
                func=self.ask_question,
                name="Ask question to co-worker",
                description=self.i18n.tools("ask_question").format(
                    coworkers=f"[{', '.join([f'{agent.role}' for agent in self.agents])}]"
                ),
            ),
        ]
        return tools

    def delegate_work(self, coworker: str, task: str, context: str):
        """Useful to delegate a specific task to a coworker passing all necessary context and names."""
        return self._execute(coworker, task, context)

    def ask_question(self, coworker: str, question: str, context: str):
        """Useful to ask a question, opinion or take from a coworker passing all necessary context and names."""
        return self._execute(coworker, question, context)

    def _execute(self, agent, task, context):
        """Execute the command."""
        try:
            agent = [
                available_agent
                for available_agent in self.agents
                if available_agent.role.casefold().strip() == agent.casefold().strip()
            ]
        except:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers="\n".join(
                    [f"- {agent.role.casefold()}" for agent in self.agents]
                )
            )

        if not agent:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers="\n".join(
                    [f"- {agent.role.casefold()}" for agent in self.agents]
                )
            )

        agent = agent[0]
        task = Task(
            description=task,
            agent=agent,
            expected_output="Your best answer to your coworker asking you this, accounting for the context shared.",
        )
        return agent.execute_task(task, context)
