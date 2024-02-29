from typing import List

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities import I18N


class AgentTools(BaseModel):
    """
    This class represents the default tools for agent delegation. It contains a list of agents and internationalization settings.
    It also provides methods to delegate work to a coworker and ask a question to a coworker.
    """

    agents: List[Agent] = Field(description="List of agents in this crew.")
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")

    def tools(self):
        """
        This method returns a list of StructuredTool objects. Each tool is created from a function (delegate_work or ask_question)
        and has a name and a description.
        """
        return [
            StructuredTool.from_function(
                func=self.delegate_work,
                name="Delegate work to co-worker",
                description=self.i18n.tools("delegate_work").format(
                    coworkers="\n".join([f"- {agent.role}" for agent in self.agents])
                ),
            ),
            StructuredTool.from_function(
                func=self.ask_question,
                name="Ask question to co-worker",
                description=self.i18n.tools("ask_question").format(
                    coworkers="\n".join([f"- {agent.role}" for agent in self.agents])
                ),
            ),
        ]

    def delegate_work(self, coworker: str, task: str, context: str):
        """
        This method is used to delegate a specific task to a coworker. It takes the coworker's name, the task, and the context as input.
        It then calls the _execute method to perform the delegation.
        """
        return self._execute(coworker, task, context)

    def ask_question(self, coworker: str, question: str, context: str):
        """
        This method is used to ask a question, opinion or take from a coworker. It takes the coworker's name, the question, and the context as input.
        It then calls the _execute method to perform the action.
        """
        return self._execute(coworker, question, context)

    def _execute(self, agent, task, context):
        """
        This method is used to execute the command. It first checks if the agent exists in the list of agents.
        If the agent does not exist, it returns an error message. If the agent exists, it creates a new Task object and calls the execute_task method on the agent.
        """
        try:
            agent = [
                available_agent
                for available_agent in self.agents
                if available_agent.role.strip().lower() == agent.strip().lower()
            ]
        except:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers="\n".join([f"- {agent.role}" for agent in self.agents])
            )

        if not agent:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers="\n".join([f"- {agent.role}" for agent in self.agents])
            )

        agent = agent[0]
        task = Task(description=task, agent=agent)
        return agent.execute_task(task, context)
