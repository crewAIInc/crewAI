from textwrap import dedent
from typing import List

from langchain.tools import Tool
from pydantic import BaseModel, Field

from crewai.agent import Agent


class AgentTools(BaseModel):
    """Default tools around agent delegation"""

    agents: List[Agent] = Field(description="List of agents in this crew.")

    def tools(self):
        return [
            Tool.from_function(
                func=self.delegate_work,
                name="Delegate work to co-worker",
                description=dedent(
                    f"""\
                Useful to delegate a specific task to one of the
				following co-workers: [{', '.join([agent.role for agent in self.agents])}].
				The input to this tool should be a pipe (|) separated text of length
				three, representing the co-worker you want to ask it to (one of the options),
                the task and all actual context you have for the task.
                For example, `coworker|task|context`.
				"""
                ),
            ),
            Tool.from_function(
                func=self.ask_question,
                name="Ask question to co-worker",
                description=dedent(
                    f"""\
                Useful to ask a question, opinion or take from on
				of the following co-workers: [{', '.join([agent.role for agent in self.agents])}].
				The input to this tool should be a pipe (|) separated text of length
				three, representing the co-worker you want to ask it to (one of the options),
                the question and all actual context you have for the question.
                For example, `coworker|question|context`.
				"""
                ),
            ),
        ]

    def delegate_work(self, command):
        """Useful to delegate a specific task to a coworker."""
        return self.__execute(command)

    def ask_question(self, command):
        """Useful to ask a question, opinion or take from a coworker."""
        return self.__execute(command)

    def __execute(self, command):
        """Execute the command."""
        try:
            agent, task, context = command.split("|")
        except ValueError:
            return "\nError executing tool. Missing exact 3 pipe (|) separated values. For example, `coworker|task|context`. I need to make sure to pass context as context\n"

        if not agent or not task or not context:
            return "\nError executing tool. Missing exact 3 pipe (|) separated values. For example, `coworker|task|context`. I need to make sure to pass context as context.\n"

        agent = [
            available_agent
            for available_agent in self.agents
            if available_agent.role == agent
        ]

        if len(agent) == 0:
            return f"\nError executing tool. Co-worker mentioned on the Action Input not found, it must to be one of the following options: {', '.join([agent.role for agent in self.agents])}.\n"

        agent = agent[0]
        result = agent.execute_task(task, context)
        return result
