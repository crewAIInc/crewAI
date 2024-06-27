from typing import List, Union

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
                name="Delegate work to coworker",
                description=self.i18n.tools("delegate_work").format(
                    coworkers=f"[{', '.join([f'{agent.role}' for agent in self.agents])}]"
                ),
            ),
            StructuredTool.from_function(
                func=self.ask_question,
                name="Ask question to coworker",
                description=self.i18n.tools("ask_question").format(
                    coworkers=f"[{', '.join([f'{agent.role}' for agent in self.agents])}]"
                ),
            ),
        ]
        return tools

    def delegate_work(
        self,
        task: str,
        context: Union[str, None] = None,
        coworker: Union[str, None] = None,
        **kwargs,
    ):
        """Useful to delegate a specific task to a coworker passing all necessary context and names."""
        coworker = coworker or kwargs.get("co_worker") or kwargs.get("coworker")
        if coworker:
            is_list = coworker.startswith("[") and coworker.endswith("]")
            if is_list:
                coworker = coworker[1:-1].split(",")[0]
        return self._execute(coworker, task, context)

    def ask_question(
        self,
        question: str,
        context: Union[str, None] = None,
        coworker: Union[str, None] = None,
        **kwargs,
    ):
        """Useful to ask a question, opinion or take from a coworker passing all necessary context and names."""
        coworker = coworker or kwargs.get("co_worker") or kwargs.get("coworker")
        if coworker:
            is_list = coworker.startswith("[") and coworker.endswith("]")
            if is_list:
                coworker = coworker[1:-1].split(",")[0]
        return self._execute(coworker, question, context)

    def _execute(self, agent: Union[str, None], task: str, context: Union[str, None]):
        """Execute the command."""
        try:
            if agent is None:
                agent = ""

            # It is important to remove the quotes from the agent name.
            # The reason we have to do this is because less-powerful LLM's
            # have difficulty producing valid JSON.
            # As a result, we end up with invalid JSON that is truncated like this:
            # {"task": "....", "coworker": "....
            # when it should look like this:
            # {"task": "....", "coworker": "...."}
            agent_name = agent.casefold().replace('"', "").replace("\n", "")

            agent = [
                available_agent
                for available_agent in self.agents
                if available_agent.role.casefold().replace("\n", "") == agent_name
            ]
        except Exception as _:
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
