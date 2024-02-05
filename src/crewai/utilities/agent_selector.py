from typing import List

from pydantic import BaseModel, Field, InstanceOf, PrivateAttr, model_validator

from crewai.agent import Agent


class AgentSelector(BaseModel):
    """
    We take a list of Agent descriptions and a specific task and provide return the best matching agent for the given task
    Attributes:
        agents: List of agents considered for the crew.
    """

    agents: List[InstanceOf[Agent]] = Field(default_factory=list)
    _persona: Agent = PrivateAttr()
    _persona_task: str = PrivateAttr()

    @model_validator(mode="after")
    def set_private_attrs(self) -> "AgentSelector":
        """
        Initialize the AgentSelector
        """

        self._persona = Agent(
            role="agent_selector",
            goal="select_agent",
            backstory="I am an agent selector",
            allow_delegation=False,
        )
        # Note: the _persona_task is a first draft for the prompt to the LLM it is currently only tested against
        # * OpenAI gpt-4
        self._persona_task = (
            f"Given the tasks and the agent profiles, select the best matching agent, "
            f"do not explain your reasoning,just return the name of the agent\n"
        )
        return self

    def _build_task_description(self, parent_task: str) -> str:
        """
        Returns the task description for the AgentSelector to find the best matching agent. By joining a
            static description and the provided agents as well as the tasks

        Args:
            parent_task: the tasks we need an agent for
        Returns:
            the agent selector tasks
        """
        return (
            self._persona_task
            + f'Task:\n{parent_task}\n\nAgent name: "Agent description"'
            + "\n".join(
                [
                    f'* {agent.role}: "{agent.goal}, {agent.backstory}"'
                    for agent in self.agents
                ]
            )
        )

    def lookup_agent(self, task: str) -> Agent:
        """
        Returns the best matching agent for the given task

        Args:
            task: the task we need an agent for
        Returns:
            agent: the best matching agent
        """
        llm_response = self._persona.execute_task(
            self._build_task_description(parent_task=task), tools=None
        )
        found_agent = [agent for agent in self.agents if agent.role == llm_response]
        if not found_agent:
            raise Exception(
                f"No agent was provided for the task {task}, and no agent was found suitable for the task, review task or switch Crew using a specific process that support that, like hierarchical. "
            )
        return found_agent[0]
