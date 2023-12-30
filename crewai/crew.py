import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, Json, field_validator, model_validator
from pydantic_core import PydanticCustomError

from .agent import Agent
from .process import Process
from .task import Task
from .tools.agent_tools import AgentTools


class Crew(BaseModel):
    """Class that represents a group of agents, how they should work together and their tasks."""

    tasks: List[Task] = Field(description="List of tasks", default_factory=list)
    agents: List[Agent] = Field(
        description="List of agents in this crew.", default_factory=list
    )
    process: Process = Field(
        description="Process that the crew will follow.", default=Process.sequential
    )
    verbose: bool = Field(
        description="Verbose mode for the Agent Execution", default=False
    )
    config: Optional[Union[Json, Dict[str, Any]]] = Field(
        description="Configuration of the crew.", default=None
    )

    @classmethod
    @field_validator("config", mode="before")
    def check_config_type(cls, v: Union[Json, Dict[str, Any]]):
        if isinstance(v, Json):
            return json.loads(v)
        return v

    @model_validator(mode="after")
    def check_config(self):
        if not self.config and not self.tasks and not self.agents:
            raise PydanticCustomError(
                "missing_keys", "Either agents and task need to be set or config.", {}
            )

        if self.config:
            if not self.config.get("agents") or not self.config.get("tasks"):
                raise PydanticCustomError(
                    "missing_keys_in_config", "Config should have agents and tasks", {}
                )

            self.agents = [Agent(**agent) for agent in self.config["agents"]]

            tasks = []
            for task in self.config["tasks"]:
                task_agent = [agt for agt in self.agents if agt.role == task["agent"]][
                    0
                ]
                del task["agent"]
                tasks.append(Task(**task, agent=task_agent))

            self.tasks = tasks
        return self

    def kickoff(self) -> str:
        """Kickoff the crew to work on its tasks.

        Returns:
            Output of the crew for each task.
        """
        if self.process == Process.sequential:
            return self.__sequential_loop()

    def __sequential_loop(self) -> str:
        """Loop that executes the sequential process.

        Returns:
            Output of the crew.
        """
        task_outcome = None
        for task in self.tasks:
            # Add delegation tools to the task if the agent allows it
            if task.agent.allow_delegation:
                tools = AgentTools(agents=self.agents).tools()
                task.tools += tools

            self.__log(f"\nWorking Agent: {task.agent.role}")
            self.__log(f"Starting Task: {task.description} ...")

            task_outcome = task.execute(task_outcome)

            self.__log(f"Task output: {task_outcome}")

        return task_outcome

    def __log(self, message):
        if self.verbose:
            print(message)
