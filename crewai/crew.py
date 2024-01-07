import json
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    Json,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler
from crewai.process import Process
from crewai.task import Task
from crewai.tools.agent_tools import AgentTools


class Crew(BaseModel):
    """Class that represents a group of agents, how they should work together and their tasks."""

    __hash__ = object.__hash__
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[Task] = Field(description="List of tasks", default_factory=list)
    agents: List[Agent] = Field(
        description="List of agents in this crew.", default_factory=list
    )
    process: Process = Field(
        description="Process that the crew will follow.", default=Process.sequential
    )
    verbose: Union[int, bool] = Field(
        description="Verbose mode for the Agent Execution", default=0
    )
    config: Optional[Union[Json, Dict[str, Any]]] = Field(
        description="Configuration of the crew.", default=None
    )
    cache_handler: Optional[InstanceOf[CacheHandler]] = Field(
        default=CacheHandler(), description="An instance of the CacheHandler class."
    )
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
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

        if self.agents:
            for agent in self.agents:
                agent.set_cache_handler(self.cache_handler)
        return self

    def kickoff(self) -> str:
        """Kickoff the crew to work on its tasks.

        Returns:
            Output of the crew for each task.
        """
        for agent in self.agents:
            agent.cache_handler = self.cache_handler

        if self.process == Process.sequential:
            return self.__sequential_loop()

    def __sequential_loop(self) -> str:
        """Loop that executes the sequential process.

        Returns:
            Output of the crew.
        """
        task_output = None
        for task in self.tasks:
            # Add delegation tools to the task if the agent allows it
            if task.agent.allow_delegation:
                agent_tools = AgentTools(agents=self.agents).tools()
                task.tools += agent_tools

            self.__log("debug", f"Working Agent: {task.agent.role}")
            self.__log("info", f"Starting Task: {task.description}")

            task_output = task.execute(task_output)
            self.__log(
                "debug", f"\n\n[{task.agent.role}] Task output: {task_output}\n\n"
            )
        return task_output

    def __log(self, level, message):
        """Log a message"""
        level_map = {"debug": 1, "info": 2}
        verbose_level = (
            2 if isinstance(self.verbose, bool) and self.verbose else self.verbose
        )
        if verbose_level and level_map[level] <= verbose_level:
            print(message)
