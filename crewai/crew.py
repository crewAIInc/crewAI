import json
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel, ConfigDict, Field, InstanceOf, Json, UUID4
)
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler
from crewai.process import Process
from crewai.task import Task
from crewai.tools.agent_tools import AgentTools

class Crew(BaseModel):
    """
    Represents a group of agents, defining how they should collaborate and the tasks they should perform.

    Attributes:
        tasks: List of tasks assigned to the crew.
        agents: List of agents part of this crew.
        process: The process flow that the crew will follow (e.g., sequential).
        verbose: Indicates the verbosity level for logging during execution.
        config: Configuration settings for the crew.
        cache_handler: Handles caching for the crew's operations.
        id: A unique identifier for the crew instance.
    """

    __hash__ = object.__hash__
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[Task] = Field(default_factory=list)
    agents: List[Agent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: Union[int, bool] = Field(default=0)
    config: Optional[Union[Json, Dict[str, Any]]] = Field(default=None)
    cache_handler: Optional[InstanceOf[CacheHandler]] = Field(
        default=CacheHandler()
    )
    id: UUID4 = Field(
        default_factory=uuid.uuid4, frozen=True
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        """Prevent manual setting of the 'id' field by users."""
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "The 'id' field cannot be set by the user.", {}
            )

    @classmethod
    @field_validator("config", mode="before")
    def check_config_type(cls, v: Union[Json, Dict[str, Any]]):
        """Ensures the 'config' field is a valid JSON or dictionary."""
        if isinstance(v, Json):
            return json.loads(v)
        return v

    @model_validator(mode="after")
    def check_config(self):
        """Validates that the crew is properly configured with agents and tasks."""
        if not self.config and not self.tasks and not self.agents:
            raise PydanticCustomError(
                "missing_keys", "Either 'agents' and 'tasks' need to be set or 'config'.", {}
            )

        if self.config:
            self._setup_from_config()

        if self.agents:
            for agent in self.agents:
                agent.set_cache_handler(self.cache_handler)
        return self

    def _setup_from_config(self):
        """Initializes agents and tasks from the provided config."""
        if not self.config.get("agents") or not self.config.get("tasks"):
            raise PydanticCustomError(
                "missing_keys_in_config", "Config should have 'agents' and 'tasks'.", {}
            )

        self.agents = [Agent(**agent) for agent in self.config["agents"]]
        self.tasks = [self._create_task(task) for task in self.config["tasks"]]

    def _create_task(self, task_config):
        """Creates a task instance from its configuration."""
        task_agent = next(agt for agt in self.agents if agt.role == task_config["agent"])
        del task_config["agent"]
        return Task(**task_config, agent=task_agent)

    def kickoff(self) -> str:
        """Starts the crew to work on its assigned tasks."""
        for agent in self.agents:
            agent.cache_handler = self.cache_handler

        if self.process == Process.sequential:
            return self._sequential_loop()

    def _sequential_loop(self) -> str:
        """Executes tasks sequentially and returns the final output."""
        task_output = None
        for task in self.tasks:
            self._prepare_and_execute_task(task)
            task_output = task.execute(task_output)
        return task_output

    def _prepare_and_execute_task(self, task):
        """Prepares and logs information about the task being executed."""
        if task.agent.allow_delegation:
            task.tools += AgentTools(agents=self.agents).tools()

        self._log("debug", f"Working Agent: {task.agent.role}")
        self._log("info", f"Starting Task: {task.description}")

    def _log(self, level, message):
        """Logs a message at the specified verbosity level."""
        level_map = {"debug": 1, "info": 2}
        verbose_level = 2 if isinstance(self.verbose, bool) and self.verbose else self.verbose
        if verbose_level and level_map[level] <= verbose_level:
            print(message)
