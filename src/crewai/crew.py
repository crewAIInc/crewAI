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
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler
from crewai.process import Process
from crewai.task import Task
from crewai.tools.agent_tools import AgentTools
from crewai.utilities import I18N, Logger, RPMController


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
        max_rpm: Maximum number of requests per minute for the crew execution to be respected.
        id: A unique identifier for the crew instance.
    """

    __hash__ = object.__hash__
    _rpm_controller: RPMController = PrivateAttr()
    _logger: Logger = PrivateAttr()
    _cache_handler: Optional[InstanceOf[CacheHandler]] = PrivateAttr(
        default=CacheHandler()
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[Task] = Field(default_factory=list)
    agents: List[Agent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: Union[int, bool] = Field(default=0)
    config: Optional[Union[Json, Dict[str, Any]]] = Field(default=None)
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the crew execution to be respected.",
    )
    language: str = Field(
        default="en",
        description="Language used for the crew, defaults to English.",
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
        return json.loads(v) if isinstance(v, Json) else v

    @model_validator(mode="after")
    def set_private_attrs(self):
        self._cache_handler = CacheHandler()
        self._logger = Logger(self.verbose)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        return self

    @model_validator(mode="after")
    def check_config(self):
        """Validates that the crew is properly configured with agents and tasks."""
        if not self.config and not self.tasks and not self.agents:
            raise PydanticCustomError(
                "missing_keys",
                "Either 'agents' and 'tasks' need to be set or 'config'.",
                {},
            )

        if self.config:
            self._setup_from_config()

        if self.agents:
            for agent in self.agents:
                agent.set_cache_handler(self._cache_handler)
                agent.set_rpm_controller(self._rpm_controller)
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
        task_agent = next(
            agt for agt in self.agents if agt.role == task_config["agent"]
        )
        del task_config["agent"]
        return Task(**task_config, agent=task_agent)

    def kickoff(self) -> str:
        """Starts the crew to work on its assigned tasks."""
        for agent in self.agents:
            agent.i18n = I18N(language=self.language)

        if self.process == Process.sequential:
            return self._sequential_loop()

    def _sequential_loop(self) -> str:
        """Executes tasks sequentially and returns the final output."""
        task_output = None
        for task in self.tasks:
            self._prepare_and_execute_task(task)
            task_output = task.execute(task_output)
            self._logger.log(
                "debug", f"[{task.agent.role}] Task output: {task_output}\n\n"
            )

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()
        return task_output

    def _prepare_and_execute_task(self, task):
        """Prepares and logs information about the task being executed."""
        if task.agent.allow_delegation:
            task.tools += AgentTools(agents=self.agents).tools()

        self._logger.log("debug", f"Working Agent: {task.agent.role}")
        self._logger.log("info", f"Starting Task: {task.description}")
