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
from crewai.telemtry import Telemetry
from crewai.tools.agent_tools import AgentTools
from crewai.utilities import I18N, Logger, RPMController


class Crew(BaseModel):
    """
    Represents a group of agents, defining how they should collaborate and the tasks they should perform.

    Attributes:
        tasks: List of tasks assigned to the crew.
        agents: List of agents part of this crew.
        manager_llm: The language model that will run manager agent.
        function_calling_llm: The language model that will run the tool calling for all the agents.
        process: The process flow that the crew will follow (e.g., sequential).
        verbose: Indicates the verbosity level for logging during execution.
        config: Configuration settings for the crew.
        max_rpm: Maximum number of requests per minute for the crew execution to be respected.
        id: A unique identifier for the crew instance.
        full_output: Whether the crew should return the full output with all tasks outputs or just the final output.
        step_callback: Callback to be executed after each step for every agents execution.
        share_crew: Whether you want to share the complete crew infromation and execution with crewAI to make the library better, and allow us to train models.
        _cache_handler: Handles caching for the crew's operations.
    """

    __hash__ = object.__hash__  # type: ignore
    _execution_span: Any = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr()
    _logger: Logger = PrivateAttr()
    _cache_handler: InstanceOf[CacheHandler] = PrivateAttr(default=CacheHandler())
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[Task] = Field(default_factory=list)
    agents: List[Agent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: Union[int, bool] = Field(default=0)
    full_output: Optional[bool] = Field(
        default=False,
        description="Whether the crew should return the full output with all tasks outputs or just the final output.",
    )
    manager_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    function_calling_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    config: Optional[Union[Json, Dict[str, Any]]] = Field(default=None)
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    share_crew: Optional[bool] = Field(default=False)
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step for all agents execution.",
    )
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

    @field_validator("config", mode="before")
    @classmethod
    def check_config_type(
        cls, v: Union[Json, Dict[str, Any]]
    ) -> Union[Json, Dict[str, Any]]:
        """Validates that the config is a valid type.
        Args:
            v: The config to be validated.
        Returns:
            The config if it is valid.
        """

        # TODO: Improve typing
        return json.loads(v) if isinstance(v, Json) else v  # type: ignore

    @model_validator(mode="after")
    def set_private_attrs(self) -> "Crew":
        """Set private attributes."""
        self._cache_handler = CacheHandler()
        self._logger = Logger(self.verbose)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()
        self._telemetry.crew_creation(self)
        return self

    @model_validator(mode="after")
    def check_manager_llm(self):
        """Validates that the language model is set when using hierarchical process."""
        if self.process == Process.hierarchical and not self.manager_llm:
            raise PydanticCustomError(
                "missing_manager_llm",
                "Attribute `manager_llm` is required when using hierarchical process.",
                {},
            )
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
                if self.max_rpm:
                    agent.set_rpm_controller(self._rpm_controller)
        return self

    def _setup_from_config(self):
        assert self.config is not None, "Config should not be None."

        """Initializes agents and tasks from the provided config."""
        if not self.config.get("agents") or not self.config.get("tasks"):
            raise PydanticCustomError(
                "missing_keys_in_config", "Config should have 'agents' and 'tasks'.", {}
            )

        self.process = self.config.get("process", self.process)
        self.agents = [Agent(**agent) for agent in self.config["agents"]]
        self.tasks = [self._create_task(task) for task in self.config["tasks"]]

    def _create_task(self, task_config: Dict[str, Any]) -> Task:
        """Creates a task instance from its configuration.

        Args:
            task_config: The configuration of the task.

        Returns:
            A task instance.
        """
        task_agent = next(
            agt for agt in self.agents if agt.role == task_config["agent"]
        )
        del task_config["agent"]
        return Task(**task_config, agent=task_agent)

    def kickoff(self) -> str:
        """Starts the crew to work on its assigned tasks."""
        self._execution_span = self._telemetry.crew_execution_span(self)

        for agent in self.agents:
            agent.i18n = I18N(language=self.language)

            if not agent.function_calling_llm:
                agent.function_calling_llm = self.function_calling_llm
                agent.create_agent_executor()
            if not agent.step_callback:
                agent.step_callback = self.step_callback
                agent.create_agent_executor()

        if self.process == Process.sequential:
            return self._run_sequential_process()
        if self.process == Process.hierarchical:
            return self._run_hierarchical_process()

        raise NotImplementedError(
            f"The process '{self.process}' is not implemented yet."
        )

    def _run_sequential_process(self) -> str:
        """Executes tasks sequentially and returns the final output."""
        task_output = ""
        for task in self.tasks:
            if task.agent is not None and task.agent.allow_delegation:
                agents_for_delegation = [
                    agent for agent in self.agents if agent != task.agent
                ]
                task.tools += AgentTools(agents=agents_for_delegation).tools()

            role = task.agent.role if task.agent is not None else "None"
            self._logger.log("debug", f"Working Agent: {role}")
            self._logger.log("info", f"Starting Task: {task.description}")

            output = task.execute(context=task_output)
            if not task.async_execution:
                task_output = output

            role = task.agent.role if task.agent is not None else "None"
            self._logger.log("debug", f"[{role}] Task output: {task_output}\n\n")

        self._finish_execution(task_output)
        return self._format_output(task_output)

    def _run_hierarchical_process(self) -> str:
        """Creates and assigns a manager agent to make sure the crew completes the tasks."""

        i18n = I18N(language=self.language)
        manager = Agent(
            role=i18n.retrieve("hierarchical_manager_agent", "role"),
            goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
            backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
            tools=AgentTools(agents=self.agents).tools(),
            llm=self.manager_llm,
            verbose=True,
        )

        task_output = ""
        for task in self.tasks:
            self._logger.log("debug", f"Working Agent: {manager.role}")
            self._logger.log("info", f"Starting Task: {task.description}")

            task_output = task.execute(
                agent=manager, context=task_output, tools=manager.tools
            )

            self._logger.log(
                "debug", f"[{manager.role}] Task output: {task_output}\n\n"
            )

        self._finish_execution(task_output)
        return self._format_output(task_output)

    def _format_output(self, output: str) -> str:
        """Formats the output of the crew execution."""
        if self.full_output:
            return {
                "final_output": output,
                "tasks_outputs": [task.output for task in self.tasks],
            }
        else:
            return output

    def _finish_execution(self, output) -> None:
        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()
        self._telemetry.end_crew(self, output)
