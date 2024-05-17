import copy
import json
from queue import Queue
import threading
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
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
from pydantic.json import pydantic_encoder

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.process import Process
from crewai.task import Task
from crewai.telemetry import Telemetry
from crewai.tools.agent_tools import AgentTools
from crewai.utilities import I18N, FileHandler, Logger, RPMController


class Crew(BaseModel):
    """
    Represents a group of agents, defining how they should collaborate and the tasks they should perform.

    Attributes:
        tasks: List of tasks assigned to the crew.
        agents: List of agents part of this crew.
        manager_llm: The language model that will run manager agent.
        manager_agent: Custom agent that will be used as manager.
        memory: Whether the crew should use memory to store memories of it's execution.
        manager_callbacks: The callback handlers to be executed by the manager agent when hierarchical process is used
        cache: Whether the crew should use a cache to store the results of the tools execution.
        function_calling_llm: The language model that will run the tool calling for all the agents.
        process: The process flow that the crew will follow (e.g., sequential, hierarchical).
        verbose: Indicates the verbosity level for logging during execution.
        config: Configuration settings for the crew.
        max_rpm: Maximum number of requests per minute for the crew execution to be respected.
        prompt_file: Path to the prompt json file to be used for the crew.
        id: A unique identifier for the crew instance.
        full_output: Whether the crew should return the full output with all tasks outputs or just the final output.
        task_callback: Callback to be executed after each task for every agents execution.
        step_callback: Callback to be executed after each step for every agents execution.
        share_crew: Whether you want to share the complete crew infromation and execution with crewAI to make the library better, and allow us to train models.
    """

    __hash__ = object.__hash__  # type: ignore
    _execution_span: Any = PrivateAttr()
    _rpm_controller = PrivateAttr(default=None)
    _logger = PrivateAttr(default=None)
    _file_handler = PrivateAttr(default=None)
    _cache_handler: InstanceOf[CacheHandler] = PrivateAttr(
        default=CacheHandler())
    _short_term_memory: Optional[InstanceOf[ShortTermMemory]] = PrivateAttr()
    _long_term_memory: Optional[InstanceOf[LongTermMemory]] = PrivateAttr()
    _entity_memory: Optional[InstanceOf[EntityMemory]] = PrivateAttr()
    _thread_local: threading.local = PrivateAttr(
        default_factory=threading.local)

    cache: bool = Field(default=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[Task] = Field(default_factory=list)
    agents: List[Agent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: Union[int, bool] = Field(default=0)
    memory: bool = Field(
        default=False,
        description="Whether the crew should use memory to store memories of it's execution",
    )
    embedder: Optional[dict] = Field(
        default={"provider": "openai"},
        description="Configuration for the embedder to be used for the crew.",
    )
    usage_metrics: Optional[dict] = Field(
        default=None,
        description="Metrics for the LLM usage during all tasks execution.",
    )
    full_output: Optional[bool] = Field(
        default=False,
        description="Whether the crew should return the full output with all tasks outputs or just the final output.",
    )
    manager_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    manager_agent: Optional[Any] = Field(
        description="Custom agent that will be used as manager.", default=None
    )
    manager_callbacks: Optional[List[InstanceOf[BaseCallbackHandler]]] = Field(
        default=None,
        description="A list of callback handlers to be executed by the manager agent when hierarchical process is used",
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
    task_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each task for all agents execution.",
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the crew execution to be respected.",
    )
    prompt_file: str = Field(
        default=None,
        description="Path to the prompt json file to be used for the crew.",
    )
    output_log_file: Optional[Union[bool, str]] = Field(
        default=False,
        description="output_log_file",
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
        self._file_handler = None
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        return self

    @model_validator(mode="after")
    def create_crew_memory(self) -> "Crew":
        """Set private attributes."""
        self._short_term_memory = None
        self._long_term_memory = None
        self._entity_memory = None
        if self.memory:
            self._long_term_memory = LongTermMemory()
            self._short_term_memory = ShortTermMemory(
                embedder_config=self.embedder)
            self._entity_memory = EntityMemory(embedder_config=self.embedder)
        return self

    @model_validator(mode="after")
    def check_manager_llm(self):
        """Validates that the language model is set when using hierarchical process."""
        if self.process == Process.hierarchical:
            if not self.manager_llm and not self.manager_agent:
                raise PydanticCustomError(
                    "missing_manager_llm_or_manager_agent",
                    "Attribute `manager_llm` or `manager_agent` is required when using hierarchical process.",
                    {},
                )

            if (self.manager_agent is not None) and (
                self.agents.count(self.manager_agent) > 0
            ):
                raise PydanticCustomError(
                    "manager_agent_in_agents",
                    "Manager agent should not be included in agents list.",
                    {},
                )

        return self

    @model_validator(mode="after")
    def check_config(self):
        """Validates that the crew is properly configured with agents and tasks."""
        # TODO: See if we can drop not self.tasks and not self.agents since moving to thread safe
        if not self.config and not self.tasks and not self.agents:
            raise PydanticCustomError(
                "missing_keys",
                "Either 'agents' and 'tasks' need to be set or 'config'.",
                {},
            )

        if self.config:
            self._setup_from_config()

        # TODO: See if we can drop cache check since moving to thread safe
        if self.agents:
            for agent in self.agents:
                if self.cache:
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
        self.tasks = [self._create_task(task)
                      for task in self.config["tasks"]]

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

    def kickoff(self, inputs: Optional[Dict[str, Any]] = {}) -> str:
        """Starts the crew to work on its assigned tasks."""
        if not hasattr(self._thread_local, 'initialized'):
            self._initialize_thread_specific_components()
            self._thread_local.initialized = True

         # Clone tasks for thread-local storage
        self._thread_local.tasks = [task.clone() for task in self.tasks]
        self._thread_local.agents = [agent.clone() for agent in self.agents]

        # type: ignore # Argument 1 to "_interpolate_inputs" of "Crew" has incompatible type "dict[str, Any] | None"; expected "dict[str, Any]"
        self._interpolate_inputs(inputs)
        self._set_tasks_callbacks

        self._thread_local.execution_span = self._telemetry.crew_execution_span(
            self)

        i18n = I18N(prompt_file=self.prompt_file)

        for agent in self._thread_local.agents:
            agent.i18n = i18n
            agent.crew = self

            if not agent.function_calling_llm:
                agent.function_calling_llm = self.function_calling_llm
            if not agent.step_callback:
                agent.step_callback = self.step_callback

            agent.create_agent_executor()

        metrics = []

        try:
            if self.process == Process.sequential:
                result = self._run_sequential_process(
                    thread_local=self._thread_local)
            elif self.process == Process.hierarchical:
                # type: ignore # Unpacking a string is disallowed
                result, manager_metrics = self._run_hierarchical_process(
                    thread_local=self._thread_local)
                # type: ignore # Cannot determine type of "manager_metrics"
                metrics.append(manager_metrics)
            else:
                raise NotImplementedError(
                    f"The process '{self.process}' is not implemented yet.")

            metrics = metrics + [agent._token_process.get_summary()
                                 for agent in self._thread_local.agents]
            self.usage_metrics = {
                key: sum([m[key] for m in metrics if m is not None]) for key in metrics[0]
            }
        finally:
            self._reset_thread_local()

        return result

    def _initialize_thread_specific_components(self):
        """Initialize thread-specific properties, agents, and tasks."""
        self._thread_local._telemetry = Telemetry()
        self._thread_local._telemetry.set_tracer()
        self._thread_local._telemetry.crew_creation(self)
        self._thread_local.logger = copy.deepcopy(
            self._logger)
        self._thread_local.rpm_controller = RPMController(
            max_rpm=self.max_rpm, logger=self._thread_local.logger)
        self._thread_local.file_handler = copy.deepcopy(
            self._file_handler)
        self._thread_local.cache_handler = copy.deepcopy(
            self._cache_handler)
        self._thread_local.short_term_memory = copy.deepcopy(
            self._short_term_memory)
        self._thread_local.long_term_memory = copy.deepcopy(
            self._long_term_memory)
        self._thread_local.entity_memory = copy.deepcopy(
            self._entity_memory)

    def kickoff_for_each(self, inputs_list: List[Dict[str, Any]], use_threading: bool = True, max_threads: int = 10) -> List[str]:
        """Start multiple instances of crew to work on its assigned tasks for each input in a new thread if threading is enabled."""
        if not use_threading:
            return [self.kickoff(inputs) for inputs in inputs_list]

        def worker():
            while True:
                inputs = q.get()
                if inputs is None:
                    break
                try:
                    result = self.kickoff(inputs)
                    results.append(result)
                except Exception as e:
                    self._logger.log('error', f"Error processing inputs {
                                     inputs}: {str(e)}")
                finally:
                    q.task_done()

        q = Queue()
        results = []
        threads = []

        for _ in range(min(max_threads, len(inputs_list))):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

        for inputs in inputs_list:
            q.put(inputs)

        q.join()

        for _ in range(len(threads)):
            q.put(None)
        for thread in threads:
            thread.join()

        return results

    def _run_sequential_process(self, thread_local=None) -> str:
        """Executes tasks sequentially and returns the final output."""
        task_output = ""
        for task in self._thread_local.tasks:
            if task.agent.allow_delegation:  # type: ignore #  Item "None" of "Agent | None" has no attribute "allow_delegation"
                agents_for_delegation = [
                    agent for agent in self._thread_local.agents if agent != task.agent]
                if len(self._thread_local.agents) > 1 and len(agents_for_delegation) > 0:
                    task.tools += AgentTools(agents=agents_for_delegation).tools()

            role = task.agent.role if task.agent is not None else "None"
            thread_local.logger.log("debug", f"== Working Agent: {
                                    role}", color="bold_purple")
            thread_local.logger.log("info", f"== Starting Task: {
                                    task.description}", color="bold_purple")

            if self.output_log_file:
                thread_local.file_handler.log(
                    agent=role, task=task.description, status="started")

            output = task.execute(context=task_output)
            if not task.async_execution:
                task_output = output

            role = task.agent.role if task.agent is not None else "None"
            thread_local.logger.log(
                "debug", f"== [{role}] Task output: {task_output}\n\n")

            if self.output_log_file:
                thread_local.file_handler.log(
                    agent=role, task=task_output, status="completed")

        self._finish_execution(task_output, thread_local=thread_local)
        return self._format_output(task_output)

    def _run_hierarchical_process(self, thread_local=None) -> str:
        """Creates and assigns a manager agent to make sure the crew completes the tasks."""
        i18n = I18N(prompt_file=self.prompt_file)
        if self.manager_agent is not None:
            self.manager_agent.allow_delegation = True
            manager = self.manager_agent
            if len(manager.tools) > 0:
                raise Exception("Manager agent should not have tools")
            manager.tools = AgentTools(
                agents=self._thread_local.agents).tools()
        else:
            manager = Agent(
                role=i18n.retrieve("hierarchical_manager_agent", "role"),
                goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
                backstory=i18n.retrieve(
                    "hierarchical_manager_agent", "backstory"),
                tools=AgentTools(agents=self._thread_local.agents).tools(),
                llm=self.manager_llm,
                verbose=True,
            )

        task_output = ""
        for task in self._thread_local.tasks:
            thread_local.logger.log("debug", f"Working Agent: {manager.role}")
            thread_local.logger.log(
                "info", f"Starting Task: {task.description}")

            if self.output_log_file:
                thread_local.file_handler.log(
                    agent=manager.role, task=task.description, status="started")

            task_output = task.execute(
                agent=manager, context=task_output, tools=manager.tools)

            thread_local.logger.log(
                "debug", f"[{manager.role}] Task output: {task_output}")

            if self.output_log_file:
                thread_local.file_handler.log(
                    agent=manager.role, task=task_output, status="completed")

        self._finish_execution(task_output, thread_local=thread_local)
        # type: ignore # Incompatible return value type (got "tuple[str, Any]", expected "str")
        return self._format_output(task_output), manager._token_process.get_summary()

    def _set_tasks_callbacks(self) -> None:
        """Sets callback for every task using task_callback."""
        for task in self._thread_local.tasks:
            if not task.callback:
                task.callback = self.task_callback

    def _interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolates the inputs in the tasks and agents."""
        print("Interpolating inputs")
        print(inputs)
        # type: ignore # "interpolate_inputs" of "Task" does not return a value (it only ever returns None)
        [task.interpolate_inputs(inputs) for task in self._thread_local.tasks]
        # type: ignore # "interpolate_inputs" of "Agent" does not return a value (it only ever returns None)
        [agent.interpolate_inputs(inputs)
         for agent in self._thread_local.agents]

    def _format_output(self, output: str) -> str:
        """Formats the output of the crew execution."""
        if self.full_output:
            return {  # type: ignore # Incompatible return value type (got "dict[str, Sequence[str | TaskOutput | None]]", expected "str")
                "final_output": output,
                "tasks_outputs": [task.output for task in self._thread_local.tasks if task],
            }
        else:
            return output

    def _finish_execution(self, output, thread_local=None) -> None:
        if self.max_rpm:
            if thread_local:
                thread_local.rpm_controller.stop_rpm_counter()
            else:
                self._rpm_controller.stop_rpm_counter()
        self._telemetry.end_crew(self, output)

    def _reset_thread_local(self):
        """Resets the thread-local storage."""
        self._thread_local.initialized = False

    def _deep_copy_without_threading(self, obj: Any) -> Any:
        if isinstance(obj, list):
            return [self._deep_copy_without_threading(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._deep_copy_without_threading(value) for key, value in obj.items()}
        elif isinstance(obj, BaseModel):
            new_obj = obj.__class__.__new__(obj.__class__)
            for key, value in obj.__dict__.items():
                if key not in ['_thread_local', '_lock', '_timer']:
                    setattr(new_obj, key, self._deep_copy_without_threading(value))
            # Copy internal Pydantic attributes
            if hasattr(obj, '__pydantic_validator__'):
                new_obj.__pydantic_validator__ = obj.__pydantic_validator__
            if hasattr(obj, '__pydantic_fields_set__'):
                new_obj.__pydantic_fields_set__ = obj.__pydantic_fields_set__
            return new_obj
        elif hasattr(obj, '__dict__'):
            new_obj = obj.__class__.__new__(obj.__class__)
            for key, value in obj.__dict__.items():
                if key not in ['_thread_local', '_lock', '_timer']:
                    setattr(new_obj, key, self._deep_copy_without_threading(value))
            return new_obj
        else:
            return copy.deepcopy(obj)

    def __repr__(self):
        return f"Crew(id={self.id}, process={self.process}, number_of_agents={len(self.agents)}, number_of_tasks={len(self.tasks)})"
