import asyncio
import json
import os
import uuid
from concurrent.futures import Future
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    InstanceOf,
    Json,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.cache import CacheHandler
from crewai.crews.crew_output import CrewOutput
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry import Telemetry
from crewai.tools.agent_tools import AgentTools
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities import I18N, FileHandler, Logger, RPMController
from crewai.utilities.constants import (
    TRAINING_DATA_FILE,
)
from crewai.utilities.evaluators.crew_evaluator_handler import CrewEvaluator
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from crewai.utilities.formatter import (
    aggregate_raw_outputs_from_task_outputs,
    aggregate_raw_outputs_from_tasks,
)
from crewai.utilities.planning_handler import CrewPlanner
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler
from crewai.utilities.training_handler import CrewTrainingHandler

agentops = None
if os.environ.get("AGENTOPS_API_KEY"):
    try:
        import agentops  # type: ignore
    except ImportError:
        pass

if TYPE_CHECKING:
    from crewai.pipeline.pipeline import Pipeline


class Crew(BaseModel):
    """
    Represents a group of agents, defining how they should collaborate and the tasks they should perform.

    Attributes:
        tasks: List of tasks assigned to the crew.
        agents: List of agents part of this crew.
        manager_llm: The language model that will run manager agent.
        manager_agent: Custom agent that will be used as manager.
        memory: Whether the crew should use memory to store memories of it's execution.
        cache: Whether the crew should use a cache to store the results of the tools execution.
        function_calling_llm: The language model that will run the tool calling for all the agents.
        process: The process flow that the crew will follow (e.g., sequential, hierarchical).
        verbose: Indicates the verbosity level for logging during execution.
        config: Configuration settings for the crew.
        max_rpm: Maximum number of requests per minute for the crew execution to be respected.
        prompt_file: Path to the prompt json file to be used for the crew.
        id: A unique identifier for the crew instance.
        task_callback: Callback to be executed after each task for every agents execution.
        step_callback: Callback to be executed after each step for every agents execution.
        share_crew: Whether you want to share the complete crew information and execution with crewAI to make the library better, and allow us to train models.
        planning: Plan the crew execution and add the plan to the crew.
    """

    __hash__ = object.__hash__  # type: ignore
    _execution_span: Any = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr()
    _logger: Logger = PrivateAttr()
    _file_handler: FileHandler = PrivateAttr()
    _cache_handler: InstanceOf[CacheHandler] = PrivateAttr(default=CacheHandler())
    _short_term_memory: Optional[InstanceOf[ShortTermMemory]] = PrivateAttr()
    _long_term_memory: Optional[InstanceOf[LongTermMemory]] = PrivateAttr()
    _entity_memory: Optional[InstanceOf[EntityMemory]] = PrivateAttr()
    _train: Optional[bool] = PrivateAttr(default=False)
    _train_iteration: Optional[int] = PrivateAttr()
    _inputs: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _logging_color: str = PrivateAttr(
        default="bold_purple",
    )
    _task_output_handler: TaskOutputStorageHandler = PrivateAttr(
        default_factory=TaskOutputStorageHandler
    )

    name: Optional[str] = Field(default=None)
    cache: bool = Field(default=True)
    tasks: List[Task] = Field(default_factory=list)
    agents: List[BaseAgent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: bool = Field(default=False)
    memory: bool = Field(
        default=False,
        description="Whether the crew should use memory to store memories of it's execution",
    )
    embedder: Optional[dict] = Field(
        default={"provider": "openai"},
        description="Configuration for the embedder to be used for the crew.",
    )
    usage_metrics: Optional[UsageMetrics] = Field(
        default=None,
        description="Metrics for the LLM usage during all tasks execution.",
    )
    manager_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    manager_agent: Optional[BaseAgent] = Field(
        description="Custom agent that will be used as manager.", default=None
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
    output_log_file: Optional[str] = Field(
        default=None,
        description="output_log_file",
    )
    planning: Optional[bool] = Field(
        default=False,
        description="Plan the crew execution and add the plan to the crew.",
    )
    planning_llm: Optional[Any] = Field(
        default=None,
        description="Language model that will run the AgentPlanner if planning is True.",
    )
    task_execution_output_json_files: Optional[List[str]] = Field(
        default=None,
        description="List of file paths for task execution JSON files.",
    )
    execution_logs: List[Dict[str, Any]] = Field(
        default=[],
        description="List of execution logs for tasks",
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
        self._logger = Logger(verbose=self.verbose)
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        self.function_calling_llm = (
            getattr(self.function_calling_llm, "model_name", None)
            or getattr(self.function_calling_llm, "deployment_name", None)
            or self.function_calling_llm
        )
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()
        return self

    @model_validator(mode="after")
    def create_crew_memory(self) -> "Crew":
        """Set private attributes."""
        if self.memory:
            self._long_term_memory = LongTermMemory()
            self._short_term_memory = ShortTermMemory(
                crew=self, embedder_config=self.embedder
            )
            self._entity_memory = EntityMemory(crew=self, embedder_config=self.embedder)
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
                if self.cache:
                    agent.set_cache_handler(self._cache_handler)
                if self.max_rpm:
                    agent.set_rpm_controller(self._rpm_controller)
        return self

    @model_validator(mode="after")
    def validate_tasks(self):
        if self.process == Process.sequential:
            for task in self.tasks:
                if task.agent is None:
                    raise PydanticCustomError(
                        "missing_agent_in_task",
                        f"Sequential process error: Agent is missing in the task with the following description: {task.description}",  # type: ignore # Argument of type "str" cannot be assigned to parameter "message_template" of type "LiteralString"
                        {},
                    )

        return self

    @model_validator(mode="after")
    def validate_end_with_at_most_one_async_task(self):
        """Validates that the crew ends with at most one asynchronous task."""
        final_async_task_count = 0

        # Traverse tasks backward
        for task in reversed(self.tasks):
            if task.async_execution:
                final_async_task_count += 1
            else:
                break  # Stop traversing as soon as a non-async task is encountered

        if final_async_task_count > 1:
            raise PydanticCustomError(
                "async_task_count",
                "The crew must end with at most one asynchronous task.",
                {},
            )

        return self

    @model_validator(mode="after")
    def validate_first_task(self) -> "Crew":
        """Ensure the first task is not a ConditionalTask."""
        if self.tasks and isinstance(self.tasks[0], ConditionalTask):
            raise PydanticCustomError(
                "invalid_first_task",
                "The first task cannot be a ConditionalTask.",
                {},
            )
        return self

    @model_validator(mode="after")
    def validate_async_tasks_not_async(self) -> "Crew":
        """Ensure that ConditionalTask is not async."""
        for task in self.tasks:
            if task.async_execution and isinstance(task, ConditionalTask):
                raise PydanticCustomError(
                    "invalid_async_conditional_task",
                    f"Conditional Task: {task.description} , cannot be executed asynchronously.",  # type: ignore # Argument of type "str" cannot be assigned to parameter "message_template" of type "LiteralString"
                    {},
                )
        return self

    @model_validator(mode="after")
    def validate_async_task_cannot_include_sequential_async_tasks_in_context(self):
        """
        Validates that if a task is set to be executed asynchronously,
        it cannot include other asynchronous tasks in its context unless
        separated by a synchronous task.
        """
        for i, task in enumerate(self.tasks):
            if task.async_execution and task.context:
                for context_task in task.context:
                    if context_task.async_execution:
                        for j in range(i - 1, -1, -1):
                            if self.tasks[j] == context_task:
                                raise ValueError(
                                    f"Task '{task.description}' is asynchronous and cannot include other sequential asynchronous tasks in its context."
                                )
                            if not self.tasks[j].async_execution:
                                break
        return self

    @model_validator(mode="after")
    def validate_context_no_future_tasks(self):
        """Validates that a task's context does not include future tasks."""
        task_indices = {id(task): i for i, task in enumerate(self.tasks)}

        for task in self.tasks:
            if task.context:
                for context_task in task.context:
                    if id(context_task) not in task_indices:
                        continue  # Skip context tasks not in the main tasks list
                    if task_indices[id(context_task)] > task_indices[id(task)]:
                        raise ValueError(
                            f"Task '{task.description}' has a context dependency on a future task '{context_task.description}', which is not allowed."
                        )
        return self

    @property
    def key(self) -> str:
        source = [agent.key for agent in self.agents] + [
            task.key for task in self.tasks
        ]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()

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

    def _setup_for_training(self, filename: str) -> None:
        """Sets up the crew for training."""
        self._train = True

        for task in self.tasks:
            task.human_input = True

        for agent in self.agents:
            agent.allow_delegation = False

        CrewTrainingHandler(TRAINING_DATA_FILE).initialize_file()
        CrewTrainingHandler(filename).initialize_file()

    def train(
        self, n_iterations: int, filename: str, inputs: Optional[Dict[str, Any]] = {}
    ) -> None:
        """Trains the crew for a given number of iterations."""
        self._setup_for_training(filename)

        for n_iteration in range(n_iterations):
            self._train_iteration = n_iteration
            self.kickoff(inputs=inputs)

        training_data = CrewTrainingHandler(TRAINING_DATA_FILE).load()

        for agent in self.agents:
            result = TaskEvaluator(agent).evaluate_training_data(
                training_data=training_data, agent_id=str(agent.id)
            )

            CrewTrainingHandler(filename).save_trained_data(
                agent_id=str(agent.role), trained_data=result.model_dump()
            )

    def kickoff(
        self,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> CrewOutput:
        """Starts the crew to work on its assigned tasks."""
        self._execution_span = self._telemetry.crew_execution_span(self, inputs)
        self._task_output_handler.reset()
        self._logging_color = "bold_purple"

        if inputs is not None:
            self._inputs = inputs
            self._interpolate_inputs(inputs)
        self._set_tasks_callbacks()

        i18n = I18N(prompt_file=self.prompt_file)

        for agent in self.agents:
            agent.i18n = i18n
            # type: ignore[attr-defined] # Argument 1 to "_interpolate_inputs" of "Crew" has incompatible type "dict[str, Any] | None"; expected "dict[str, Any]"
            agent.crew = self  # type: ignore[attr-defined]
            # TODO: Create an AgentFunctionCalling protocol for future refactoring
            if not agent.function_calling_llm:  # type: ignore # "BaseAgent" has no attribute "function_calling_llm"
                agent.function_calling_llm = self.function_calling_llm  # type: ignore # "BaseAgent" has no attribute "function_calling_llm"

            if agent.allow_code_execution:  # type: ignore # BaseAgent" has no attribute "allow_code_execution"
                agent.tools += agent.get_code_execution_tools()  # type: ignore # "BaseAgent" has no attribute "get_code_execution_tools"; maybe "get_delegation_tools"?

            if not agent.step_callback:  # type: ignore # "BaseAgent" has no attribute "step_callback"
                agent.step_callback = self.step_callback  # type: ignore # "BaseAgent" has no attribute "step_callback"

            agent.create_agent_executor()

        if self.planning:
            self._handle_crew_planning()

        metrics: List[UsageMetrics] = []

        if self.process == Process.sequential:
            result = self._run_sequential_process()
        elif self.process == Process.hierarchical:
            result = self._run_hierarchical_process()
        else:
            raise NotImplementedError(
                f"The process '{self.process}' is not implemented yet."
            )

        metrics += [agent._token_process.get_summary() for agent in self.agents]

        self.usage_metrics = UsageMetrics()
        for metric in metrics:
            self.usage_metrics.add_usage_metrics(metric)

        return result

    def kickoff_for_each(self, inputs: List[Dict[str, Any]]) -> List[CrewOutput]:
        """Executes the Crew's workflow for each input in the list and aggregates results."""
        results: List[CrewOutput] = []

        # Initialize the parent crew's usage metrics
        total_usage_metrics = UsageMetrics()

        for input_data in inputs:
            crew = self.copy()

            output = crew.kickoff(inputs=input_data)

            if crew.usage_metrics:
                total_usage_metrics.add_usage_metrics(crew.usage_metrics)

            results.append(output)

        self.usage_metrics = total_usage_metrics
        self._task_output_handler.reset()
        return results

    async def kickoff_async(self, inputs: Optional[Dict[str, Any]] = {}) -> CrewOutput:
        """Asynchronous kickoff method to start the crew execution."""
        return await asyncio.to_thread(self.kickoff, inputs)

    async def kickoff_for_each_async(self, inputs: List[Dict]) -> List[CrewOutput]:
        crew_copies = [self.copy() for _ in inputs]

        async def run_crew(crew, input_data):
            return await crew.kickoff_async(inputs=input_data)

        tasks = [
            asyncio.create_task(run_crew(crew_copies[i], inputs[i]))
            for i in range(len(inputs))
        ]
        tasks = [
            asyncio.create_task(run_crew(crew_copies[i], inputs[i]))
            for i in range(len(inputs))
        ]

        results = await asyncio.gather(*tasks)

        total_usage_metrics = UsageMetrics()
        for crew in crew_copies:
            if crew.usage_metrics:
                total_usage_metrics.add_usage_metrics(crew.usage_metrics)

        self.usage_metrics = total_usage_metrics
        self._task_output_handler.reset()
        return results

    def _handle_crew_planning(self):
        """Handles the Crew planning."""
        self._logger.log("info", "Planning the crew execution")
        result = CrewPlanner(
            tasks=self.tasks, planning_agent_llm=self.planning_llm
        )._handle_crew_planning()

        for task, step_plan in zip(self.tasks, result.list_of_plans_per_task):
            task.description += step_plan.plan

    def _store_execution_log(
        self,
        task: Task,
        output: TaskOutput,
        task_index: int,
        was_replayed: bool = False,
    ):
        if self._inputs:
            inputs = self._inputs
        else:
            inputs = {}

        log = {
            "task": task,
            "output": {
                "description": output.description,
                "summary": output.summary,
                "raw": output.raw,
                "pydantic": output.pydantic,
                "json_dict": output.json_dict,
                "output_format": output.output_format,
                "agent": output.agent,
            },
            "task_index": task_index,
            "inputs": inputs,
            "was_replayed": was_replayed,
        }
        self._task_output_handler.update(task_index, log)

    def _run_sequential_process(self) -> CrewOutput:
        """Executes tasks sequentially and returns the final output."""
        return self._execute_tasks(self.tasks)

    def _run_hierarchical_process(self) -> CrewOutput:
        """Creates and assigns a manager agent to make sure the crew completes the tasks."""
        self._create_manager_agent()
        return self._execute_tasks(self.tasks)

    def _create_manager_agent(self):
        i18n = I18N(prompt_file=self.prompt_file)
        if self.manager_agent is not None:
            self.manager_agent.allow_delegation = True
            manager = self.manager_agent
            if manager.tools is not None and len(manager.tools) > 0:
                self._logger.log(
                    "warning", "Manager agent should not have tools", color="orange"
                )
                manager.tools = []
                raise Exception("Manager agent should not have tools")
            manager.tools = self.manager_agent.get_delegation_tools(self.agents)
        else:
            self.manager_llm = (
                getattr(self.manager_llm, "model_name", None)
                or getattr(self.manager_llm, "deployment_name", None)
                or self.manager_llm
            )
            manager = Agent(
                role=i18n.retrieve("hierarchical_manager_agent", "role"),
                goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
                backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
                tools=AgentTools(agents=self.agents).tools(),
                llm=self.manager_llm,
                verbose=self.verbose,
            )
            self.manager_agent = manager
        manager.crew = self

    def _execute_tasks(
        self,
        tasks: List[Task],
        start_index: Optional[int] = 0,
        was_replayed: bool = False,
    ) -> CrewOutput:
        """Executes tasks sequentially and returns the final output.

        Args:
            tasks (List[Task]): List of tasks to execute
            manager (Optional[BaseAgent], optional): Manager agent to use for delegation. Defaults to None.

        Returns:
            CrewOutput: Final output of the crew
        """

        task_outputs: List[TaskOutput] = []
        futures: List[Tuple[Task, Future[TaskOutput], int]] = []
        last_sync_output: Optional[TaskOutput] = None

        for task_index, task in enumerate(tasks):
            if start_index is not None and task_index < start_index:
                if task.output:
                    if task.async_execution:
                        task_outputs.append(task.output)
                    else:
                        task_outputs = [task.output]
                        last_sync_output = task.output
                continue

            agent_to_use = self._get_agent_to_use(task)
            if agent_to_use is None:
                raise ValueError(
                    f"No agent available for task: {task.description}. Ensure that either the task has an assigned agent or a manager agent is provided."
                )

            self._prepare_agent_tools(task)
            self._log_task_start(task, agent_to_use.role)

            if isinstance(task, ConditionalTask):
                skipped_task_output = self._handle_conditional_task(
                    task, task_outputs, futures, task_index, was_replayed
                )
                if skipped_task_output:
                    continue

            if task.async_execution:
                context = self._get_context(
                    task, [last_sync_output] if last_sync_output else []
                )
                future = task.execute_async(
                    agent=agent_to_use,
                    context=context,
                    tools=agent_to_use.tools,
                )
                futures.append((task, future, task_index))
            else:
                if futures:
                    task_outputs = self._process_async_tasks(futures, was_replayed)
                    futures.clear()

                context = self._get_context(task, task_outputs)
                task_output = task.execute_sync(
                    agent=agent_to_use,
                    context=context,
                    tools=agent_to_use.tools,
                )
                task_outputs = [task_output]
                self._process_task_result(task, task_output)
                self._store_execution_log(task, task_output, task_index, was_replayed)

        if futures:
            task_outputs = self._process_async_tasks(futures, was_replayed)

        return self._create_crew_output(task_outputs)

    def _handle_conditional_task(
        self,
        task: ConditionalTask,
        task_outputs: List[TaskOutput],
        futures: List[Tuple[Task, Future[TaskOutput], int]],
        task_index: int,
        was_replayed: bool,
    ) -> Optional[TaskOutput]:
        if futures:
            task_outputs = self._process_async_tasks(futures, was_replayed)
            futures.clear()

        previous_output = task_outputs[task_index - 1] if task_outputs else None
        if previous_output is not None and not task.should_execute(previous_output):
            self._logger.log(
                "debug",
                f"Skipping conditional task: {task.description}",
                color="yellow",
            )
            skipped_task_output = task.get_skipped_task_output()

            if not was_replayed:
                self._store_execution_log(task, skipped_task_output, task_index)
            return skipped_task_output
        return None

    def _prepare_agent_tools(self, task: Task):
        if self.process == Process.hierarchical:
            if self.manager_agent:
                self._update_manager_tools(task)
            else:
                raise ValueError("Manager agent is required for hierarchical process.")
        elif task.agent and task.agent.allow_delegation:
            self._add_delegation_tools(task)

    def _get_agent_to_use(self, task: Task) -> Optional[BaseAgent]:
        if self.process == Process.hierarchical:
            return self.manager_agent
        return task.agent

    def _add_delegation_tools(self, task: Task):
        agents_for_delegation = [agent for agent in self.agents if agent != task.agent]
        if len(self.agents) > 1 and len(agents_for_delegation) > 0 and task.agent:
            delegation_tools = task.agent.get_delegation_tools(agents_for_delegation)

            # Add tools if they are not already in task.tools
            for new_tool in delegation_tools:
                # Find the index of the tool with the same name
                existing_tool_index = next(
                    (
                        index
                        for index, tool in enumerate(task.tools or [])
                        if tool.name == new_tool.name
                    ),
                    None,
                )
                if not task.tools:
                    task.tools = []

                if existing_tool_index is not None:
                    # Replace the existing tool
                    task.tools[existing_tool_index] = new_tool
                else:
                    # Add the new tool
                    task.tools.append(new_tool)

    def _log_task_start(self, task: Task, role: str = "None"):
        if self.output_log_file:
            self._file_handler.log(agent=role, task=task.description, status="started")

    def _update_manager_tools(self, task: Task):
        if self.manager_agent:
            if task.agent:
                self.manager_agent.tools = task.agent.get_delegation_tools([task.agent])
            else:
                self.manager_agent.tools = self.manager_agent.get_delegation_tools(
                    self.agents
                )

    def _get_context(self, task: Task, task_outputs: List[TaskOutput]):
        context = (
            aggregate_raw_outputs_from_tasks(task.context)
            if task.context
            else aggregate_raw_outputs_from_task_outputs(task_outputs)
        )
        return context

    def _process_task_result(self, task: Task, output: TaskOutput) -> None:
        role = task.agent.role if task.agent is not None else "None"
        if self.output_log_file:
            self._file_handler.log(agent=role, task=output, status="completed")

    def _create_crew_output(self, task_outputs: List[TaskOutput]) -> CrewOutput:
        if len(task_outputs) != 1:
            raise ValueError(
                "Something went wrong. Kickoff should return only one task output."
            )
        final_task_output = task_outputs[0]
        final_string_output = final_task_output.raw
        self._finish_execution(final_string_output)
        token_usage = self.calculate_usage_metrics()

        return CrewOutput(
            raw=final_task_output.raw,
            pydantic=final_task_output.pydantic,
            json_dict=final_task_output.json_dict,
            tasks_output=[task.output for task in self.tasks if task.output],
            token_usage=token_usage,
        )

    def _process_async_tasks(
        self,
        futures: List[Tuple[Task, Future[TaskOutput], int]],
        was_replayed: bool = False,
    ) -> List[TaskOutput]:
        task_outputs: List[TaskOutput] = []
        for future_task, future, task_index in futures:
            task_output = future.result()
            task_outputs.append(task_output)
            self._process_task_result(future_task, task_output)
            self._store_execution_log(
                future_task, task_output, task_index, was_replayed
            )
        return task_outputs

    def _find_task_index(
        self, task_id: str, stored_outputs: List[Any]
    ) -> Optional[int]:
        return next(
            (
                index
                for (index, d) in enumerate(stored_outputs)
                if d["task_id"] == str(task_id)
            ),
            None,
        )

    def replay(
        self, task_id: str, inputs: Optional[Dict[str, Any]] = None
    ) -> CrewOutput:
        stored_outputs = self._task_output_handler.load()
        if not stored_outputs:
            raise ValueError(f"Task with id {task_id} not found in the crew's tasks.")

        start_index = self._find_task_index(task_id, stored_outputs)

        if start_index is None:
            raise ValueError(f"Task with id {task_id} not found in the crew's tasks.")

        replay_inputs = (
            inputs if inputs is not None else stored_outputs[start_index]["inputs"]
        )
        self._inputs = replay_inputs

        if replay_inputs:
            self._interpolate_inputs(replay_inputs)

        if self.process == Process.hierarchical:
            self._create_manager_agent()

        for i in range(start_index):
            stored_output = stored_outputs[i][
                "output"
            ]  # for adding context to the task
            task_output = TaskOutput(
                description=stored_output["description"],
                agent=stored_output["agent"],
                raw=stored_output["raw"],
                pydantic=stored_output["pydantic"],
                json_dict=stored_output["json_dict"],
                output_format=stored_output["output_format"],
            )
            self.tasks[i].output = task_output

        self._logging_color = "bold_blue"
        result = self._execute_tasks(self.tasks, start_index, True)
        return result

    def copy(self):
        """Create a deep copy of the Crew."""

        exclude = {
            "id",
            "_rpm_controller",
            "_logger",
            "_execution_span",
            "_file_handler",
            "_cache_handler",
            "_short_term_memory",
            "_long_term_memory",
            "_entity_memory",
            "_telemetry",
            "agents",
            "tasks",
        }

        cloned_agents = [agent.copy() for agent in self.agents]
        cloned_tasks = [task.copy(cloned_agents) for task in self.tasks]

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}

        copied_data.pop("agents", None)
        copied_data.pop("tasks", None)

        copied_crew = Crew(**copied_data, agents=cloned_agents, tasks=cloned_tasks)

        return copied_crew

    def _set_tasks_callbacks(self) -> None:
        """Sets callback for every task suing task_callback"""
        for task in self.tasks:
            if not task.callback:
                task.callback = self.task_callback

    def _interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolates the inputs in the tasks and agents."""
        [
            task.interpolate_inputs(
                # type: ignore # "interpolate_inputs" of "Task" does not return a value (it only ever returns None)
                inputs
            )
            for task in self.tasks
        ]
        # type: ignore # "interpolate_inputs" of "Agent" does not return a value (it only ever returns None)
        for agent in self.agents:
            agent.interpolate_inputs(inputs)

    def _finish_execution(self, final_string_output: str) -> None:
        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()
        if agentops:
            agentops.end_session(
                end_state="Success",
                end_state_reason="Finished Execution",
            )
        self._telemetry.end_crew(self, final_string_output)

    def calculate_usage_metrics(self) -> UsageMetrics:
        """Calculates and returns the usage metrics."""
        total_usage_metrics = UsageMetrics()
        for agent in self.agents:
            if hasattr(agent, "_token_process"):
                token_sum = agent._token_process.get_summary()
                total_usage_metrics.add_usage_metrics(token_sum)
        if self.manager_agent and hasattr(self.manager_agent, "_token_process"):
            token_sum = self.manager_agent._token_process.get_summary()
            total_usage_metrics.add_usage_metrics(token_sum)
        self.usage_metrics = total_usage_metrics
        return total_usage_metrics

    def test(
        self,
        n_iterations: int,
        openai_model_name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Test and evaluate the Crew with the given inputs for n iterations concurrently using concurrent.futures."""
        self._test_execution_span = self._telemetry.test_execution_span(
            self, n_iterations, inputs, openai_model_name
        )
        evaluator = CrewEvaluator(self, openai_model_name)

        for i in range(1, n_iterations + 1):
            evaluator.set_iteration(i)
            self.kickoff(inputs=inputs)

        evaluator.print_crew_evaluation_result()

    def __rshift__(self, other: "Crew") -> "Pipeline":
        """
        Implements the >> operator to add another Crew to an existing Pipeline.
        """
        from crewai.pipeline.pipeline import Pipeline

        if not isinstance(other, Crew):
            raise TypeError(
                f"Unsupported operand type for >>: '{type(self).__name__}' and '{type(other).__name__}'"
            )
        return Pipeline(stages=[self, other])

    def __repr__(self):
        return f"Crew(id={self.id}, process={self.process}, number_of_agents={len(self.agents)}, number_of_tasks={len(self.tasks)})"
