from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from copy import copy as shallow_copy
from hashlib import md5
import json
import re
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import uuid
import warnings

from opentelemetry import baggage
from opentelemetry.context import attach, detach
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
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Self


if TYPE_CHECKING:
    from crewai_files import FileInput

try:
    from crewai_files import get_supported_content_types

    HAS_CREWAI_FILES = True
except ImportError:
    HAS_CREWAI_FILES = False

    def get_supported_content_types(provider: str, api: str | None = None) -> list[str]:
        return []


from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.crews.crew_output import CrewOutput
from crewai.crews.utils import (
    StreamingContext,
    check_conditional_skip,
    enable_agent_streaming,
    prepare_kickoff,
    prepare_task_execution,
    run_for_each_async,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_listener import EventListener
from crewai.events.listeners.tracing.trace_listener import (
    TraceCollectionListener,
)
from crewai.events.listeners.tracing.utils import (
    set_tracing_enabled,
    should_enable_tracing,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewTestCompletedEvent,
    CrewTestFailedEvent,
    CrewTestStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainFailedEvent,
    CrewTrainStartedEvent,
)
from crewai.flow.flow_trackable import FlowTrackable
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.external.external_memory import ExternalMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.process import Process
from crewai.rag.embeddings.types import EmbedderConfig
from crewai.rag.types import SearchResult
from crewai.security.fingerprint import Fingerprint
from crewai.security.security_config import SecurityConfig
from crewai.task import Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.tools.agent_tools.read_file_tool import ReadFileTool
from crewai.tools.base_tool import BaseTool
from crewai.types.streaming import CrewStreamingOutput
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities.constants import NOT_SPECIFIED, TRAINING_DATA_FILE
from crewai.utilities.crew.models import CrewContext
from crewai.utilities.evaluators.crew_evaluator_handler import CrewEvaluator
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from crewai.utilities.file_handler import FileHandler
from crewai.utilities.file_store import clear_files, get_all_files
from crewai.utilities.formatter import (
    aggregate_raw_outputs_from_task_outputs,
    aggregate_raw_outputs_from_tasks,
)
from crewai.utilities.i18n import get_i18n
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.logger import Logger
from crewai.utilities.planning_handler import CrewPlanner
from crewai.utilities.printer import PrinterColor
from crewai.utilities.rpm_controller import RPMController
from crewai.utilities.streaming import (
    create_async_chunk_generator,
    create_chunk_generator,
    signal_end,
    signal_error,
)
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler
from crewai.utilities.training_handler import CrewTrainingHandler


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


class Crew(FlowTrackable, BaseModel):
    """
    Represents a group of agents, defining how they should collaborate and the
    tasks they should perform.

    Attributes:
        tasks: list of tasks assigned to the crew.
        agents: list of agents part of this crew.
        manager_llm: The language model that will run manager agent.
        manager_agent: Custom agent that will be used as manager.
        memory: Whether the crew should use memory to store memories of it's
            execution.
        cache: Whether the crew should use a cache to store the results of the
            tools execution.
        function_calling_llm: The language model that will run the tool calling
            for all the agents.
        process: The process flow that the crew will follow (e.g., sequential,
            hierarchical).
        verbose: Indicates the verbosity level for logging during execution.
        config: Configuration settings for the crew.
        max_rpm: Maximum number of requests per minute for the crew execution to
            be respected.
        prompt_file: Path to the prompt json file to be used for the crew.
        id: A unique identifier for the crew instance.
        task_callback: Callback to be executed after each task for every agents
            execution.
        step_callback: Callback to be executed after each step for every agents
            execution.
        share_crew: Whether you want to share the complete crew information and
            execution with crewAI to make the library better, and allow us to
            train models.
        planning: Plan the crew execution and add the plan to the crew.
        chat_llm: The language model used for orchestrating chat interactions
            with the crew.
        security_config: Security configuration for the crew, including
            fingerprinting.
    """

    __hash__ = object.__hash__
    _execution_span: Any = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr()
    _logger: Logger = PrivateAttr()
    _file_handler: FileHandler = PrivateAttr()
    _cache_handler: InstanceOf[CacheHandler] = PrivateAttr(default_factory=CacheHandler)
    _short_term_memory: InstanceOf[ShortTermMemory] | None = PrivateAttr()
    _long_term_memory: InstanceOf[LongTermMemory] | None = PrivateAttr()
    _entity_memory: InstanceOf[EntityMemory] | None = PrivateAttr()
    _external_memory: InstanceOf[ExternalMemory] | None = PrivateAttr()
    _train: bool | None = PrivateAttr(default=False)
    _train_iteration: int | None = PrivateAttr()
    _inputs: dict[str, Any] | None = PrivateAttr(default=None)
    _logging_color: PrinterColor = PrivateAttr(
        default="bold_purple",
    )
    _task_output_handler: TaskOutputStorageHandler = PrivateAttr(
        default_factory=TaskOutputStorageHandler
    )

    name: str | None = Field(default="crew")
    cache: bool = Field(default=True)
    tasks: list[Task] = Field(default_factory=list)
    agents: list[BaseAgent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: bool = Field(default=False)
    memory: bool = Field(
        default=False,
        description="If crew should use memory to store memories of it's execution",
    )
    short_term_memory: InstanceOf[ShortTermMemory] | None = Field(
        default=None,
        description="An Instance of the ShortTermMemory to be used by the Crew",
    )
    long_term_memory: InstanceOf[LongTermMemory] | None = Field(
        default=None,
        description="An Instance of the LongTermMemory to be used by the Crew",
    )
    entity_memory: InstanceOf[EntityMemory] | None = Field(
        default=None,
        description="An Instance of the EntityMemory to be used by the Crew",
    )
    external_memory: InstanceOf[ExternalMemory] | None = Field(
        default=None,
        description="An Instance of the ExternalMemory to be used by the Crew",
    )
    embedder: EmbedderConfig | None = Field(
        default=None,
        description="Configuration for the embedder to be used for the crew.",
    )
    usage_metrics: UsageMetrics | None = Field(
        default=None,
        description="Metrics for the LLM usage during all tasks execution.",
    )
    manager_llm: str | InstanceOf[BaseLLM] | Any | None = Field(
        description="Language model that will run the agent.", default=None
    )
    manager_agent: BaseAgent | None = Field(
        description="Custom agent that will be used as manager.", default=None
    )
    function_calling_llm: str | InstanceOf[LLM] | Any | None = Field(
        description="Language model that will run the agent.", default=None
    )
    config: Json[dict[str, Any]] | dict[str, Any] | None = Field(default=None)
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    share_crew: bool | None = Field(default=False)
    step_callback: Any | None = Field(
        default=None,
        description="Callback to be executed after each step for all agents execution.",
    )
    task_callback: Any | None = Field(
        default=None,
        description="Callback to be executed after each task for all agents execution.",
    )
    before_kickoff_callbacks: list[
        Callable[[dict[str, Any] | None], dict[str, Any] | None]
    ] = Field(
        default_factory=list,
        description=(
            "List of callbacks to be executed before crew kickoff. "
            "It may be used to adjust inputs before the crew is executed."
        ),
    )
    after_kickoff_callbacks: list[Callable[[CrewOutput], CrewOutput]] = Field(
        default_factory=list,
        description=(
            "List of callbacks to be executed after crew kickoff. "
            "It may be used to adjust the output of the crew."
        ),
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream output from the crew execution.",
    )
    max_rpm: int | None = Field(
        default=None,
        description=(
            "Maximum number of requests per minute for the crew execution "
            "to be respected."
        ),
    )
    prompt_file: str | None = Field(
        default=None,
        description="Path to the prompt json file to be used for the crew.",
    )
    output_log_file: bool | str | None = Field(
        default=None,
        description="Path to the log file to be saved",
    )
    planning: bool | None = Field(
        default=False,
        description="Plan the crew execution and add the plan to the crew.",
    )
    planning_llm: str | InstanceOf[BaseLLM] | Any | None = Field(
        default=None,
        description=(
            "Language model that will run the AgentPlanner if planning is True."
        ),
    )
    task_execution_output_json_files: list[str] | None = Field(
        default=None,
        description="list of file paths for task execution JSON files.",
    )
    execution_logs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="list of execution logs for tasks",
    )
    knowledge_sources: list[BaseKnowledgeSource] | None = Field(
        default=None,
        description=(
            "Knowledge sources for the crew. Add knowledge sources to the "
            "knowledge object."
        ),
    )
    chat_llm: str | InstanceOf[BaseLLM] | Any | None = Field(
        default=None,
        description="LLM used to handle chatting with the crew.",
    )
    knowledge: Knowledge | None = Field(
        default=None,
        description="Knowledge for the crew.",
    )
    security_config: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration for the crew, including fingerprinting.",
    )
    token_usage: UsageMetrics | None = Field(
        default=None,
        description="Metrics for the LLM usage during all tasks execution.",
    )
    tracing: bool | None = Field(
        default=None,
        description="Whether to enable tracing for the crew. True=always enable, False=always disable, None=check environment/user settings.",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: UUID4 | None) -> None:
        """Prevent manual setting of the 'id' field by users."""
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "The 'id' field cannot be set by the user.", {}
            )

    @field_validator("config", mode="before")
    @classmethod
    def check_config_type(
        cls, v: Json[dict[str, Any]] | dict[str, Any]
    ) -> dict[str, Any]:
        """Validates that the config is a valid type.
        Args:
            v: The config to be validated.
        Returns:
            The config if it is valid.
        """

        # TODO: Improve typing
        return json.loads(v) if isinstance(v, Json) else v  # type: ignore

    @model_validator(mode="after")
    def set_private_attrs(self) -> Crew:
        """set private attributes."""
        self._cache_handler = CacheHandler()
        event_listener = EventListener()

        # Determine and set tracing state once for this execution
        tracing_enabled = should_enable_tracing(override=self.tracing)
        set_tracing_enabled(tracing_enabled)

        # Always setup trace listener - actual execution control is via contextvar
        trace_listener = TraceCollectionListener()
        trace_listener.setup_listeners(crewai_event_bus)
        event_listener.verbose = self.verbose
        event_listener.formatter.verbose = self.verbose
        self._logger = Logger(verbose=self.verbose)
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        if self.function_calling_llm and not isinstance(self.function_calling_llm, LLM):
            self.function_calling_llm = create_llm(self.function_calling_llm)

        return self

    def _initialize_default_memories(self) -> None:
        self._long_term_memory = self._long_term_memory or LongTermMemory()
        self._short_term_memory = self._short_term_memory or ShortTermMemory(
            crew=self,
            embedder_config=self.embedder,
        )
        self._entity_memory = self.entity_memory or EntityMemory(
            crew=self, embedder_config=self.embedder
        )

    @model_validator(mode="after")
    def create_crew_memory(self) -> Crew:
        """Initialize private memory attributes."""
        self._external_memory = (
            # External memory does not support a default value since it was
            # designed to be managed entirely externally
            self.external_memory.set_crew(self) if self.external_memory else None
        )

        self._long_term_memory = self.long_term_memory
        self._short_term_memory = self.short_term_memory
        self._entity_memory = self.entity_memory

        if self.memory:
            self._initialize_default_memories()

        return self

    @model_validator(mode="after")
    def create_crew_knowledge(self) -> Crew:
        """Create the knowledge for the crew."""
        if self.knowledge_sources:
            try:
                if isinstance(self.knowledge_sources, list) and all(
                    isinstance(k, BaseKnowledgeSource) for k in self.knowledge_sources
                ):
                    self.knowledge = Knowledge(
                        sources=self.knowledge_sources,
                        embedder=self.embedder,
                        collection_name="crew",
                    )
                    self.knowledge.add_sources()

            except Exception as e:
                self._logger.log(
                    "warning", f"Failed to init knowledge: {e}", color="yellow"
                )
        return self

    @model_validator(mode="after")
    def check_manager_llm(self) -> Self:
        """Validates that the language model is set when using hierarchical process."""
        if self.process == Process.hierarchical:
            if not self.manager_llm and not self.manager_agent:
                raise PydanticCustomError(
                    "missing_manager_llm_or_manager_agent",
                    (
                        "Attribute `manager_llm` or `manager_agent` is required when using hierarchical process."
                    ),
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
    def check_config(self) -> Self:
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
    def validate_tasks(self) -> Self:
        if self.process == Process.sequential:
            for task in self.tasks:
                if task.agent is None:
                    raise PydanticCustomError(
                        "missing_agent_in_task",
                        "Sequential process error: Agent is missing in the task with the following description: {description}",
                        {"description": task.description},
                    )

        return self

    @model_validator(mode="after")
    def validate_end_with_at_most_one_async_task(self) -> Self:
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
    def validate_must_have_non_conditional_task(self) -> Crew:
        """Ensure that a crew has at least one non-conditional task."""
        if not self.tasks:
            return self
        non_conditional_count = sum(
            1 for task in self.tasks if not isinstance(task, ConditionalTask)
        )
        if non_conditional_count == 0:
            raise PydanticCustomError(
                "only_conditional_tasks",
                "Crew must include at least one non-conditional task",
                {},
            )
        return self

    @model_validator(mode="after")
    def validate_first_task(self) -> Crew:
        """Ensure the first task is not a ConditionalTask."""
        if self.tasks and isinstance(self.tasks[0], ConditionalTask):
            raise PydanticCustomError(
                "invalid_first_task",
                "The first task cannot be a ConditionalTask.",
                {},
            )
        return self

    @model_validator(mode="after")
    def validate_async_tasks_not_async(self) -> Crew:
        """Ensure that ConditionalTask is not async."""
        for task in self.tasks:
            if task.async_execution and isinstance(task, ConditionalTask):
                raise PydanticCustomError(
                    "invalid_async_conditional_task",
                    (
                        "Conditional Task: {description}, cannot be executed asynchronously."
                    ),
                    {"description": task.description},
                )
        return self

    @model_validator(mode="after")
    def validate_async_task_cannot_include_sequential_async_tasks_in_context(
        self,
    ) -> Self:
        """
        Validates that if a task is set to be executed asynchronously,
        it cannot include other asynchronous tasks in its context unless
        separated by a synchronous task.
        """
        for i, task in enumerate(self.tasks):
            if task.async_execution and isinstance(task.context, list):
                for context_task in task.context:
                    if context_task.async_execution:
                        for j in range(i - 1, -1, -1):
                            if self.tasks[j] == context_task:
                                raise ValueError(
                                    f"Task '{task.description}' is asynchronous and "
                                    f"cannot include other sequential asynchronous "
                                    f"tasks in its context."
                                )
                            if not self.tasks[j].async_execution:
                                break
        return self

    @model_validator(mode="after")
    def validate_context_no_future_tasks(self) -> Self:
        """Validates that a task's context does not include future tasks."""
        task_indices = {id(task): i for i, task in enumerate(self.tasks)}

        for task in self.tasks:
            if isinstance(task.context, list):
                for context_task in task.context:
                    if id(context_task) not in task_indices:
                        continue  # Skip context tasks not in the main tasks list
                    if task_indices[id(context_task)] > task_indices[id(task)]:
                        raise ValueError(
                            f"Task '{task.description}' has a context dependency "
                            f"on a future task '{context_task.description}', "
                            f"which is not allowed."
                        )
        return self

    @property
    def key(self) -> str:
        source: list[str] = [agent.key for agent in self.agents] + [
            task.key for task in self.tasks
        ]
        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()

    @property
    def fingerprint(self) -> Fingerprint:
        """
        Get the crew's fingerprint.

        Returns:
            Fingerprint: The crew's fingerprint
        """
        return self.security_config.fingerprint

    def _setup_from_config(self) -> None:
        """Initializes agents and tasks from the provided config."""
        if self.config is None:
            raise ValueError("Config should not be None.")
        if not self.config.get("agents") or not self.config.get("tasks"):
            raise PydanticCustomError(
                "missing_keys_in_config", "Config should have 'agents' and 'tasks'.", {}
            )

        self.process = self.config.get("process", self.process)
        self.agents = [Agent(**agent) for agent in self.config["agents"]]
        self.tasks = [self._create_task(task) for task in self.config["tasks"]]

    def _create_task(self, task_config: dict[str, Any]) -> Task:
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
        self, n_iterations: int, filename: str, inputs: dict[str, Any] | None = None
    ) -> None:
        """Trains the crew for a given number of iterations."""
        inputs = inputs or {}
        try:
            crewai_event_bus.emit(
                self,
                CrewTrainStartedEvent(
                    crew_name=self.name,
                    n_iterations=n_iterations,
                    filename=filename,
                    inputs=inputs,
                ),
            )
            train_crew = self.copy()
            train_crew._setup_for_training(filename)

            for n_iteration in range(n_iterations):
                train_crew._train_iteration = n_iteration
                train_crew.kickoff(inputs=inputs)

            training_data = CrewTrainingHandler(TRAINING_DATA_FILE).load()

            for agent in train_crew.agents:
                if training_data.get(str(agent.id)):
                    result = TaskEvaluator(agent).evaluate_training_data(  # type: ignore[arg-type]
                        training_data=training_data, agent_id=str(agent.id)
                    )
                    CrewTrainingHandler(filename).save_trained_data(
                        agent_id=str(agent.role),
                        trained_data=result.model_dump(),
                    )

            crewai_event_bus.emit(
                self,
                CrewTrainCompletedEvent(
                    crew_name=self.name,
                    n_iterations=n_iterations,
                    filename=filename,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                CrewTrainFailedEvent(error=str(e), crew_name=self.name),
            )
            self._logger.log("error", f"Training failed: {e}", color="red")
            CrewTrainingHandler(TRAINING_DATA_FILE).clear()
            CrewTrainingHandler(filename).clear()
            raise

    def kickoff(
        self,
        inputs: dict[str, Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> CrewOutput | CrewStreamingOutput:
        """Execute the crew's workflow.

        Args:
            inputs: Optional input dictionary for task interpolation.
            input_files: Optional dict of named file inputs for the crew.

        Returns:
            CrewOutput or CrewStreamingOutput if streaming is enabled.
        """
        if self.stream:
            enable_agent_streaming(self.agents)
            ctx = StreamingContext()

            def run_crew() -> None:
                """Execute the crew and capture the result."""
                try:
                    self.stream = False
                    crew_result = self.kickoff(inputs=inputs, input_files=input_files)
                    if isinstance(crew_result, CrewOutput):
                        ctx.result_holder.append(crew_result)
                except Exception as exc:
                    signal_error(ctx.state, exc)
                finally:
                    self.stream = True
                    signal_end(ctx.state)

            streaming_output = CrewStreamingOutput(
                sync_iterator=create_chunk_generator(
                    ctx.state, run_crew, ctx.output_holder
                )
            )
            ctx.output_holder.append(streaming_output)
            return streaming_output

        baggage_ctx = baggage.set_baggage(
            "crew_context", CrewContext(id=str(self.id), key=self.key)
        )
        token = attach(baggage_ctx)

        try:
            inputs = prepare_kickoff(self, inputs, input_files)

            if self.process == Process.sequential:
                result = self._run_sequential_process()
            elif self.process == Process.hierarchical:
                result = self._run_hierarchical_process()
            else:
                raise NotImplementedError(
                    f"The process '{self.process}' is not implemented yet."
                )

            for after_callback in self.after_kickoff_callbacks:
                result = after_callback(result)

            self.usage_metrics = self.calculate_usage_metrics()

            return result
        except Exception as e:
            crewai_event_bus.emit(
                self,
                CrewKickoffFailedEvent(error=str(e), crew_name=self.name),
            )
            raise
        finally:
            clear_files(self.id)
            detach(token)

    def kickoff_for_each(
        self,
        inputs: list[dict[str, Any]],
        input_files: dict[str, FileInput] | None = None,
    ) -> list[CrewOutput | CrewStreamingOutput]:
        """Executes the Crew's workflow for each input and aggregates results.

        Args:
            inputs: List of input dictionaries, one per execution.
            input_files: Optional dict of named file inputs shared across all executions.

        Returns:
            List of CrewOutput or CrewStreamingOutput objects.

        If stream=True, returns a list of CrewStreamingOutput objects that must
        each be iterated to get stream chunks and access results.
        """
        results: list[CrewOutput | CrewStreamingOutput] = []

        total_usage_metrics = UsageMetrics()

        for input_data in inputs:
            crew = self.copy()

            output = crew.kickoff(inputs=input_data, input_files=input_files)

            if not self.stream and crew.usage_metrics:
                total_usage_metrics.add_usage_metrics(crew.usage_metrics)

            results.append(output)

        if not self.stream:
            self.usage_metrics = total_usage_metrics
        self._task_output_handler.reset()
        return results

    async def kickoff_async(
        self,
        inputs: dict[str, Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> CrewOutput | CrewStreamingOutput:
        """Asynchronous kickoff method to start the crew execution.

        Args:
            inputs: Optional input dictionary for task interpolation.
            input_files: Optional dict of named file inputs for the crew.

        Returns:
            CrewOutput or CrewStreamingOutput if streaming is enabled.

        If stream=True, returns a CrewStreamingOutput that can be async-iterated
        to get stream chunks. After iteration completes, access the final result
        via .result.
        """
        inputs = inputs or {}

        if self.stream:
            enable_agent_streaming(self.agents)
            ctx = StreamingContext(use_async=True)

            async def run_crew() -> None:
                try:
                    self.stream = False
                    result = await asyncio.to_thread(self.kickoff, inputs, input_files)
                    if isinstance(result, CrewOutput):
                        ctx.result_holder.append(result)
                except Exception as e:
                    signal_error(ctx.state, e, is_async=True)
                finally:
                    self.stream = True
                    signal_end(ctx.state, is_async=True)

            streaming_output = CrewStreamingOutput(
                async_iterator=create_async_chunk_generator(
                    ctx.state, run_crew, ctx.output_holder
                )
            )
            ctx.output_holder.append(streaming_output)

            return streaming_output

        return await asyncio.to_thread(self.kickoff, inputs, input_files)

    async def kickoff_for_each_async(
        self,
        inputs: list[dict[str, Any]],
        input_files: dict[str, FileInput] | None = None,
    ) -> list[CrewOutput | CrewStreamingOutput] | CrewStreamingOutput:
        """Executes the Crew's workflow for each input asynchronously.

        Args:
            inputs: List of input dictionaries, one per execution.
            input_files: Optional dict of named file inputs shared across all executions.

        Returns:
            List of CrewOutput or CrewStreamingOutput objects.

        If stream=True, returns a single CrewStreamingOutput that yields chunks
        from all crews as they arrive. After iteration, access results via .results
        (list of CrewOutput).
        """

        async def kickoff_fn(
            crew: Crew, input_data: dict[str, Any]
        ) -> CrewOutput | CrewStreamingOutput:
            return await crew.kickoff_async(inputs=input_data, input_files=input_files)

        return await run_for_each_async(self, inputs, kickoff_fn)

    async def akickoff(
        self,
        inputs: dict[str, Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> CrewOutput | CrewStreamingOutput:
        """Native async kickoff method using async task execution throughout.

        Unlike kickoff_async which wraps sync kickoff in a thread, this method
        uses native async/await for all operations including task execution,
        memory operations, and knowledge queries.

        Args:
            inputs: Optional input dictionary for task interpolation.
            input_files: Optional dict of named file inputs for the crew.

        Returns:
            CrewOutput or CrewStreamingOutput if streaming is enabled.
        """
        if self.stream:
            enable_agent_streaming(self.agents)
            ctx = StreamingContext(use_async=True)

            async def run_crew() -> None:
                try:
                    self.stream = False
                    inner_result = await self.akickoff(inputs, input_files)
                    if isinstance(inner_result, CrewOutput):
                        ctx.result_holder.append(inner_result)
                except Exception as exc:
                    signal_error(ctx.state, exc, is_async=True)
                finally:
                    self.stream = True
                    signal_end(ctx.state, is_async=True)

            streaming_output = CrewStreamingOutput(
                async_iterator=create_async_chunk_generator(
                    ctx.state, run_crew, ctx.output_holder
                )
            )
            ctx.output_holder.append(streaming_output)

            return streaming_output

        baggage_ctx = baggage.set_baggage(
            "crew_context", CrewContext(id=str(self.id), key=self.key)
        )
        token = attach(baggage_ctx)

        try:
            inputs = prepare_kickoff(self, inputs, input_files)

            if self.process == Process.sequential:
                result = await self._arun_sequential_process()
            elif self.process == Process.hierarchical:
                result = await self._arun_hierarchical_process()
            else:
                raise NotImplementedError(
                    f"The process '{self.process}' is not implemented yet."
                )

            for after_callback in self.after_kickoff_callbacks:
                result = after_callback(result)

            self.usage_metrics = self.calculate_usage_metrics()

            return result
        except Exception as e:
            crewai_event_bus.emit(
                self,
                CrewKickoffFailedEvent(error=str(e), crew_name=self.name),
            )
            raise
        finally:
            clear_files(self.id)
            detach(token)

    async def akickoff_for_each(
        self,
        inputs: list[dict[str, Any]],
        input_files: dict[str, FileInput] | None = None,
    ) -> list[CrewOutput | CrewStreamingOutput] | CrewStreamingOutput:
        """Native async execution of the Crew's workflow for each input.

        Uses native async throughout rather than thread-based async.

        Args:
            inputs: List of input dictionaries, one per execution.
            input_files: Optional dict of named file inputs shared across all executions.

        Returns:
            List of CrewOutput or CrewStreamingOutput objects.

        If stream=True, returns a single CrewStreamingOutput that yields chunks
        from all crews as they arrive.
        """

        async def kickoff_fn(
            crew: Crew, input_data: dict[str, Any]
        ) -> CrewOutput | CrewStreamingOutput:
            return await crew.akickoff(inputs=input_data, input_files=input_files)

        return await run_for_each_async(self, inputs, kickoff_fn)

    async def _arun_sequential_process(self) -> CrewOutput:
        """Executes tasks sequentially using native async and returns the final output."""
        return await self._aexecute_tasks(self.tasks)

    async def _arun_hierarchical_process(self) -> CrewOutput:
        """Creates and assigns a manager agent to complete the tasks using native async."""
        self._create_manager_agent()
        return await self._aexecute_tasks(self.tasks)

    async def _aexecute_tasks(
        self,
        tasks: list[Task],
        start_index: int | None = 0,
        was_replayed: bool = False,
    ) -> CrewOutput:
        """Executes tasks using native async and returns the final output.

        Args:
            tasks: List of tasks to execute
            start_index: Index to start execution from (for replay)
            was_replayed: Whether this is a replayed execution

        Returns:
            CrewOutput: Final output of the crew
        """
        task_outputs: list[TaskOutput] = []
        pending_tasks: list[tuple[Task, asyncio.Task[TaskOutput], int]] = []
        last_sync_output: TaskOutput | None = None

        for task_index, task in enumerate(tasks):
            exec_data, task_outputs, last_sync_output = prepare_task_execution(
                self, task, task_index, start_index, task_outputs, last_sync_output
            )
            if exec_data.should_skip:
                continue

            if isinstance(task, ConditionalTask):
                skipped_task_output = await self._ahandle_conditional_task(
                    task, task_outputs, pending_tasks, task_index, was_replayed
                )
                if skipped_task_output:
                    task_outputs.append(skipped_task_output)
                    continue

            if task.async_execution:
                context = self._get_context(
                    task, [last_sync_output] if last_sync_output else []
                )
                async_task = asyncio.create_task(
                    task.aexecute_sync(
                        agent=exec_data.agent,
                        context=context,
                        tools=exec_data.tools,
                    )
                )
                pending_tasks.append((task, async_task, task_index))
            else:
                if pending_tasks:
                    task_outputs = await self._aprocess_async_tasks(
                        pending_tasks, was_replayed
                    )
                    pending_tasks.clear()

                context = self._get_context(task, task_outputs)
                task_output = await task.aexecute_sync(
                    agent=exec_data.agent,
                    context=context,
                    tools=exec_data.tools,
                )
                task_outputs.append(task_output)
                self._process_task_result(task, task_output)
                self._store_execution_log(task, task_output, task_index, was_replayed)

        if pending_tasks:
            task_outputs = await self._aprocess_async_tasks(pending_tasks, was_replayed)

        return self._create_crew_output(task_outputs)

    async def _ahandle_conditional_task(
        self,
        task: ConditionalTask,
        task_outputs: list[TaskOutput],
        pending_tasks: list[tuple[Task, asyncio.Task[TaskOutput], int]],
        task_index: int,
        was_replayed: bool,
    ) -> TaskOutput | None:
        """Handle conditional task evaluation using native async."""
        if pending_tasks:
            task_outputs = await self._aprocess_async_tasks(pending_tasks, was_replayed)
            pending_tasks.clear()

        return check_conditional_skip(
            self, task, task_outputs, task_index, was_replayed
        )

    async def _aprocess_async_tasks(
        self,
        pending_tasks: list[tuple[Task, asyncio.Task[TaskOutput], int]],
        was_replayed: bool = False,
    ) -> list[TaskOutput]:
        """Process pending async tasks and return their outputs."""
        task_outputs: list[TaskOutput] = []
        for future_task, async_task, task_index in pending_tasks:
            task_output = await async_task
            task_outputs.append(task_output)
            self._process_task_result(future_task, task_output)
            self._store_execution_log(
                future_task, task_output, task_index, was_replayed
            )
        return task_outputs

    def _handle_crew_planning(self) -> None:
        """Handles the Crew planning."""
        self._logger.log("info", "Planning the crew execution")
        result = CrewPlanner(
            tasks=self.tasks, planning_agent_llm=self.planning_llm
        )._handle_crew_planning()

        plan_map: dict[int, str] = {}
        for step_plan in result.list_of_plans_per_task:
            if step_plan.task_number in plan_map:
                self._logger.log(
                    "warning",
                    f"Duplicate plan for Task Number {step_plan.task_number}, "
                    "using the first plan",
                )
            else:
                plan_map[step_plan.task_number] = step_plan.plan

        for idx, task in enumerate(self.tasks):
            task_number = idx + 1
            if task_number in plan_map:
                task.description += plan_map[task_number]
            else:
                self._logger.log(
                    "warning",
                    f"No plan found for Task Number {task_number}",
                )

    def _store_execution_log(
        self,
        task: Task,
        output: TaskOutput,
        task_index: int,
        was_replayed: bool = False,
    ) -> None:
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
                "messages": output.messages,
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
        """Creates and assigns a manager agent to complete the tasks."""
        self._create_manager_agent()
        return self._execute_tasks(self.tasks)

    def _create_manager_agent(self) -> None:
        if self.manager_agent is not None:
            self.manager_agent.allow_delegation = True
            manager = self.manager_agent
            if manager.tools is not None and len(manager.tools) > 0:
                self._logger.log(
                    "warning",
                    "Manager agent should not have tools",
                    color="bold_yellow",
                )
                manager.tools = []
                raise Exception("Manager agent should not have tools")
        else:
            self.manager_llm = create_llm(self.manager_llm)
            i18n = get_i18n(prompt_file=self.prompt_file)
            manager = Agent(
                role=i18n.retrieve("hierarchical_manager_agent", "role"),
                goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
                backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
                tools=AgentTools(agents=self.agents).tools(),
                allow_delegation=True,
                llm=self.manager_llm,
                verbose=self.verbose,
            )
            self.manager_agent = manager
        manager.crew = self

    def _execute_tasks(
        self,
        tasks: list[Task],
        start_index: int | None = 0,
        was_replayed: bool = False,
    ) -> CrewOutput:
        """Executes tasks sequentially and returns the final output.

        Args:
            tasks (List[Task]): List of tasks to execute
            manager (Optional[BaseAgent], optional): Manager agent to use for
                delegation. Defaults to None.

        Returns:
            CrewOutput: Final output of the crew
        """

        task_outputs: list[TaskOutput] = []
        futures: list[tuple[Task, Future[TaskOutput], int]] = []
        last_sync_output: TaskOutput | None = None

        for task_index, task in enumerate(tasks):
            exec_data, task_outputs, last_sync_output = prepare_task_execution(
                self, task, task_index, start_index, task_outputs, last_sync_output
            )
            if exec_data.should_skip:
                continue

            if isinstance(task, ConditionalTask):
                skipped_task_output = self._handle_conditional_task(
                    task, task_outputs, futures, task_index, was_replayed
                )
                if skipped_task_output:
                    task_outputs.append(skipped_task_output)
                    continue

            if task.async_execution:
                context = self._get_context(
                    task, [last_sync_output] if last_sync_output else []
                )
                future = task.execute_async(
                    agent=exec_data.agent,
                    context=context,
                    tools=exec_data.tools,
                )
                futures.append((task, future, task_index))
            else:
                if futures:
                    task_outputs = self._process_async_tasks(futures, was_replayed)
                    futures.clear()

                context = self._get_context(task, task_outputs)
                task_output = task.execute_sync(
                    agent=exec_data.agent,
                    context=context,
                    tools=exec_data.tools,
                )
                task_outputs.append(task_output)
                self._process_task_result(task, task_output)
                self._store_execution_log(task, task_output, task_index, was_replayed)

        if futures:
            task_outputs = self._process_async_tasks(futures, was_replayed)

        return self._create_crew_output(task_outputs)

    def _handle_conditional_task(
        self,
        task: ConditionalTask,
        task_outputs: list[TaskOutput],
        futures: list[tuple[Task, Future[TaskOutput], int]],
        task_index: int,
        was_replayed: bool,
    ) -> TaskOutput | None:
        if futures:
            task_outputs = self._process_async_tasks(futures, was_replayed)
            futures.clear()

        return check_conditional_skip(
            self, task, task_outputs, task_index, was_replayed
        )

    def _prepare_tools(
        self, agent: BaseAgent, task: Task, tools: list[BaseTool]
    ) -> list[BaseTool]:
        # Add delegation tools if agent allows delegation
        if hasattr(agent, "allow_delegation") and getattr(
            agent, "allow_delegation", False
        ):
            if self.process == Process.hierarchical:
                if self.manager_agent:
                    tools = self._update_manager_tools(task, tools)
                else:
                    raise ValueError(
                        "Manager agent is required for hierarchical process."
                    )

            elif agent:
                tools = self._add_delegation_tools(task, tools)

        # Add code execution tools if agent allows code execution
        if hasattr(agent, "allow_code_execution") and getattr(
            agent, "allow_code_execution", False
        ):
            tools = self._add_code_execution_tools(agent, tools)

        if (
            agent
            and hasattr(agent, "multimodal")
            and getattr(agent, "multimodal", False)
        ):
            if not (agent.llm and agent.llm.supports_multimodal()):
                tools = self._add_multimodal_tools(agent, tools)

        if agent and (hasattr(agent, "apps") and getattr(agent, "apps", None)):
            tools = self._add_platform_tools(task, tools)

        if agent and (hasattr(agent, "mcps") and getattr(agent, "mcps", None)):
            tools = self._add_mcp_tools(task, tools)

        files = get_all_files(self.id, task.id)
        if files:
            supported_types: list[str] = []
            if agent and agent.llm and agent.llm.supports_multimodal():
                provider = getattr(agent.llm, "provider", None) or getattr(
                    agent.llm, "model", "openai"
                )
                api = getattr(agent.llm, "api", None)
                supported_types = get_supported_content_types(provider, api)

            def is_auto_injected(content_type: str) -> bool:
                return any(content_type.startswith(t) for t in supported_types)

            # Only add read_file tool if there are files that need it
            files_needing_tool = {
                name: f
                for name, f in files.items()
                if not is_auto_injected(f.content_type)
            }
            if files_needing_tool:
                tools = self._add_file_tools(tools, files_needing_tool)

        return tools

    def _get_agent_to_use(self, task: Task) -> BaseAgent | None:
        if self.process == Process.hierarchical:
            return self.manager_agent
        return task.agent

    @staticmethod
    def _merge_tools(
        existing_tools: list[BaseTool],
        new_tools: list[BaseTool],
    ) -> list[BaseTool]:
        """Merge new tools into existing tools list, avoiding duplicates."""
        if not new_tools:
            return existing_tools

        # Create mapping of tool names to new tools
        new_tool_map = {sanitize_tool_name(tool.name): tool for tool in new_tools}

        # Remove any existing tools that will be replaced
        tools = [
            tool
            for tool in existing_tools
            if sanitize_tool_name(tool.name) not in new_tool_map
        ]

        # Add all new tools
        tools.extend(new_tools)

        return tools

    def _inject_delegation_tools(
        self,
        tools: list[BaseTool],
        task_agent: BaseAgent,
        agents: list[BaseAgent],
    ) -> list[BaseTool]:
        if hasattr(task_agent, "get_delegation_tools"):
            delegation_tools = task_agent.get_delegation_tools(agents)
            # Cast delegation_tools to the expected type for _merge_tools
            return self._merge_tools(tools, delegation_tools)
        return tools

    def _inject_platform_tools(
        self,
        tools: list[BaseTool],
        task_agent: BaseAgent,
    ) -> list[BaseTool]:
        apps = getattr(task_agent, "apps", None) or []

        if hasattr(task_agent, "get_platform_tools") and apps:
            platform_tools = task_agent.get_platform_tools(apps=apps)
            return self._merge_tools(tools, platform_tools)
        return tools

    def _inject_mcp_tools(
        self,
        tools: list[BaseTool],
        task_agent: BaseAgent,
    ) -> list[BaseTool]:
        mcps = getattr(task_agent, "mcps", None) or []
        if hasattr(task_agent, "get_mcp_tools") and mcps:
            mcp_tools = task_agent.get_mcp_tools(mcps=mcps)
            return self._merge_tools(tools, mcp_tools)
        return tools

    def _add_multimodal_tools(
        self, agent: BaseAgent, tools: list[BaseTool]
    ) -> list[BaseTool]:
        if hasattr(agent, "get_multimodal_tools"):
            multimodal_tools = agent.get_multimodal_tools()
            return self._merge_tools(tools, cast(list[BaseTool], multimodal_tools))
        return tools

    def _add_code_execution_tools(
        self, agent: BaseAgent, tools: list[BaseTool]
    ) -> list[BaseTool]:
        if hasattr(agent, "get_code_execution_tools"):
            code_tools = agent.get_code_execution_tools()
            # Cast code_tools to the expected type for _merge_tools
            return self._merge_tools(tools, cast(list[BaseTool], code_tools))
        return tools

    def _add_file_tools(
        self, tools: list[BaseTool], files: dict[str, Any]
    ) -> list[BaseTool]:
        """Add file reading tool when input files are available.

        Args:
            tools: Current list of tools.
            files: Dictionary of input files.

        Returns:
            Updated list with file tool added.
        """
        read_file_tool = ReadFileTool()
        read_file_tool.set_files(files)
        return self._merge_tools(tools, [read_file_tool])

    def _add_delegation_tools(
        self, task: Task, tools: list[BaseTool]
    ) -> list[BaseTool]:
        agents_for_delegation = [agent for agent in self.agents if agent != task.agent]
        if len(self.agents) > 1 and len(agents_for_delegation) > 0 and task.agent:
            if not tools:
                tools = []
            tools = self._inject_delegation_tools(
                tools, task.agent, agents_for_delegation
            )
        return tools

    def _add_platform_tools(self, task: Task, tools: list[BaseTool]) -> list[BaseTool]:
        if task.agent:
            tools = self._inject_platform_tools(tools, task.agent)

        return tools or []

    def _add_mcp_tools(self, task: Task, tools: list[BaseTool]) -> list[BaseTool]:
        if task.agent:
            tools = self._inject_mcp_tools(tools, task.agent)

        return tools or []

    def _log_task_start(self, task: Task, role: str = "None") -> None:
        if self.output_log_file:
            self._file_handler.log(
                task_name=task.name,  # type: ignore[arg-type]
                task=task.description,
                agent=role,
                status="started",
            )

    def _update_manager_tools(
        self, task: Task, tools: list[BaseTool]
    ) -> list[BaseTool]:
        if self.manager_agent:
            if task.agent:
                tools = self._inject_delegation_tools(tools, task.agent, [task.agent])
            else:
                tools = self._inject_delegation_tools(
                    tools, self.manager_agent, self.agents
                )
        return tools

    @staticmethod
    def _get_context(task: Task, task_outputs: list[TaskOutput]) -> str:
        if not task.context:
            return ""

        return (
            aggregate_raw_outputs_from_task_outputs(task_outputs)
            if task.context is NOT_SPECIFIED
            else aggregate_raw_outputs_from_tasks(task.context)
        )

    def _process_task_result(self, task: Task, output: TaskOutput) -> None:
        role = task.agent.role if task.agent is not None else "None"
        if self.output_log_file:
            self._file_handler.log(
                task_name=task.name,  # type: ignore[arg-type]
                task=task.description,
                agent=role,
                status="completed",
                output=output.raw,
            )

    def _create_crew_output(self, task_outputs: list[TaskOutput]) -> CrewOutput:
        if not task_outputs:
            raise ValueError("No task outputs available to create crew output.")

        # Filter out empty outputs and get the last valid one as the main output
        valid_outputs = [t for t in task_outputs if t.raw]
        if not valid_outputs:
            raise ValueError("No valid task outputs available to create crew output.")
        final_task_output = valid_outputs[-1]

        final_string_output = final_task_output.raw
        self._finish_execution(final_string_output)
        self.token_usage = self.calculate_usage_metrics()
        crewai_event_bus.emit(
            self,
            CrewKickoffCompletedEvent(
                crew_name=self.name,
                output=final_task_output,
                total_tokens=self.token_usage.total_tokens,
            ),
        )

        # Finalization is handled by trace listener (always initialized)
        # The batch manager checks contextvar to determine if tracing is enabled

        return CrewOutput(
            raw=final_task_output.raw,
            pydantic=final_task_output.pydantic,
            json_dict=final_task_output.json_dict,
            tasks_output=task_outputs,
            token_usage=self.token_usage,
        )

    def _process_async_tasks(
        self,
        futures: list[tuple[Task, Future[TaskOutput], int]],
        was_replayed: bool = False,
    ) -> list[TaskOutput]:
        task_outputs: list[TaskOutput] = []
        for future_task, future, task_index in futures:
            task_output = future.result()
            task_outputs.append(task_output)
            self._process_task_result(future_task, task_output)
            self._store_execution_log(
                future_task, task_output, task_index, was_replayed
            )
        return task_outputs

    @staticmethod
    def _find_task_index(task_id: str, stored_outputs: list[Any]) -> int | None:
        return next(
            (
                index
                for (index, d) in enumerate(stored_outputs)
                if d["task_id"] == str(task_id)
            ),
            None,
        )

    def replay(self, task_id: str, inputs: dict[str, Any] | None = None) -> CrewOutput:
        """Replay the crew execution from a specific task."""
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
                messages=stored_output.get("messages", []),
            )
            self.tasks[i].output = task_output

        self._logging_color = "bold_blue"
        return self._execute_tasks(self.tasks, start_index, True)

    def query_knowledge(
        self, query: list[str], results_limit: int = 3, score_threshold: float = 0.35
    ) -> list[SearchResult] | None:
        """Query the crew's knowledge base for relevant information."""
        if self.knowledge:
            return self.knowledge.query(
                query, results_limit=results_limit, score_threshold=score_threshold
            )
        return None

    async def aquery_knowledge(
        self, query: list[str], results_limit: int = 3, score_threshold: float = 0.35
    ) -> list[SearchResult] | None:
        """Query the crew's knowledge base for relevant information asynchronously."""
        if self.knowledge:
            return await self.knowledge.aquery(
                query, results_limit=results_limit, score_threshold=score_threshold
            )
        return None

    def fetch_inputs(self) -> set[str]:
        """
        Gathers placeholders (e.g., {something}) referenced in tasks or agents.
        Scans each task's 'description' + 'expected_output', and each agent's
        'role', 'goal', and 'backstory'.

        Returns a set of all discovered placeholder names.
        """
        placeholder_pattern = re.compile(r"\{(.+?)}")
        required_inputs: set[str] = set()

        # Scan tasks for inputs
        for task in self.tasks:
            # description and expected_output might contain e.g. {topic}, {user_name}
            text = f"{task.description or ''} {task.expected_output or ''}"
            required_inputs.update(placeholder_pattern.findall(text))

        # Scan agents for inputs
        for agent in self.agents:
            # role, goal, backstory might have placeholders like {role_detail}, etc.
            text = f"{agent.role or ''} {agent.goal or ''} {agent.backstory or ''}"
            required_inputs.update(placeholder_pattern.findall(text))

        return required_inputs

    def copy(self) -> Crew:  # type: ignore[override]
        """
        Creates a deep copy of the Crew instance.

        Returns:
            Crew: A new instance with copied components
        """

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
            "_external_memory",
            "agents",
            "tasks",
            "knowledge_sources",
            "knowledge",
            "manager_agent",
            "manager_llm",
        }

        cloned_agents = [agent.copy() for agent in self.agents]
        manager_agent = self.manager_agent.copy() if self.manager_agent else None
        manager_llm = shallow_copy(self.manager_llm) if self.manager_llm else None

        task_mapping: dict[str, Any] = {}

        cloned_tasks = []
        existing_knowledge_sources = shallow_copy(self.knowledge_sources)
        existing_knowledge = shallow_copy(self.knowledge)

        for task in self.tasks:
            cloned_task = task.copy(cloned_agents, task_mapping)
            cloned_tasks.append(cloned_task)
            task_mapping[task.key] = cloned_task

        for cloned_task, original_task in zip(cloned_tasks, self.tasks, strict=False):
            if isinstance(original_task.context, list):
                cloned_context = [
                    task_mapping[context_task.key]
                    for context_task in original_task.context
                ]
                cloned_task.context = cloned_context

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}
        if self.short_term_memory:
            copied_data["short_term_memory"] = self.short_term_memory.model_copy(
                deep=True
            )
        if self.long_term_memory:
            copied_data["long_term_memory"] = self.long_term_memory.model_copy(
                deep=True
            )
        if self.entity_memory:
            copied_data["entity_memory"] = self.entity_memory.model_copy(deep=True)
        if self.external_memory:
            copied_data["external_memory"] = self.external_memory.model_copy(deep=True)

        copied_data.pop("agents", None)
        copied_data.pop("tasks", None)

        return Crew(
            **copied_data,
            agents=cloned_agents,
            tasks=cloned_tasks,
            knowledge_sources=existing_knowledge_sources,
            knowledge=existing_knowledge,
            manager_agent=manager_agent,
            manager_llm=manager_llm,
        )

    def _set_tasks_callbacks(self) -> None:
        """Sets callback for every task suing task_callback"""
        for task in self.tasks:
            if not task.callback:
                task.callback = self.task_callback

    def _interpolate_inputs(self, inputs: dict[str, Any]) -> None:
        """Interpolates the inputs in the tasks and agents."""
        [
            task.interpolate_inputs_and_add_conversation_history(
                # type: ignore # "interpolate_inputs" of "Task" does not return a value (it only ever returns None)
                inputs
            )
            for task in self.tasks
        ]
        for agent in self.agents:
            agent.interpolate_inputs(inputs)

    def _finish_execution(self, final_string_output: str) -> None:
        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

    def calculate_usage_metrics(self) -> UsageMetrics:
        """Calculates and returns the usage metrics."""
        total_usage_metrics = UsageMetrics()

        for agent in self.agents:
            if isinstance(agent.llm, BaseLLM):
                llm_usage = agent.llm.get_token_usage_summary()

                total_usage_metrics.add_usage_metrics(llm_usage)
            else:
                # fallback litellm
                if hasattr(agent, "_token_process"):
                    token_sum = agent._token_process.get_summary()
                    total_usage_metrics.add_usage_metrics(token_sum)

        if self.manager_agent and hasattr(self.manager_agent, "_token_process"):
            token_sum = self.manager_agent._token_process.get_summary()
            total_usage_metrics.add_usage_metrics(token_sum)

        if (
            self.manager_agent
            and hasattr(self.manager_agent, "llm")
            and hasattr(self.manager_agent.llm, "get_token_usage_summary")
        ):
            if isinstance(self.manager_agent.llm, BaseLLM):
                llm_usage = self.manager_agent.llm.get_token_usage_summary()
            else:
                llm_usage = self.manager_agent.llm._token_process.get_summary()

            total_usage_metrics.add_usage_metrics(llm_usage)

        self.usage_metrics = total_usage_metrics
        return total_usage_metrics

    def test(
        self,
        n_iterations: int,
        eval_llm: str | InstanceOf[BaseLLM],
        inputs: dict[str, Any] | None = None,
    ) -> None:
        """Test and evaluate the Crew with the given inputs for n iterations.

        Uses concurrent.futures for concurrent execution.
        """
        try:
            # Create LLM instance and ensure it's of type LLM for CrewEvaluator
            llm_instance = create_llm(eval_llm)
            if not llm_instance:
                raise ValueError("Failed to create LLM instance.")

            crewai_event_bus.emit(
                self,
                CrewTestStartedEvent(
                    crew_name=self.name,
                    n_iterations=n_iterations,
                    eval_llm=llm_instance,
                    inputs=inputs,
                ),
            )
            test_crew = self.copy()

            evaluator = CrewEvaluator(test_crew, llm_instance)

            for i in range(1, n_iterations + 1):
                evaluator.set_iteration(i)
                test_crew.kickoff(inputs=inputs)

            evaluator.print_crew_evaluation_result()

            crewai_event_bus.emit(
                self,
                CrewTestCompletedEvent(
                    crew_name=self.name,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                CrewTestFailedEvent(error=str(e), crew_name=self.name),
            )
            raise

    def __repr__(self) -> str:
        return (
            f"Crew(id={self.id}, process={self.process}, "
            f"number_of_agents={len(self.agents)}, "
            f"number_of_tasks={len(self.tasks)})"
        )

    def reset_memories(self, command_type: str) -> None:
        """Reset specific or all memories for the crew.

        Args:
            command_type: Type of memory to reset.
                Valid options: 'long', 'short', 'entity', 'knowledge', 'agent_knowledge'
                'kickoff_outputs', or 'all'

        Raises:
            ValueError: If an invalid command type is provided.
            RuntimeError: If memory reset operation fails.
        """
        valid_types = frozenset(
            [
                "long",
                "short",
                "entity",
                "knowledge",
                "agent_knowledge",
                "kickoff_outputs",
                "all",
                "external",
            ]
        )

        if command_type not in valid_types:
            raise ValueError(
                f"Invalid command type. Must be one of: "
                f"{', '.join(sorted(valid_types))}"
            )

        try:
            if command_type == "all":
                self._reset_all_memories()
            else:
                self._reset_specific_memory(command_type)

        except Exception as e:
            error_msg = f"Failed to reset {command_type} memory: {e!s}"
            self._logger.log("error", error_msg)
            raise RuntimeError(error_msg) from e

    def _reset_memory_system(
        self, system: Any, name: str, reset_fn: Callable[[Any], Any]
    ) -> None:
        """Reset a single memory system.

        Args:
            system: The memory system instance to reset.
            name: Display name of the memory system for logging.
            reset_fn: Function to call to reset the system.

        Raises:
            RuntimeError: If the reset operation fails.
        """
        try:
            reset_fn(system)
            self._logger.log(
                "info",
                f"[Crew ({self.name if self.name else self.id})] "
                f"{name} memory has been reset",
            )
        except Exception as e:
            raise RuntimeError(
                f"[Crew ({self.name if self.name else self.id})] "
                f"Failed to reset {name} memory: {e!s}"
            ) from e

    def _reset_all_memories(self) -> None:
        """Reset all available memory systems."""
        memory_systems = self._get_memory_systems()

        for config in memory_systems.values():
            if (system := config.get("system")) is not None:
                name = config.get("name")
                reset_fn: Callable[[Any], Any] = cast(
                    Callable[[Any], Any], config.get("reset")
                )
                self._reset_memory_system(system, name, reset_fn)

    def _reset_specific_memory(self, memory_type: str) -> None:
        """Reset a specific memory system.

        Args:
            memory_type: Type of memory to reset

        Raises:
            RuntimeError: If the specified memory system fails to reset
        """
        memory_systems = self._get_memory_systems()
        config = memory_systems[memory_type]
        system = config.get("system")
        name = config.get("name")

        if system is None:
            raise RuntimeError(f"{name} memory system is not initialized")

        reset_fn: Callable[[Any], Any] = cast(Callable[[Any], Any], config.get("reset"))
        self._reset_memory_system(system, name, reset_fn)

    def _get_memory_systems(self) -> dict[str, Any]:
        """Get all available memory systems with their configuration.

        Returns:
            Dict containing all memory systems with their reset functions and
            display names.
        """

        def default_reset(memory: Any) -> Any:
            return memory.reset()

        def knowledge_reset(memory: Any) -> Any:
            return self.reset_knowledge(memory)

        # Get knowledge for agents
        agent_knowledges = [
            getattr(agent, "knowledge", None)
            for agent in self.agents
            if getattr(agent, "knowledge", None) is not None
        ]
        # Get knowledge for crew and agents
        crew_knowledge = getattr(self, "knowledge", None)
        crew_and_agent_knowledges = (
            [crew_knowledge] if crew_knowledge is not None else []
        ) + agent_knowledges

        return {
            "short": {
                "system": getattr(self, "_short_term_memory", None),
                "reset": default_reset,
                "name": "Short Term",
            },
            "entity": {
                "system": getattr(self, "_entity_memory", None),
                "reset": default_reset,
                "name": "Entity",
            },
            "external": {
                "system": getattr(self, "_external_memory", None),
                "reset": default_reset,
                "name": "External",
            },
            "long": {
                "system": getattr(self, "_long_term_memory", None),
                "reset": default_reset,
                "name": "Long Term",
            },
            "kickoff_outputs": {
                "system": getattr(self, "_task_output_handler", None),
                "reset": default_reset,
                "name": "Task Output",
            },
            "knowledge": {
                "system": crew_and_agent_knowledges
                if crew_and_agent_knowledges
                else None,
                "reset": knowledge_reset,
                "name": "Crew Knowledge and Agent Knowledge",
            },
            "agent_knowledge": {
                "system": agent_knowledges if agent_knowledges else None,
                "reset": knowledge_reset,
                "name": "Agent Knowledge",
            },
        }

    def reset_knowledge(self, knowledges: list[Knowledge]) -> None:
        """Reset crew and agent knowledge storage."""
        for ks in knowledges:
            ks.reset()

    def _set_allow_crewai_trigger_context_for_first_task(self) -> None:
        crewai_trigger_payload = self._inputs and self._inputs.get(
            "crewai_trigger_payload"
        )
        able_to_inject = (
            self.tasks and self.tasks[0].allow_crewai_trigger_context is None
        )

        if (
            self.process == Process.sequential
            and crewai_trigger_payload
            and able_to_inject
        ):
            self.tasks[0].allow_crewai_trigger_context = True

    @staticmethod
    def _show_tracing_disabled_message() -> None:
        """Show a message when tracing is disabled."""
        from crewai.events.listeners.tracing.utils import has_user_declined_tracing

        console = Console()

        if has_user_declined_tracing():
            message = """Info: Tracing is disabled.

To enable tracing, do any one of these:
• Set tracing=True in your Crew code
• Set CREWAI_TRACING_ENABLED=true in your project's .env file
• Run: crewai traces enable"""
        else:
            message = """Info: Tracing is disabled.

To enable tracing, do any one of these:
• Set tracing=True in your Crew code
• Set CREWAI_TRACING_ENABLED=true in your project's .env file
• Run: crewai traces enable"""

        panel = Panel(
            message,
            title="Tracing Status",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)
