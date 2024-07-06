import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

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

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.cache import CacheHandler
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.process import Process
from crewai.task import Task
from crewai.telemetry import Telemetry
from crewai.tools.agent_tools import AgentTools
from crewai.utilities import I18N, FileHandler, Logger, RPMController
from crewai.utilities.constants import TRAINED_AGENTS_DATA_FILE, TRAINING_DATA_FILE
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from crewai.utilities.training_handler import CrewTrainingHandler

try:
    import agentops
except ImportError:
    agentops = None


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
        full_output: Whether the crew should return the full output with all tasks outputs and token usage metrics or just the final output.
        task_callback: Callback to be executed after each task for every agents execution.
        step_callback: Callback to be executed after each step for every agents execution.
        share_crew: Whether you want to share the complete crew information and execution with crewAI to make the library better, and allow us to train models.
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

    cache: bool = Field(default=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[Task] = Field(default_factory=list)
    agents: List[BaseAgent] = Field(default_factory=list)
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
        description="Whether the crew should return the full output with all tasks outputs and token usage metrics or just the final output.",
    )
    manager_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    manager_agent: Optional[BaseAgent] = Field(
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
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()
        self._telemetry.crew_creation(self)
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
    def check_tasks_in_hierarchical_process_not_async(self):
        """Validates that the tasks in hierarchical process are not flagged with async_execution."""
        if self.process == Process.hierarchical:
            for task in self.tasks:
                if task.async_execution:
                    raise PydanticCustomError(
                        "async_execution_in_hierarchical_process",
                        "Hierarchical process error: Tasks cannot be flagged with async_execution.",
                        {},
                    )

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

    def _setup_for_training(self) -> None:
        """Sets up the crew for training."""
        self._train = True

        for task in self.tasks:
            task.human_input = True

        for agent in self.agents:
            agent.allow_delegation = False

        CrewTrainingHandler(TRAINING_DATA_FILE).initialize_file()
        CrewTrainingHandler(TRAINED_AGENTS_DATA_FILE).initialize_file()

    def train(self, n_iterations: int, inputs: Optional[Dict[str, Any]] = {}) -> None:
        """Trains the crew for a given number of iterations."""
        self._setup_for_training()

        for n_iteration in range(n_iterations):
            self._train_iteration = n_iteration
            self.kickoff(inputs=inputs)

        training_data = CrewTrainingHandler(TRAINING_DATA_FILE).load()

        for agent in self.agents:
            result = TaskEvaluator(agent).evaluate_training_data(
                training_data=training_data, agent_id=str(agent.id)
            )

            CrewTrainingHandler(TRAINED_AGENTS_DATA_FILE).save_trained_data(
                agent_id=str(agent.role), trained_data=result.model_dump()
            )

    def kickoff(
        self,
        inputs: Optional[Dict[str, Any]] = {},
    ) -> Union[str, Dict[str, Any]]:
        """Starts the crew to work on its assigned tasks."""
        self._execution_span = self._telemetry.crew_execution_span(self, inputs)

        self._interpolate_inputs(inputs)  # type: ignore # Argument 1 to "_interpolate_inputs" of "Crew" has incompatible type "dict[str, Any] | None"; expected "dict[str, Any]"
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

        metrics = []

        if self.process == Process.sequential:
            result = self._run_sequential_process()
        elif self.process == Process.hierarchical:
            result, manager_metrics = self._run_hierarchical_process()  # type: ignore # Incompatible types in assignment (expression has type "str | dict[str, Any]", variable has type "str")
            metrics.append(manager_metrics)
        else:
            raise NotImplementedError(
                f"The process '{self.process}' is not implemented yet."
            )
        metrics += [agent._token_process.get_summary() for agent in self.agents]

        self.usage_metrics = {
            key: sum([m[key] for m in metrics if m is not None]) for key in metrics[0]
        }

        return result

    def kickoff_for_each(
        self, inputs: List[Dict[str, Any]]
    ) -> List[Union[str, Dict[str, Any]]]:
        """Executes the Crew's workflow for each input in the list and aggregates results."""
        results = []

        # Initialize the parent crew's usage metrics
        total_usage_metrics = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
        }

        for input_data in inputs:
            crew = self.copy()

            output = crew.kickoff(inputs=input_data)

            if crew.usage_metrics:
                for key in total_usage_metrics:
                    total_usage_metrics[key] += crew.usage_metrics.get(key, 0)

            results.append(output)

        self.usage_metrics = total_usage_metrics
        return results

    async def kickoff_async(
        self, inputs: Optional[Dict[str, Any]] = {}
    ) -> Union[str, Dict]:
        """Asynchronous kickoff method to start the crew execution."""
        return await asyncio.to_thread(self.kickoff, inputs)

    async def kickoff_for_each_async(self, inputs: List[Dict]) -> List[Any]:
        crew_copies = [self.copy() for _ in inputs]

        async def run_crew(crew, input_data):
            return await crew.kickoff_async(inputs=input_data)

        tasks = [
            asyncio.create_task(run_crew(crew_copies[i], inputs[i]))
            for i in range(len(inputs))
        ]

        results = await asyncio.gather(*tasks)

        total_usage_metrics = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
        }
        for crew in crew_copies:
            if crew.usage_metrics:
                for key in total_usage_metrics:
                    total_usage_metrics[key] += crew.usage_metrics.get(key, 0)

        self.usage_metrics = total_usage_metrics

        return results

    def _run_sequential_process(self) -> Union[str, Dict[str, Any]]:
        """Executes tasks sequentially and returns the final output."""
        task_output = None

        for task in self.tasks:
            if task.agent and task.agent.allow_delegation:
                agents_for_delegation = [
                    agent for agent in self.agents if agent != task.agent
                ]
                if len(self.agents) > 1 and len(agents_for_delegation) > 0:
                    task.tools += task.agent.get_delegation_tools(agents_for_delegation)

            role = task.agent.role if task.agent is not None else "None"
            self._logger.log("debug", f"== Working Agent: {role}", color="bold_purple")
            self._logger.log(
                "info", f"== Starting Task: {task.description}", color="bold_purple"
            )

            if self.output_log_file:
                self._file_handler.log(
                    agent=role, task=task.description, status="started"
                )
            output = task.execute(context=task_output)

            if not task.async_execution:
                task_output = output

            role = task.agent.role if task.agent is not None else "None"
            self._logger.log("debug", f"== [{role}] Task output: {task_output}\n\n")

            if self.output_log_file:
                self._file_handler.log(agent=role, task=task_output, status="completed")

        self._finish_execution(task_output)

        token_usage = self.calculate_usage_metrics()

        return self._format_output(task_output if task_output else "", token_usage)

    def _run_hierarchical_process(
        self,
    ) -> Tuple[Union[str, Dict[str, Any]], Dict[str, Any]]:
        """Creates and assigns a manager agent to make sure the crew completes the tasks."""

        i18n = I18N(prompt_file=self.prompt_file)
        if self.manager_agent is not None:
            self.manager_agent.allow_delegation = True
            manager = self.manager_agent
            if manager.tools is not None and len(manager.tools) > 0:
                raise Exception("Manager agent should not have tools")
            manager.tools = self.manager_agent.get_delegation_tools(self.agents)
        else:
            manager = Agent(
                role=i18n.retrieve("hierarchical_manager_agent", "role"),
                goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
                backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
                tools=AgentTools(agents=self.agents).tools(),
                llm=self.manager_llm,
                verbose=self.verbose,
            )
            self.manager_agent = manager

        task_output = None

        for task in self.tasks:
            self._logger.log("debug", f"Working Agent: {manager.role}")
            self._logger.log("info", f"Starting Task: {task.description}")

            if self.output_log_file:
                self._file_handler.log(
                    agent=manager.role, task=task.description, status="started"
                )

            if task.agent:
                manager.tools = task.agent.get_delegation_tools([task.agent])
            else:
                manager.tools = manager.get_delegation_tools(self.agents)
            task_output = task.execute(
                agent=manager, context=task_output, tools=manager.tools
            )

            self._logger.log("debug", f"[{manager.role}] Task output: {task_output}")
            if self.output_log_file:
                self._file_handler.log(
                    agent=manager.role, task=task_output, status="completed"
                )

        self._finish_execution(task_output)

        token_usage = self.calculate_usage_metrics()

        return self._format_output(
            task_output if task_output else "", token_usage
        ), token_usage

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

    def _format_output(
        self, output: str, token_usage: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Formats the output of the crew execution.
        If full_output is True, then returned data type will be a dictionary else returned outputs are string
        """

        if self.full_output:
            return {
                "final_output": output,
                "tasks_outputs": [task.output for task in self.tasks if task],
                "usage_metrics": token_usage,
            }
        else:
            return output

    def _finish_execution(self, output) -> None:
        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()
        if agentops:
            agentops.end_session(
                end_state="Success", end_state_reason="Finished Execution"
            )
        self._telemetry.end_crew(self, output)

    def calculate_usage_metrics(self) -> Dict[str, int]:
        """Calculates and returns the usage metrics."""
        total_usage_metrics = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
        }

        for agent in self.agents:
            if hasattr(agent, "_token_process"):
                token_sum = agent._token_process.get_summary()
                for key in total_usage_metrics:
                    total_usage_metrics[key] += token_sum.get(key, 0)

        if self.manager_agent and hasattr(self.manager_agent, "_token_process"):
            token_sum = self.manager_agent._token_process.get_summary()
            for key in total_usage_metrics:
                total_usage_metrics[key] += token_sum.get(key, 0)

        return total_usage_metrics

    def __repr__(self):
        return f"Crew(id={self.id}, process={self.process}, number_of_agents={len(self.agents)}, number_of_tasks={len(self.tasks)})"
