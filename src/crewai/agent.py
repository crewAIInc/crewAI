import re
import shutil
import subprocess
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import Field, InstanceOf, PrivateAttr, model_validator

from crewai.agents import CacheHandler
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.utils.knowledge_utils import extract_knowledge_context
from crewai.llm import LLM
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.security import Fingerprint
from crewai.task import Task
from crewai.tools import BaseTool
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities import Converter, Prompts
from crewai.utilities.constants import TRAINED_AGENTS_DATA_FILE, TRAINING_DATA_FILE
from crewai.utilities.converter import generate_model_description
from crewai.utilities.events.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.training_handler import CrewTrainingHandler


class Agent(BaseAgent):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the CrewAgentExecutor class.
            role: The role of the agent.
            goal: The objective of the agent.
            backstory: The backstory of the agent.
            knowledge: The knowledge base of the agent.
            config: Dict representation of agent configuration.
            llm: The language model that will run the agent.
            function_calling_llm: The language model that will handle the tool calling for this agent, it overrides the crew function_calling_llm.
            max_iter: Maximum number of iterations for an agent to execute a task.
            max_rpm: Maximum number of requests per minute for the agent execution to be respected.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
            tools: Tools at agents disposal
            step_callback: Callback to be executed after each step of the agent execution.
            knowledge_sources: Knowledge sources for the agent.
            embedder: Embedder configuration for the agent.
    """

    _times_executed: int = PrivateAttr(default=0)
    max_execution_time: Optional[int] = Field(
        default=None,
        description="Maximum execution time for an agent to execute a task",
    )
    agent_ops_agent_name: str = None  # type: ignore # Incompatible types in assignment (expression has type "None", variable has type "str")
    agent_ops_agent_id: str = None  # type: ignore # Incompatible types in assignment (expression has type "None", variable has type "str")
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    use_system_prompt: Optional[bool] = Field(
        default=True,
        description="Use system prompt for the agent.",
    )
    llm: Union[str, InstanceOf[LLM], Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    function_calling_llm: Optional[Union[str, InstanceOf[LLM], Any]] = Field(
        description="Language model that will run the agent.", default=None
    )
    system_template: Optional[str] = Field(
        default=None, description="System format for the agent."
    )
    prompt_template: Optional[str] = Field(
        default=None, description="Prompt format for the agent."
    )
    response_template: Optional[str] = Field(
        default=None, description="Response format for the agent."
    )
    tools_results: Optional[List[Any]] = Field(
        default=[], description="Results of the tools used by the agent."
    )
    allow_code_execution: Optional[bool] = Field(
        default=False, description="Enable code execution for the agent."
    )
    respect_context_window: bool = Field(
        default=True,
        description="Keep messages under the context window size by summarizing content.",
    )
    max_retry_limit: int = Field(
        default=2,
        description="Maximum number of retries for an agent to execute a task when an error occurs.",
    )
    multimodal: bool = Field(
        default=False,
        description="Whether the agent is multimodal.",
    )
    code_execution_mode: Literal["safe", "unsafe"] = Field(
        default="safe",
        description="Mode for code execution: 'safe' (using Docker) or 'unsafe' (direct execution).",
    )
    embedder: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Embedder configuration for the agent.",
    )

    @model_validator(mode="after")
    def post_init_setup(self):
        self.agent_ops_agent_name = self.role

        self.llm = create_llm(self.llm)
        if self.function_calling_llm and not isinstance(self.function_calling_llm, LLM):
            self.function_calling_llm = create_llm(self.function_calling_llm)

        if not self.agent_executor:
            self._setup_agent_executor()

        if self.allow_code_execution:
            self._validate_docker_installation()

        return self

    def _setup_agent_executor(self):
        if not self.cache_handler:
            self.cache_handler = CacheHandler()
        self.set_cache_handler(self.cache_handler)

    def set_knowledge(self, crew_embedder: Optional[Dict[str, Any]] = None):
        try:
            if self.embedder is None and crew_embedder:
                self.embedder = crew_embedder

            if self.knowledge_sources:
                full_pattern = re.compile(r"[^a-zA-Z0-9\-_\r\n]|(\.\.)")
                knowledge_agent_name = f"{re.sub(full_pattern, '_', self.role)}"
                if isinstance(self.knowledge_sources, list) and all(
                    isinstance(k, BaseKnowledgeSource) for k in self.knowledge_sources
                ):
                    self.knowledge = Knowledge(
                        sources=self.knowledge_sources,
                        embedder=self.embedder,
                        collection_name=knowledge_agent_name,
                        storage=self.knowledge_storage or None,
                    )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid Knowledge Configuration: {str(e)}")

    def execute_task(
        self,
        task: Task,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        if self.tools_handler:
            self.tools_handler.last_used_tool = {}  # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")

        task_prompt = task.prompt()

        # If the task requires output in JSON or Pydantic format,
        # append specific instructions to the task prompt to ensure
        # that the final answer does not include any code block markers
        if task.output_json or task.output_pydantic:
            # Generate the schema based on the output format
            if task.output_json:
                # schema = json.dumps(task.output_json, indent=2)
                schema = generate_model_description(task.output_json)
                task_prompt += "\n" + self.i18n.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

            elif task.output_pydantic:
                schema = generate_model_description(task.output_pydantic)
                task_prompt += "\n" + self.i18n.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            contextual_memory = ContextualMemory(
                self.crew.memory_config,
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
                self.crew._user_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        if self.knowledge:
            agent_knowledge_snippets = self.knowledge.query([task.prompt()])
            if agent_knowledge_snippets:
                agent_knowledge_context = extract_knowledge_context(
                    agent_knowledge_snippets
                )
                if agent_knowledge_context:
                    task_prompt += agent_knowledge_context

        if self.crew:
            knowledge_snippets = self.crew.query_knowledge([task.prompt()])
            if knowledge_snippets:
                crew_knowledge_context = extract_knowledge_context(knowledge_snippets)
                if crew_knowledge_context:
                    task_prompt += crew_knowledge_context

        tools = tools or self.tools or []
        self.create_agent_executor(tools=tools, task=task)

        if self.crew and self.crew._train:
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

        try:
            crewai_event_bus.emit(
                self,
                event=AgentExecutionStartedEvent(
                    agent=self,
                    tools=self.tools,
                    task_prompt=task_prompt,
                    task=task,
                ),
            )
            result = self.agent_executor.invoke(
                {
                    "input": task_prompt,
                    "tool_names": self.agent_executor.tools_names,
                    "tools": self.agent_executor.tools_description,
                    "ask_for_human_input": task.human_input,
                }
            )["output"]
        except Exception as e:
            if e.__class__.__module__.startswith("litellm"):
                # Do not retry on litellm errors
                crewai_event_bus.emit(
                    self,
                    event=AgentExecutionErrorEvent(
                        agent=self,
                        task=task,
                        error=str(e),
                    ),
                )
                raise e
            self._times_executed += 1
            if self._times_executed > self.max_retry_limit:
                crewai_event_bus.emit(
                    self,
                    event=AgentExecutionErrorEvent(
                        agent=self,
                        task=task,
                        error=str(e),
                    ),
                )
                raise e
            result = self.execute_task(task, context, tools)

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        # If there was any tool in self.tools_results that had result_as_answer
        # set to True, return the results of the last tool that had
        # result_as_answer set to True
        for tool_result in self.tools_results:  # type: ignore # Item "None" of "list[Any] | None" has no attribute "__iter__" (not iterable)
            if tool_result.get("result_as_answer", False):
                result = tool_result["result"]
        crewai_event_bus.emit(
            self,
            event=AgentExecutionCompletedEvent(agent=self, task=task, output=result),
        )
        return result

    def create_agent_executor(
        self, tools: Optional[List[BaseTool]] = None, task=None
    ) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        tools = tools or self.tools or []
        parsed_tools = self._parse_tools(tools)

        prompt = Prompts(
            agent=self,
            tools=tools,
            i18n=self.i18n,
            use_system_prompt=self.use_system_prompt,
            system_template=self.system_template,
            prompt_template=self.prompt_template,
            response_template=self.response_template,
        ).task_execution()

        stop_words = [self.i18n.slice("observation")]

        if self.response_template:
            stop_words.append(
                self.response_template.split("{{ .Response }}")[1].strip()
            )

        self.agent_executor = CrewAgentExecutor(
            llm=self.llm,
            task=task,
            agent=self,
            crew=self.crew,
            tools=parsed_tools,
            prompt=prompt,
            original_tools=tools,
            stop_words=stop_words,
            max_iter=self.max_iter,
            tools_handler=self.tools_handler,
            tools_names=self.__tools_names(parsed_tools),
            tools_description=self._render_text_description_and_args(parsed_tools),
            step_callback=self.step_callback,
            function_calling_llm=self.function_calling_llm,
            respect_context_window=self.respect_context_window,
            request_within_rpm_limit=(
                self._rpm_controller.check_or_wait if self._rpm_controller else None
            ),
            callbacks=[TokenCalcHandler(self._token_process)],
        )

    def get_delegation_tools(self, agents: List[BaseAgent]):
        agent_tools = AgentTools(agents=agents)
        tools = agent_tools.tools()
        return tools

    def get_multimodal_tools(self) -> Sequence[BaseTool]:
        from crewai.tools.agent_tools.add_image_tool import AddImageTool

        return [AddImageTool()]

    def get_code_execution_tools(self):
        try:
            from crewai_tools import CodeInterpreterTool  # type: ignore

            # Set the unsafe_mode based on the code_execution_mode attribute
            unsafe_mode = self.code_execution_mode == "unsafe"
            return [CodeInterpreterTool(unsafe_mode=unsafe_mode)]
        except ModuleNotFoundError:
            self._logger.log(
                "info", "Coding tools not available. Install crewai_tools. "
            )

    def get_output_converter(self, llm, text, model, instructions):
        return Converter(llm=llm, text=text, model=model, instructions=instructions)

    def _parse_tools(self, tools: List[Any]) -> List[Any]:  # type: ignore
        """Parse tools to be used for the task."""
        tools_list = []
        try:
            # tentatively try to import from crewai_tools import BaseTool as CrewAITool
            from crewai.tools import BaseTool as CrewAITool

            for tool in tools:
                if isinstance(tool, CrewAITool):
                    tools_list.append(tool.to_structured_tool())
                else:
                    tools_list.append(tool)
        except ModuleNotFoundError:
            tools_list = []
            for tool in tools:
                tools_list.append(tool)

        return tools_list

    def _training_handler(self, task_prompt: str) -> str:
        """Handle training data for the agent task prompt to improve output on Training."""
        if data := CrewTrainingHandler(TRAINING_DATA_FILE).load():
            agent_id = str(self.id)

            if data.get(agent_id):
                human_feedbacks = [
                    i["human_feedback"] for i in data.get(agent_id, {}).values()
                ]
                task_prompt += (
                    "\n\nYou MUST follow these instructions: \n "
                    + "\n - ".join(human_feedbacks)
                )

        return task_prompt

    def _use_trained_data(self, task_prompt: str) -> str:
        """Use trained data for the agent task prompt to improve output."""
        if data := CrewTrainingHandler(TRAINED_AGENTS_DATA_FILE).load():
            if trained_data_output := data.get(self.role):
                task_prompt += (
                    "\n\nYou MUST follow these instructions: \n - "
                    + "\n - ".join(trained_data_output["suggestions"])
                )
        return task_prompt

    def _render_text_description(self, tools: List[Any]) -> str:
        """Render the tool name and description in plain text.

        Output will be in the format of:

        .. code-block:: markdown

            search: This tool is used for search
            calculator: This tool is used for math
        """
        description = "\n".join(
            [
                f"Tool name: {tool.name}\nTool description:\n{tool.description}"
                for tool in tools
            ]
        )

        return description

    def _render_text_description_and_args(self, tools: List[BaseTool]) -> str:
        """Render the tool name, description, and args in plain text.

            Output will be in the format of:

            .. code-block:: markdown

            search: This tool is used for search, args: {"query": {"type": "string"}}
            calculator: This tool is used for math, \
            args: {"expression": {"type": "string"}}
        """
        tool_strings = []
        for tool in tools:
            tool_strings.append(tool.description)

        return "\n".join(tool_strings)

    def _validate_docker_installation(self) -> None:
        """Check if Docker is installed and running."""
        if not shutil.which("docker"):
            raise RuntimeError(
                f"Docker is not installed. Please install Docker to use code execution with agent: {self.role}"
            )

        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"Docker is not running. Please start Docker to use code execution with agent: {self.role}"
            )

    @staticmethod
    def __tools_names(tools) -> str:
        return ", ".join([t.name for t in tools])

    def __repr__(self):
        return f"Agent(role={self.role}, goal={self.goal}, backstory={self.backstory})"

    @property
    def fingerprint(self) -> Fingerprint:
        """
        Get the agent's fingerprint.

        Returns:
            Fingerprint: The agent's fingerprint
        """
        return self.security_config.fingerprint
