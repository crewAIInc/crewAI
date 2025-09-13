"""OpenAI agents adapter for CrewAI integration.

This module contains the OpenAIAgentAdapter class that integrates OpenAI Assistants
with CrewAI's agent system, providing tool integration and structured output support.
"""

from typing import Any, cast

from pydantic import ConfigDict, Field, PrivateAttr
from typing_extensions import Unpack

from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai.agents.agent_adapters.openai_agents.openai_agent_tool_adapter import (
    OpenAIAgentToolAdapter,
)
from crewai.agents.agent_adapters.openai_agents.protocols import (
    AgentKwargs,
    OpenAIAgentsModule,
)
from crewai.agents.agent_adapters.openai_agents.protocols import (
    OpenAIAgent as OpenAIAgentProtocol,
)
from crewai.agents.agent_adapters.openai_agents.structured_output_converter import (
    OpenAIConverterAdapter,
)
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.tools import BaseTool
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities import Logger
from crewai.utilities.import_utils import require

openai_agents_module = cast(
    OpenAIAgentsModule,
    require(
        "agents",
        purpose="OpenAI agents functionality",
    ),
)
OpenAIAgent = openai_agents_module.Agent
Runner = openai_agents_module.Runner
enable_verbose_stdout_logging = openai_agents_module.enable_verbose_stdout_logging


class OpenAIAgentAdapter(BaseAgentAdapter):
    """Adapter for OpenAI Assistants.

    Integrates OpenAI Assistants API with CrewAI's agent system, providing
    tool configuration, structured output handling, and task execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _openai_agent: OpenAIAgentProtocol = PrivateAttr()
    _logger: Logger = PrivateAttr(default_factory=Logger)
    _active_thread: str | None = PrivateAttr(default=None)
    function_calling_llm: Any = Field(default=None)
    step_callback: Any = Field(default=None)
    _tool_adapter: OpenAIAgentToolAdapter = PrivateAttr()
    _converter_adapter: OpenAIConverterAdapter = PrivateAttr()

    def __init__(
        self,
        **kwargs: Unpack[AgentKwargs],
    ) -> None:
        """Initialize the OpenAI agent adapter.

        Args:
            **kwargs: All initialization arguments including role, goal, backstory,
                     model, tools, and agent_config.

        Raises:
            ImportError: If OpenAI agent dependencies are not installed.
        """
        self.llm = kwargs.pop("model", "gpt-4o-mini")
        super().__init__(**kwargs)
        self._tool_adapter = OpenAIAgentToolAdapter(tools=kwargs.get("tools"))
        self._converter_adapter = OpenAIConverterAdapter(agent_adapter=self)

    def _build_system_prompt(self) -> str:
        """Build a system prompt for the OpenAI agent.

        Creates a prompt containing the agent's role, goal, and backstory,
        then enhances it with structured output instructions if needed.

        Returns:
            The complete system prompt string.
        """
        base_prompt = f"""
            You are {self.role}.

            Your goal is: {self.goal}

            Your backstory: {self.backstory}

            When working on tasks, think step-by-step and use the available tools when necessary.
        """
        return self._converter_adapter.enhance_system_prompt(base_prompt)

    def execute_task(
        self,
        task: Any,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        """Execute a task using the OpenAI Assistant.

        Configures the assistant, processes the task, and handles event emission
        for execution tracking.

        Args:
            task: The task object to execute.
            context: Optional context information for the task.
            tools: Optional additional tools for this execution.

        Returns:
            The final answer from the task execution.

        Raises:
            Exception: If task execution fails.
        """
        self._converter_adapter.configure_structured_output(task)
        self.create_agent_executor(tools)

        if self.verbose:
            enable_verbose_stdout_logging()

        try:
            task_prompt: str = task.prompt()
            if context:
                task_prompt = self.i18n.slice("task_with_context").format(
                    task=task_prompt, context=context
                )
            crewai_event_bus.emit(
                self,
                event=AgentExecutionStartedEvent(
                    agent=self,
                    tools=self.tools,
                    task_prompt=task_prompt,
                    task=task,
                ),
            )
            result: Any = self.agent_executor.run_sync(self._openai_agent, task_prompt)
            final_answer: str = self.handle_execution_result(result)
            crewai_event_bus.emit(
                self,
                event=AgentExecutionCompletedEvent(
                    agent=self, task=task, output=final_answer
                ),
            )
            return final_answer

        except Exception as e:
            self._logger.log("error", f"Error executing OpenAI task: {e!s}")
            crewai_event_bus.emit(
                self,
                event=AgentExecutionErrorEvent(
                    agent=self,
                    task=task,
                    error=str(e),
                ),
            )
            raise

    def create_agent_executor(self, tools: list[BaseTool] | None = None) -> None:
        """Configure the OpenAI agent for execution.

        While OpenAI handles execution differently through Runner,
        this method sets up tools and agent configuration.

        Args:
            tools: Optional tools to configure for the agent.

        Notes:
            TODO: Properly type agent_executor in BaseAgent to avoid type issues
            when assigning Runner class to this attribute.
        """
        all_tools: list[BaseTool] = list(self.tools or []) + list(tools or [])

        instructions: str = self._build_system_prompt()
        self._openai_agent = OpenAIAgent(
            name=self.role,
            instructions=instructions,
            model=self.llm,
            **self._agent_config or {},
        )

        if all_tools:
            self.configure_tools(all_tools)

        self.agent_executor = Runner

    def configure_tools(self, tools: list[BaseTool] | None = None) -> None:
        """Configure tools for the OpenAI Assistant.

        Args:
            tools: Optional tools to configure for the assistant.
        """
        if tools:
            self._tool_adapter.configure_tools(tools)
            if self._tool_adapter.converted_tools:
                self._openai_agent.tools = self._tool_adapter.converted_tools

    def handle_execution_result(self, result: Any) -> str:
        """Process OpenAI Assistant execution result.

        Converts any structured output to a string through the converter adapter.

        Args:
            result: The execution result from the OpenAI assistant.

        Returns:
            Processed result as a string.
        """
        return self._converter_adapter.post_process_result(result.final_output)

    def get_delegation_tools(self, agents: list[BaseAgent]) -> list[BaseTool]:
        """Implement delegation tools support.

        Creates delegation tools that allow this agent to delegate tasks to other agents.

        Args:
            agents: List of agents available for delegation.

        Returns:
            List of delegation tools.
        """
        agent_tools: AgentTools = AgentTools(agents=agents)
        return agent_tools.tools()

    def configure_structured_output(self, task: Any) -> None:
        """Configure the structured output for the specific agent implementation.

        Args:
            task: The task object containing output format specifications.
        """
        self._converter_adapter.configure_structured_output(task)
